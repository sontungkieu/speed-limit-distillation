"""Independent sampled-token Semi-OPD/OPD reconstruction in the pinned v070 runtime."""


import argparse


from contextlib import nullcontext


import hashlib


import json


import os


from pathlib import Path


import random


import re


import time


os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open("a") as output:
        output.write(json.dumps(value, allow_nan=False) + "\n")


def token_logprobs(model, ids, prompt_length, torch, gradients=False):
    context = nullcontext() if gradients else torch.no_grad()
    with context, torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids, use_cache=False).logits
        selected = logits[:, prompt_length - 1:-1, :]
        labels = ids[:, prompt_length:]
        # Keep the log-normalizer stable without allocating FP32 logits for a whole sequence.
        parts = []
        for start in range(0, selected.shape[1], 128):
            chunk = selected[:, start:start + 128].float()
            answer = labels[:, start:start + 128]
            parts.append(chunk.gather(-1, answer.unsqueeze(-1)).squeeze(-1) - chunk.logsumexp(-1))
        return torch.cat(parts, dim=1).squeeze(0)


def check_objective(torch):
    values = torch.tensor([0.2, -0.4, 0.8], dtype=torch.float64, requires_grad=True)
    teacher = torch.tensor([-0.9, -1.7, -0.5], dtype=torch.float64)
    student = values.log_softmax(0)
    advantage = (teacher - student).detach()
    loss = -(advantage * student).sum()
    gradient = torch.autograd.grad(loss, values)[0]
    reference = -advantage + values.softmax(0) * advantage.sum()
    error = (gradient - reference).abs().max().item()
    if error > 1e-10 or advantage.requires_grad:
        raise RuntimeError("Sampled-token detached-advantage gradient check failed")
    return {"gradient_max_error": error, "advantage_detached": True}


def boxed_answer(text):
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if matches:
        start = matches[-1].end()
        depth = 1
        for index in range(start, len(text)):
            depth += (text[index] == "{") - (text[index] == "}")
            if depth == 0:
                return text[start:index].strip()
    return None


def grade(text, gold):
    prediction = boxed_answer(text)
    if prediction is None:
        return False, None, "missing_boxed_answer"
    try:
        from math_verify import parse, verify
        correct = bool(verify(parse("$" + str(gold) + "$"), parse("$" + prediction + "$")))
    except Exception as error:
        raise RuntimeError("Grader failure") from error
    return correct, prediction, "math_verify_boxed"


class Experiment:
    def render(self, text, training=False):
        result = self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        if training and len(result) > 2048:
            raise RuntimeError("Training prompt exceeds 2048 tokens")
        return result


    def refit(self, step):
        if self.last_refit == step:
            return
        weights = {name: parameter.detach().to(device="cpu", dtype=self.torch.bfloat16) for name, parameter in self.student.named_parameters()}
        temporary = self.snapshot.with_suffix(".partial")
        self.torch.save(weights, temporary)
        temporary.replace(self.snapshot)
        receipts = self.engine.collective_rpc("load_student_snapshot", args=(str(self.snapshot),))
        if not receipts or not all(value["loaded_tensors"] > 0 for value in receipts):
            raise RuntimeError("vLLM weight refit returned no loaded tensors")
        for receipt in receipts:
            for name, actual in receipt["probes"].items():
                expected = weights[name].flatten()[:16].float().tolist()
                if actual != expected:
                    raise RuntimeError("vLLM weight refit probe mismatch: " + name)
        self.last_refit = step
        append_json(self.run / "refits.jsonl", {"step": step, "loaded_tensors": receipts, "time": time.time()})


    def generate(self, prompt_ids, cap, seed, samples=1):
        parameters = self.SamplingParams(n=samples, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, repetition_penalty=1.0, max_tokens=cap, stop_token_ids=[151645, 151643], seed=seed)
        return self.engine.generate([{"prompt_token_ids": value} for value in prompt_ids], parameters, use_tqdm=False)


    def evaluate(self, step):
        self.refit(step)
        started = time.time()
        summary = {}
        samples = 1 if self.args.canary else (8 if self.args.dev else 32)
        cap = 64 if self.args.canary else (16384 if self.args.dev else 32768)
        saved = {}
        if getattr(self.args, 'resume_eval', None):
            from .eval_resume import load_batches
            saved = load_batches(self.args.resume_eval, self.benchmarks, samples,
                                 self.args.seed, step)
            print('EVAL_REUSED_SAMPLES', sum(map(len, saved.values())), flush=True)
        for name, rows in self.benchmarks.items():
            correct = total = 0
            for offset in range(0, len(rows), 4):
                previous = saved.get((name, offset))
                if previous:
                    for row in previous:
                        append_json(self.run / f"eval-step-{step:04d}.jsonl", row)
                        correct += int(row['correct'])
                        total += 1
                    continue
                batch = rows[offset:offset + 4]
                prompts = []
                for row in batch:
                    problem = row.get("problem", row.get("Problem", row.get("question")))
                    if problem is None:
                        raise RuntimeError("Unknown benchmark question schema: " + name)
                    prompts.append(self.render(problem + "\nPlease reason step by step, and put your final answer within \\boxed{}."))
                outputs = self.generate(prompts, cap, self.args.seed * 1000000 + 700000 + offset, samples)
                for position, (row, output) in enumerate(zip(batch, outputs)):
                    gold = row.get("answer", row.get("Answer", row.get("solution")))
                    if gold is None:
                        raise RuntimeError("Unknown benchmark answer schema: " + name)
                    for sample, answer in enumerate(output.outputs):
                        text = self.tokenizer.decode(answer.token_ids, skip_special_tokens=True)
                        passed, extracted, grader = grade(text, gold)
                        correct += int(passed)
                        total += 1
                        append_json(self.run / f"eval-step-{step:04d}.jsonl", {"benchmark": name, "problem_index": offset + position, "sample": sample, "seed": self.args.seed, "checkpoint_step": step, "gold": str(gold), "prediction": extracted, "correct": passed, "grader": grader, "response": text, "response_tokens": len(answer.token_ids), "finish_reason": answer.finish_reason})
            summary[name] = {"correct": correct, "total": total, "accuracy": correct / total}
        values = [item["accuracy"] for item in summary.values()]
        write_json(self.run / f"eval-step-{step:04d}.json", {"step": step, "benchmarks": summary, "mean_accuracy": sum(values) / len(values), "samples_per_problem": samples, "elapsed_seconds": time.time() - started, "prompt_wrapper": "problem + step-by-step boxed-answer instruction", "grader": "math_verify on last balanced boxed answer; missing box scored incorrect"})
        print("EVALUATION_COMPLETE", step, summary, flush=True)


    def checkpoint(self, step):
        target = self.run / f"checkpoint-{step:04d}"
        target.mkdir()
        self.student.save_pretrained(target, safe_serialization=True)
        self.tokenizer.save_pretrained(target)
        self.torch.save({"optimizer": self.optimizer.state_dict(), "torch_rng": self.torch.get_rng_state(), "cuda_rng": self.torch.cuda.get_rng_state(), "python_rng": random.getstate(), "step": step, "prompt_cursor": step * self.batch_size}, target / "training_state.pt")
        write_json(target / "receipt.json", {"step": step, "run_id": self.args.run_id, "time": time.time()})


