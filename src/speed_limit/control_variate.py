"""Detached vOPD baselines for repeated, cached student rollouts."""
import torch


@torch.no_grad()
def baseline_kl(student, teacher, top_k=None):
    if student.shape != teacher.shape:
        raise ValueError('Student and teacher vocabulary logits must align')
    if top_k is not None:
        student, ids = student.topk(top_k, dim=-1)
        teacher = teacher.gather(-1, ids)
    log_student = student.float().log_softmax(-1)
    log_teacher = teacher.float().log_softmax(-1)
    return (log_student.exp() * (log_student - log_teacher)).sum(-1)


def sampled_objective(student, teacher, labels, teacher_sampled, top_k=None):
    logits = student.float()
    current = logits.gather(-1, labels[:, None]).squeeze(-1) - logits.logsumexp(-1)
    baseline = baseline_kl(student, teacher, top_k)
    advantage = (teacher_sampled - current.detach() + baseline).detach()
    return -(advantage * current).sum(), baseline.sum()


def rollout_loss(student, teacher, record, top_k=None):
    prompt = record['prompt']
    response = record['response']
    if not prompt or not response:
        raise ValueError('Empty prompt or response')
    device = next(student.parameters()).device
    ids = torch.tensor([prompt + response], device=device, dtype=torch.long)
    start = len(prompt) - 1
    target_logp = record['teacher_logprobs'].to(device, dtype=torch.float32)
    if target_logp.shape != (len(response),):
        raise ValueError('Cached teacher token log probabilities do not align')
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        with torch.no_grad():
            teacher_logits = teacher(input_ids=ids, use_cache=False).logits[0, start:-1]
        student_logits = student(input_ids=ids, use_cache=False).logits[0, start:-1]
        losses = []
        baselines = []
        for offset in range(0, len(response), 128):
            chunk = slice(offset, offset + 128)
            labels = ids[0, len(prompt):][chunk]
            loss, baseline = sampled_objective(
                student_logits[chunk], teacher_logits[chunk], labels, target_logp[chunk], top_k)
            losses.append(loss)
            baselines.append(baseline)
    return torch.stack(losses).sum(), torch.stack(baselines).sum()
