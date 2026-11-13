"""Reuse complete evaluation batches from a stopped checkpoint evaluator."""
import json


def load_batches(path, benchmarks, samples, seed, step):
    batches = {}
    with path.open() as stream:
        for line in stream:
            # A stopped writer may leave a final partial record.
            if not line.endswith('\n'):
                break
            row = json.loads(line)
            name, problem, sample = (row[key] for key in
                                     ('benchmark', 'problem_index', 'sample'))
            if (name not in benchmarks or not 0 <= problem < len(benchmarks[name])
                    or not 0 <= sample < samples or row['seed'] != seed
                    or row['checkpoint_step'] != step or type(row['correct']) is not bool):
                raise ValueError('Saved evaluation row differs from the benchmark contract')
            offset = problem // 4 * 4
            batch = batches.setdefault((name, offset), {})
            if (problem, sample) in batch:
                raise ValueError('Duplicate saved evaluation sample')
            batch[problem, sample] = row
    return {key: [batch[index] for index in sorted(batch)]
            for key, batch in batches.items()
            if len(batch) == min(4, len(benchmarks[key[0]]) - key[1]) * samples}
