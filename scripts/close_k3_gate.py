"""Record the tested LoRA SGD rate and Gate 1 recommendation when evals finish."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_text())


def decide(summary, evidence):
    baseline = summary['baseline']
    rates = summary['means_by_lr']
    checks = {}
    for rate, metrics in rates.items():
        checks[rate] = {
            'heldout_kl_improves': metrics['heldout_kl'] < baseline['heldout_kl'],
            'fresh_mean_length_does_not_increase': metrics['fresh_mean_tokens'] <= baseline['fresh_mean_tokens'],
            'dev_accuracy_does_not_decrease': metrics['dev_accuracy'] >= baseline['dev_accuracy'],
        }
    eligible = [float(rate) for rate, result in checks.items() if all(result.values())]
    winner = max(eligible, default=None)
    if winner == 1e-3:
        recommendation = 'pivot_learning_rate_and_revisit_fixed_spectrum'
        reason = 'The faster SGD rate cleared the point-estimate quality checks. Its fixed step-zero forecast missed tau50 by more than 2x on S1.'
    elif winner == 3e-4:
        recommendation = 'hold_phase2'
        reason = 'The lower rate cleared the quality checks. The 300-step SGD window did not identify tau50 at this rate.'
    else:
        recommendation = 'hold_phase2'
        reason = 'Neither tested SGD rate cleared all quality checks against the initial student.'
    pair = [{'set_index': idx, **{
        metric: summary['cases'][f'eval-S{idx}-1e-3'][metric] - summary['cases'][f'eval-S{idx}-3e-4'][metric]
        for metric in ('heldout_kl', 'dev_accuracy', 'fresh_kl', 'fresh_mean_tokens')
    }} for idx in range(3)]
    return {'tested_lr_star': winner, 'decision': recommendation, 'reason': reason,
            'quality_rule': 'Strict point estimates of equal-set mean against the same initial student, with no tolerance margin or claims across training seeds.',
            'quality_checks': checks, 'means_by_lr': rates, 'baseline': baseline,
            'high_minus_low_by_set': pair, 'cases': summary['cases'], 'prior_evidence': evidence}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    evidence = read(args.evidence)
    if args.detach:
        args.output.mkdir(exist_ok=False)
        argv = [sys.executable, '-B', str(Path(__file__).resolve()),
                *(arg for arg in sys.argv[1:] if arg != '--detach')]
        with (args.output / 'controller.log').open('xb') as log:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {'pid': child.pid, 'deadline': args.deadline, 'summary': str(args.summary)}
        (args.output / 'launch.json').write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt), flush=True)
        return
    while not args.summary.is_file():
        if time.time() >= args.deadline:
            raise RuntimeError(f'Evaluation summary did not arrive before the deadline: {args.summary}')
        time.sleep(30)
    decision = decide(read(args.summary), evidence)
    decision.update(summary=str(args.summary), decided_at=time.time())
    pending = args.output / 'decision.partial'
    pending.write_text(json.dumps(decision, indent=2, allow_nan=False))
    pending.replace(args.output / 'decision.json')


if __name__ == '__main__':
    main()
