#!/usr/bin/env python3
"""
Analysis script to compare PlaNet and RSSM training results
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict


def parse_metrics_file(filepath):
    """Parse metrics from jsonl file"""
    metrics = []
    if not os.path.exists(filepath):
        return metrics

    with open(filepath, 'r') as f:
        for line in f:
            try:
                metrics.append(json.loads(line))
            except:
                continue
    return metrics


def summarize_run(log_dir):
    """Summarize a single training run"""
    metrics_file = os.path.join(log_dir, 'metrics.jsonl')

    if not os.path.exists(metrics_file):
        return None

    metrics = parse_metrics_file(metrics_file)

    if not metrics:
        return None

    summary = {
        'total_steps': 0,
        'final_reward': None,
        'max_reward': -float('inf'),
        'avg_model_loss': [],
        'training_time': 0,
    }

    for m in metrics:
        step = m.get('step', 0)
        summary['total_steps'] = max(summary['total_steps'], step)

        if 'eval_return' in m:
            reward = m['eval_return']
            summary['final_reward'] = reward
            summary['max_reward'] = max(summary['max_reward'], reward)

        if 'model_loss_mean' in m:
            summary['avg_model_loss'].append(m['model_loss_mean'])

        if 'time' in m:
            summary['training_time'] = m['time']

    if summary['avg_model_loss']:
        summary['avg_model_loss'] = sum(summary['avg_model_loss']) / len(summary['avg_model_loss'])
    else:
        summary['avg_model_loss'] = None

    return summary


def compare_runs(comparison_dir):
    """Compare all runs in a comparison directory"""
    comparison_dir = Path(comparison_dir)

    if not comparison_dir.exists():
        print(f"Error: Directory {comparison_dir} does not exist")
        return

    print("=" * 80)
    print(f"Analyzing comparison results from: {comparison_dir}")
    print("=" * 80)
    print()

    results = {}

    for run_dir in comparison_dir.iterdir():
        if not run_dir.is_dir():
            continue

        run_name = run_dir.name
        print(f"Processing: {run_name}...")

        summary = summarize_run(run_dir)
        if summary:
            results[run_name] = summary
            print(f"  ✓ Found {summary['total_steps']} steps")
        else:
            print(f"  ⚠ No metrics found")

    print()
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print()

    if not results:
        print("No results to compare!")
        return

    # Separate RSSM and PlaNet results
    rssm_results = {k: v for k, v in results.items() if 'rssm' in k.lower()}
    planet_results = {k: v for k, v in results.items() if 'planet' in k.lower()}

    # Print detailed comparison
    print(f"{'Model':<20} {'Steps':<12} {'Final Reward':<15} {'Max Reward':<15} {'Model Loss':<15} {'Time (h)':<10}")
    print("-" * 100)

    for name, summary in sorted(results.items()):
        steps = f"{summary['total_steps']:,}"
        final_reward = f"{summary['final_reward']:.2f}" if summary['final_reward'] is not None else "N/A"
        max_reward = f"{summary['max_reward']:.2f}" if summary['max_reward'] != -float('inf') else "N/A"
        model_loss = f"{summary['avg_model_loss']:.4f}" if summary['avg_model_loss'] is not None else "N/A"
        time_hours = f"{summary['training_time'] / 3600:.2f}" if summary['training_time'] > 0 else "N/A"

        print(f"{name:<20} {steps:<12} {final_reward:<15} {max_reward:<15} {model_loss:<15} {time_hours:<10}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    # Compare same sizes
    sizes = ['small', 'medium', 'large']

    for size in sizes:
        rssm_key = f"rssm_{size}"
        planet_key = f"planet_{size}"

        if rssm_key not in results or planet_key not in results:
            continue

        rssm = results[rssm_key]
        planet = results[planet_key]

        print(f"Comparison: {size.upper()}")
        print(f"  RSSM   - Final Reward: {rssm.get('final_reward', 'N/A')}, Training Time: {rssm.get('training_time', 0) / 3600:.2f}h")
        print(f"  PlaNet - Final Reward: {planet.get('final_reward', 'N/A')}, Training Time: {planet.get('training_time', 0) / 3600:.2f}h")

        if rssm.get('final_reward') is not None and planet.get('final_reward') is not None:
            reward_diff = ((planet['final_reward'] - rssm['final_reward']) / abs(rssm['final_reward'])) * 100
            print(f"  → Reward difference: {reward_diff:+.2f}%")

        if rssm.get('training_time', 0) > 0 and planet.get('training_time', 0) > 0:
            time_diff = ((planet['training_time'] - rssm['training_time']) / rssm['training_time']) * 100
            speedup = rssm['training_time'] / planet['training_time']
            print(f"  → Time difference: {time_diff:+.2f}% (PlaNet is {speedup:.2f}x)")

        print()

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Analyze PlaNet vs RSSM comparison results')
    parser.add_argument('comparison_dir', type=str, help='Directory containing comparison runs')
    parser.add_argument('--export', type=str, help='Export results to JSON file')

    args = parser.parse_args()

    compare_runs(args.comparison_dir)

    if args.export:
        print(f"Exporting results to {args.export}...")
        # TODO: Implement JSON export
        print("Export not yet implemented")


if __name__ == '__main__':
    main()
