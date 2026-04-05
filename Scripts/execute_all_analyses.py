"""
execute_all_analyses.py
=======================
Master script: runs statistics, bifurcation, and information analyses.

Usage:  python execute_all_analyses.py
        python execute_all_analyses.py --results-dir ../results
"""

import sys
from pathlib import Path
import traceback

SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

from analysis_statistics import StatisticsAnalyzer
from bifurcation_analysis import BifurcationAnalyzer
from information_analysis import InformationAnalyzer


def run_all(results_dir: str = '../results') -> dict:
    out = {'statistics': None, 'bifurcation': None, 'information': None, 'errors': []}

    print('\n' + '='*70)
    print('CERVO PROJECT — FULL ANALYSIS PIPELINE')
    print('='*70)

    # 1. Statistics
    print('\n' + '#'*70)
    print('# 1. STATISTICS')
    print('#'*70)
    try:
        a = StatisticsAnalyzer(results_dir=results_dir)
        out['statistics'] = a.generate_full_report(output_dir=results_dir)
    except Exception as e:
        out['errors'].append(f'Statistics: {e}')
        traceback.print_exc()

    # 2. Bifurcation
    print('\n' + '#'*70)
    print('# 2. BIFURCATION')
    print('#'*70)
    try:
        b = BifurcationAnalyzer(results_dir=results_dir)
        out['bifurcation'] = b.generate_report(output_dir=results_dir)
    except Exception as e:
        out['errors'].append(f'Bifurcation: {e}')
        traceback.print_exc()

    # 3. Information
    print('\n' + '#'*70)
    print('# 3. INFORMATION')
    print('#'*70)
    try:
        c = InformationAnalyzer(results_dir=results_dir)
        out['information'] = c.generate_report(output_dir=results_dir)
    except Exception as e:
        out['errors'].append(f'Information: {e}')
        traceback.print_exc()

    # Summary
    ok = sum(1 for v in [out['statistics'], out['bifurcation'], out['information']] if v)
    print(f'\n{"="*70}')
    print(f'DONE: {ok}/3 analyses completed, {len(out["errors"])} errors')
    if out['errors']:
        for err in out['errors']:
            print(f'  ✗ {err}')
    else:
        print('  ✓ ALL ANALYSES PASSED')
    print(f'{"="*70}')

    return out


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', default='../results')
    args = p.parse_args()
    result = run_all(results_dir=args.results_dir)
    sys.exit(1 if result['errors'] else 0)
