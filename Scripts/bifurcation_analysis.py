"""
bifurcation_analysis.py
=======================
Bifurcation boundary detection and topology comparison.

Reads .npz files with keys: P, K3, oscillation_map, osc_fraction.

Usage
-----
    from bifurcation_analysis import BifurcationAnalyzer
    analyzer = BifurcationAnalyzer(results_dir='../results')
    analyzer.generate_report(output_dir='../results')
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, Optional


class BifurcationAnalyzer:

    MODELS = {
        # 3-node
        '3N 2nd Additive':  'Pairwise_noise',
        '3N 2nd Diffusive': 'Pairwise_diffusive_noise',
        '3N 3rd Additive':  'Coupling_noise',
        '3N 3rd Diffusive': 'Coupling_diffusive_noise',
        # 10-node ring
        'Ring 2nd Additive': '10n_ring_sw_pairwise_noise',
        'Ring 3rd Additive': '10n_ring_sw_coupling_noise',
        # 10-node random
        'Random 2nd Additive': '10n_random_pairwise_noise',
        'Random 3rd Additive': '10n_random_coupling_noise',
    }

    def __init__(self, results_dir: str = '../results'):
        self.results_dir = Path(results_dir)
        self.data: Dict[str, dict] = {}

    def load(self):
        for name, tag in self.MODELS.items():
            path = self.results_dir / f'metrics_{tag}.npz'
            if path.exists():
                self.data[name] = dict(np.load(path))
                print(f'  ✓ {name}')

    def _osc(self, d: dict) -> np.ndarray:
        return d.get('osc_fraction', d.get('oscillation_map')).astype(float)

    def extract_boundary(self, d: dict, name: str) -> Optional[dict]:
        """Extract Hopf bifurcation boundary via linear interpolation."""
        osc = self._osc(d)
        P, K = d['P'], d['K3']

        bK, bP = [], []
        for i in range(len(P)):
            row = osc[i, :]
            idx = np.where(row > 0.5)[0]
            if len(idx) == 0:
                continue
            j = idx[0]
            if j > 0 and row[j] != row[j-1]:
                k_interp = K[j-1] + (0.5 - row[j-1]) / (row[j] - row[j-1]) * (K[j] - K[j-1])
            else:
                k_interp = K[j]
            bK.append(float(k_interp))
            bP.append(float(P[i]))

        if not bK:
            return None

        return {
            'model': name,
            'K_onset_min': min(bK),
            'K_onset_max': max(bK),
            'K_onset_mean': np.mean(bK),
            'P_min': min(bP),
            'P_max': max(bP),
            'n_points': len(bK),
            'osc_area_%': 100 * np.sum(osc > 0.5) / osc.size,
            'boundary_K': bK,
            'boundary_P': bP,
        }

    def generate_report(self, output_dir: Optional[str] = None) -> dict:
        if not self.data:
            self.load()

        print('\n' + '='*70)
        print('BIFURCATION ANALYSIS')
        print('='*70)

        boundaries = []
        for name, d in self.data.items():
            b = self.extract_boundary(d, name)
            if b:
                boundaries.append(b)
                print(f'\n  {name}:')
                print(f'    K onset: {b["K_onset_min"]:.4f} – {b["K_onset_max"]:.4f}')
                print(f'    P range: [{b["P_min"]:.2f}, {b["P_max"]:.2f}]')
                print(f'    Osc area: {b["osc_area_%"]:.1f}%')
            else:
                print(f'\n  {name}: no bifurcation detected')

        # Comparisons
        print('\n' + '='*70)
        print('TOPOLOGY COMPARISON (2nd-Order Additive)')
        print('='*70)
        b_ring = next((b for b in boundaries if b['model'] == 'Ring 2nd Additive'), None)
        b_rand = next((b for b in boundaries if b['model'] == 'Random 2nd Additive'), None)
        if b_ring and b_rand:
            dk = b_rand['K_onset_mean'] - b_ring['K_onset_mean']
            da = b_rand['osc_area_%'] - b_ring['osc_area_%']
            print(f'  Ring K_onset_mean:   {b_ring["K_onset_mean"]:.4f}')
            print(f'  Random K_onset_mean: {b_rand["K_onset_mean"]:.4f}')
            print(f'  Difference (Rand-Ring): {dk:+.4f}')
            print(f'  Osc area diff: {da:+.1f} pp')

        print('\n' + '='*70)
        print('COUPLING ORDER COMPARISON (Ring Additive)')
        print('='*70)
        b_r2 = next((b for b in boundaries if b['model'] == 'Ring 2nd Additive'), None)
        b_r3 = next((b for b in boundaries if b['model'] == 'Ring 3rd Additive'), None)
        if b_r2 and b_r3:
            dk = b_r3['K_onset_mean'] - b_r2['K_onset_mean']
            da = b_r3['osc_area_%'] - b_r2['osc_area_%']
            print(f'  2nd-Order K_onset_mean: {b_r2["K_onset_mean"]:.4f}')
            print(f'  3rd-Order K_onset_mean: {b_r3["K_onset_mean"]:.4f}')
            print(f'  Difference (3rd-2nd): {dk:+.4f}')
            print(f'  Osc area diff: {da:+.1f} pp')

        report = {'boundaries': boundaries}

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            # CSV (without boundary arrays)
            rows = [{k: v for k, v in b.items() if k not in ('boundary_K', 'boundary_P')}
                    for b in boundaries]
            pd.DataFrame(rows).to_csv(out / 'bifurcation_summary.csv', index=False)
            # JSON (full)
            with open(out / 'bifurcation_analysis_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            print('\n✓ Saved: bifurcation_summary.csv, bifurcation_analysis_report.json')

        return report


def main():
    a = BifurcationAnalyzer(results_dir='../results')
    a.generate_report(output_dir='../results')

if __name__ == '__main__':
    main()
