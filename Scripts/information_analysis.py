"""
information_analysis.py
=======================
Information-theoretic metric analysis across all model variants.

Reads .npz files with keys: P, K3, oscillation_map, osc_fraction,
TC, DTC, TC_ksg, DTC_ksg, Oinfo_ksg.

Analyses:
- KSG vs Gaussian estimator comparison
- Synergy vs redundancy (sign of Oinfo_ksg)
- Information in oscillatory vs. fixed-point regions
- Bifurcation-related information transitions

Usage
-----
    from information_analysis import InformationAnalyzer
    analyzer = InformationAnalyzer(results_dir='../results')
    analyzer.generate_report(output_dir='../results')
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, Optional


class InformationAnalyzer:

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

    # ──────────────────────────────────────────────────────────────────
    def ksg_vs_gaussian(self, d: dict, name: str) -> dict:
        """Compare KSG and Gaussian estimators."""
        result = {'model': name}
        pairs = [('TC', 'TC_ksg'), ('DTC', 'DTC_ksg')]

        for gauss, ksg in pairs:
            if gauss in d and ksg in d:
                g = d[gauss].astype(float).flatten()
                k = d[ksg].astype(float).flatten()
                mask = ~(np.isnan(g) | np.isnan(k))
                if mask.sum() > 2:
                    corr = np.corrcoef(g[mask], k[mask])[0, 1]
                    result[f'{gauss}_corr'] = corr
                    result[f'{gauss}_gauss_mean'] = np.mean(g[mask])
                    result[f'{gauss}_ksg_mean']   = np.mean(k[mask])
                    result[f'{gauss}_diff_mean']   = np.mean(k[mask] - g[mask])
        return result

    # ──────────────────────────────────────────────────────────────────
    def synergy_redundancy(self, d: dict, name: str) -> dict:
        """Analyze O-information sign: positive = redundancy, negative = synergy."""
        if 'Oinfo_ksg' not in d:
            return {'model': name, 'available': False}

        oinfo = d['Oinfo_ksg'].astype(float)
        valid = oinfo[~np.isnan(oinfo)]

        threshold = 0.01 * np.std(valid) if len(valid) > 0 else 0.001

        n_syn = np.sum(valid < -threshold)
        n_red = np.sum(valid > threshold)
        n_mix = np.sum(np.abs(valid) <= threshold)
        total = len(valid)

        return {
            'model': name,
            'available': True,
            'synergy_%':     100 * n_syn / total,
            'redundancy_%':  100 * n_red / total,
            'mixed_%':       100 * n_mix / total,
            'oinfo_mean':    np.mean(valid),
            'oinfo_std':     np.std(valid),
            'oinfo_min':     np.min(valid),
            'oinfo_max':     np.max(valid),
        }

    # ──────────────────────────────────────────────────────────────────
    def region_analysis(self, d: dict, name: str) -> dict:
        """Compare information in FP vs LC regions."""
        osc = self._osc(d)
        fp = osc < 0.5
        lc = osc > 0.5
        result = {'model': name}

        for k in ['TC_ksg', 'DTC_ksg', 'Oinfo_ksg', 'TC', 'DTC']:
            if k not in d:
                continue
            arr = d[k].astype(float)
            fp_v = arr[fp]; fp_v = fp_v[~np.isnan(fp_v)]
            lc_v = arr[lc]; lc_v = lc_v[~np.isnan(lc_v)]

            if len(fp_v) > 0:
                result[f'{k}_FP'] = np.mean(fp_v)
            if len(lc_v) > 0:
                result[f'{k}_LC'] = np.mean(lc_v)
            if len(fp_v) > 0 and len(lc_v) > 0:
                result[f'{k}_transition'] = np.mean(lc_v) - np.mean(fp_v)

        return result

    # ──────────────────────────────────────────────────────────────────
    def generate_report(self, output_dir: Optional[str] = None) -> dict:
        if not self.data:
            self.load()

        print('\n' + '='*70)
        print('INFORMATION-THEORETIC ANALYSIS')
        print('='*70)

        ksg_gauss_rows = []
        synergy_rows   = []
        region_rows    = []

        for name, d in self.data.items():
            kg = self.ksg_vs_gaussian(d, name)
            ksg_gauss_rows.append(kg)

            sr = self.synergy_redundancy(d, name)
            synergy_rows.append(sr)

            ra = self.region_analysis(d, name)
            region_rows.append(ra)

        # ── Print results ──
        print('\n── KSG vs Gaussian Estimators ──')
        for r in ksg_gauss_rows:
            tc_corr  = r.get('TC_corr', float('nan'))
            dtc_corr = r.get('DTC_corr', float('nan'))
            print(f'  {r["model"]:<25}  TC_corr={tc_corr:.4f}  DTC_corr={dtc_corr:.4f}')

        print('\n── Synergy vs Redundancy (Oinfo_ksg) ──')
        for r in synergy_rows:
            if r.get('available'):
                print(f'  {r["model"]:<25}  Syn={r["synergy_%"]:.1f}%  Red={r["redundancy_%"]:.1f}%  '
                      f'Mean={r["oinfo_mean"]:.6f}')

        print('\n── Information in FP vs LC Regions ──')
        for r in region_rows:
            for k in ['TC_ksg', 'DTC_ksg', 'Oinfo_ksg']:
                fp_k = f'{k}_FP'
                lc_k = f'{k}_LC'
                tr_k = f'{k}_transition'
                if fp_k in r and lc_k in r:
                    print(f'  {r["model"]:<25} {k:<12}  FP={r[fp_k]:.6f}  LC={r[lc_k]:.6f}  '
                          f'Δ={r.get(tr_k, 0):.6f}')

        # ── Topology comparison ──
        print('\n── Topology Effects on Information ──')
        r2_ring = next((r for r in region_rows if r['model'] == 'Ring 2nd Additive'), None)
        r2_rand = next((r for r in region_rows if r['model'] == 'Random 2nd Additive'), None)
        if r2_ring and r2_rand:
            for k in ['TC_ksg', 'DTC_ksg', 'Oinfo_ksg']:
                lc_k = f'{k}_LC'
                if lc_k in r2_ring and lc_k in r2_rand:
                    diff = r2_rand[lc_k] - r2_ring[lc_k]
                    print(f'  {k} in LC:  Ring={r2_ring[lc_k]:.6f}  Random={r2_rand[lc_k]:.6f}  Δ={diff:+.6f}')

        report = {
            'ksg_vs_gaussian': ksg_gauss_rows,
            'synergy_redundancy': synergy_rows,
            'region_analysis': region_rows,
        }

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            with open(out / 'information_analysis_report.json', 'w') as f:
                json.dump(self._ser(report), f, indent=2)
            pd.DataFrame(synergy_rows).to_csv(out / 'synergy_redundancy_summary.csv', index=False)
            print('\n✓ Saved: information_analysis_report.json, synergy_redundancy_summary.csv')

        return report

    def _ser(self, obj):
        if isinstance(obj, dict):
            return {k: self._ser(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._ser(i) for i in obj]
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        return obj


def main():
    a = InformationAnalyzer(results_dir='../results')
    a.generate_report(output_dir='../results')

if __name__ == '__main__':
    main()
