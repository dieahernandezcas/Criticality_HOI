"""
analysis_statistics.py
======================
Comprehensive statistical analysis for all Wilson-Cowan model variants.

Reads .npz files with keys: P, K3, oscillation_map, osc_fraction,
TC, DTC, TC_ksg, DTC_ksg, Oinfo_ksg, cumulant, powercorr, skew, kurt.

Usage
-----
    from analysis_statistics import StatisticsAnalyzer
    analyzer = StatisticsAnalyzer(results_dir='../results')
    analyzer.generate_full_report(output_dir='../results')
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats
import json
from typing import Dict, Optional


class StatisticsAnalyzer:

    # ── NPZ key names (as saved by sweep notebooks) ──
    INFO_KEYS = ['TC', 'DTC', 'TC_ksg', 'DTC_ksg', 'Oinfo_ksg',
                 'cumulant', 'powercorr', 'skew', 'kurt']

    MODELS_3N = {
        '2nd-Order Additive':  'Pairwise_noise',
        '2nd-Order Diffusive': 'Pairwise_diffusive_noise',
        '3rd-Order Additive':  'Coupling_noise',             # no KSG
        '3rd-Order Diffusive': 'Coupling_diffusive_noise',
    }

    MODELS_10N = {
        'Ring 2nd Additive':     '10n_ring_sw_pairwise_noise',
        'Ring 3rd Additive':     '10n_ring_sw_coupling_noise',
        'Ring 2nd Diffusive':    '10n_ring_sw_pairwise_diffusive_noise',
        'Ring 3rd Diffusive':    '10n_ring_sw_coupling_diffusive_noise',
        'Random 2nd Additive':   '10n_random_pairwise_noise',
        'Random 3rd Additive':   '10n_random_coupling_noise',
        'Random 2nd Diffusive':  '10n_random_pairwise_diffusive_noise',
        'Random 3rd Diffusive':  '10n_random_coupling_diffusive_noise',
    }

    def __init__(self, results_dir: str = '../results'):
        self.results_dir = Path(results_dir)
        self.results: Dict[str, dict] = {}

    # ──────────────────────────────────────────────────────────────────
    def load_results(self):
        all_models = {**self.MODELS_3N, **self.MODELS_10N}
        for name, tag in all_models.items():
            path = self.results_dir / f'metrics_{tag}.npz'
            if path.exists():
                self.results[name] = dict(np.load(path))
                print(f'  ✓ {name}')
            else:
                print(f'  ⚠ {name} — file not found')
        return self.results

    # ──────────────────────────────────────────────────────────────────
    def _osc_map(self, data: dict) -> np.ndarray:
        """Return best oscillation indicator (prefer osc_fraction)."""
        return data.get('osc_fraction', data.get('oscillation_map'))

    def compute_model_stats(self, data: dict, name: str) -> dict:
        row = {'model': name}
        osc = self._osc_map(data)

        # Oscillation
        row['osc_mean'] = np.nanmean(osc)
        row['osc_std']  = np.nanstd(osc)
        row['osc_area_%'] = 100.0 * np.sum(osc > 0.5) / osc.size

        fp = osc < 0.5
        lc = osc > 0.5

        for k in self.INFO_KEYS:
            if k not in data:
                continue
            arr = data[k].astype(float)
            valid = arr[~np.isnan(arr)]
            if len(valid) == 0:
                continue
            row[f'{k}_mean'] = np.nanmean(valid)
            row[f'{k}_std']  = np.nanstd(valid)
            row[f'{k}_min']  = np.nanmin(valid)
            row[f'{k}_max']  = np.nanmax(valid)
            row[f'{k}_median'] = np.nanmedian(valid)
            row[f'{k}_skew']   = sp_stats.skew(valid)
            row[f'{k}_kurt']   = sp_stats.kurtosis(valid)

            fp_vals = arr[fp]; fp_vals = fp_vals[~np.isnan(fp_vals)]
            lc_vals = arr[lc]; lc_vals = lc_vals[~np.isnan(lc_vals)]
            if len(fp_vals) > 0:
                row[f'{k}_FP_mean'] = np.mean(fp_vals)
            if len(lc_vals) > 0:
                row[f'{k}_LC_mean'] = np.mean(lc_vals)

        return row

    # ──────────────────────────────────────────────────────────────────
    def generate_full_report(self, output_dir: Optional[str] = None) -> dict:
        if not self.results:
            self.load_results()

        print('\n' + '='*70)
        print('COMPREHENSIVE STATISTICS REPORT')
        print('='*70)

        rows_3n, rows_10n = [], []

        print('\n── 3-Node Models ──')
        for name in self.MODELS_3N:
            if name in self.results:
                s = self.compute_model_stats(self.results[name], name)
                rows_3n.append(s)
                self._print_summary(s)

        print('\n── 10-Node Models ──')
        for name in self.MODELS_10N:
            if name in self.results:
                s = self.compute_model_stats(self.results[name], name)
                rows_10n.append(s)
                self._print_summary(s)

        df_3n  = pd.DataFrame(rows_3n).set_index('model')  if rows_3n  else pd.DataFrame()
        df_10n = pd.DataFrame(rows_10n).set_index('model') if rows_10n else pd.DataFrame()

        # Comparisons
        comparisons = self._comparisons()

        report = {'3node': rows_3n, '10node': rows_10n, 'comparisons': comparisons}

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            if not df_3n.empty:
                df_3n.to_csv(out / 'summary_3node.csv')
            if not df_10n.empty:
                df_10n.to_csv(out / 'summary_10node.csv')
            with open(out / 'full_statistics_report.json', 'w') as f:
                json.dump(self._serializable(report), f, indent=2)
            print('\n✓ Saved: summary_3node.csv, summary_10node.csv, full_statistics_report.json')

        return report

    # ──────────────────────────────────────────────────────────────────
    def _print_summary(self, s: dict):
        print(f'\n  {s["model"]}:')
        print(f'    Osc area: {s["osc_area_%"]:.1f}%')
        for k in ['TC_ksg', 'DTC_ksg', 'Oinfo_ksg']:
            if f'{k}_mean' in s:
                print(f'    {k}: {s[f"{k}_mean"]:.6f} ± {s[f"{k}_std"]:.6f}')

    def _comparisons(self) -> dict:
        comp = {}
        # Ring vs Random (2nd additive)
        r = self.results.get('Ring 2nd Additive')
        d = self.results.get('Random 2nd Additive')
        if r and d:
            comp['ring_vs_random_2nd'] = self._diff_metrics(r, d)
        # 2nd vs 3rd (Ring)
        r2 = self.results.get('Ring 2nd Additive')
        r3 = self.results.get('Ring 3rd Additive')
        if r2 and r3:
            comp['ring_2nd_vs_3rd'] = self._diff_metrics(r2, r3)
        return comp

    def _diff_metrics(self, d1, d2) -> dict:
        diffs = {}
        for k in self.INFO_KEYS:
            if k in d1 and k in d2:
                m1 = np.nanmean(d1[k])
                m2 = np.nanmean(d2[k])
                diffs[k] = {'mean_1': m1, 'mean_2': m2, 'diff': m2 - m1}
        return diffs

    def _serializable(self, obj):
        if isinstance(obj, dict):
            return {k: self._serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._serializable(i) for i in obj]
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        return obj


def main():
    a = StatisticsAnalyzer(results_dir='../results')
    a.generate_full_report(output_dir='../results')

if __name__ == '__main__':
    main()
