"""Mode A：从公开聚合核验数值并重建Figure 2；不重新拟合原模型。"""
from pathlib import Path
import argparse
import importlib.util
import json
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT / 'outputs/published_aggregates'


def load_primary():
    path = ROOT / '02_code/03_results/section_3_2/run_section_3_2.py'
    spec = importlib.util.spec_from_file_location('primary_public', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify():
    """断言使用已发表数值作验证靶点，不把它们当成拟合结果生成器。"""
    primary = pd.read_csv(AGG / 'table_s6_primary_rows.csv', float_precision='round_trip')
    meta = pd.read_csv(AGG / 'primary_lmm_meta_summary.csv', float_precision='round_trip')
    figure = pd.read_csv(AGG / 'figure_2_source.csv', float_precision='round_trip')
    targets = {'HRS': (5114, .0013, .0006, .0019), 'ELSA': (699, -.0014, -.0037, .0008), 'KLoSA': (2502, .0006, -.0006, .0017)}
    if primary['cohort'].tolist() != list(targets) or len(meta) != 1:
        raise ValueError('Unexpected aggregate cohort schema.')
    for row in primary.itertuples():
        target = targets[row.cohort]
        if int(row.participants) != target[0] or tuple(round(v, 4) for v in (row.estimate, row.ci_lower, row.ci_upper)) != target[1:]:
            raise ValueError(f'{row.cohort}: published value mismatch.')
        if row.se <= 0 or row.ci_lower > row.estimate or row.estimate > row.ci_upper:
            raise ValueError('Invalid standard error or confidence interval.')
    model = load_primary()
    data = primary.assign(is_primary_estimand=True)
    recomputed = model.meta_analysis(data)
    for key in ['estimate', 'se', 'ci_lower', 'ci_upper', 'tau2', 'i2_percent']:
        if not np.isclose(recomputed[key], float(meta.iloc[0][key]), rtol=1e-9, atol=1e-12):
            raise ValueError(f'Secondary synthesis mismatch: {key}')
    expected = pd.concat([primary[['cohort','estimate','ci_lower','ci_upper']], meta[['cohort','estimate','ci_lower','ci_upper']]], ignore_index=True)
    if figure['cohort'].tolist() != expected['cohort'].tolist():
        raise ValueError('Figure input row order differs.')
    for col in ['estimate','ci_lower','ci_upper']:
        if not np.allclose(figure[col], expected[col], atol=1e-14, rtol=1e-10):
            raise ValueError(f'Figure 2 source mismatch: {col}')
    return {'status': 'PASS', 'mode': 'AGGREGATE_BASED_VERIFICATION', 'independent_model_reestimation': False, 'cohort_participants': {k: v[0] for k,v in targets.items()}, 'secondary_summary': {k: recomputed[k] for k in ['estimate','ci_lower','ci_upper','i2_percent']}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/generated')
    parser.add_argument('--no-figure', action='store_true')
    args = parser.parse_args()
    result = verify()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.no_figure:
        subprocess.run([sys.executable, str(ROOT/'code/plot_figure_2.py'), '--source', str(AGG/'figure_2_source.csv'), '--output', str(args.output)], check=True)
    (args.output/'verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
