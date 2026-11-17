"""仅从冻结交叉人数补充Table S6恢复比例和signed SMD；不读取微观数据。"""
from __future__ import annotations

import csv
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RELEASE = ROOT / '03_output/3.3 Robustness of the Main Findings/releases/mortality_attrition_round1d_v1'


def sha256(path: Path) -> str:
    """核对统计来源身份。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary() -> dict:
    """读取已发布聚合人数，只新增恢复比例与二分类SMD。"""
    freeze = json.loads((RELEASE / 'freeze.json').read_text(encoding='utf-8'))
    assert freeze['status'] == 'FROZEN_WITH_CAVEATS'
    index_path = RELEASE / 'release_evidence_index.csv'
    assert sha256(index_path) == freeze['evidence_index_sha256']
    index = list(csv.DictReader(index_path.open(encoding='utf-8-sig', newline='')))
    names = ['Evidence/Stage0/observability_disclosure_safe.csv',
             'Evidence/Stage0/health_observability_disclosure_safe.csv',
             'QA/selection_validation.json', 'Code Snapshot/audit_descriptives.py']
    hashes = {}
    for name in names:
        identity = [r for r in index if r['path'] == name and r['source_scope'] == 'PACKAGE_PAYLOAD']
        assert len(identity) == 1
        hashes[name] = sha256(RELEASE / name)
        assert hashes[name] == identity[0]['sha256'], name
    rows = list(csv.DictReader((RELEASE / names[0]).open(encoding='utf-8-sig', newline='')))
    health = list(csv.DictReader((RELEASE / names[1]).open(encoding='utf-8-sig', newline='')))
    assert not [r for r in health if r['population_id'] == 'L2'
                and r['comparison'] == 'first_post_FI_vs_no_first_post_FI'
                and r['variable'] in ['recovery_status', 'substantial_recovery']]
    qa = json.loads((RELEASE / names[2]).read_text(encoding='utf-8'))
    assert qa['status'] == 'PASS'
    anchors = {'HRS': (6168, 5114, 1054, 1703), 'ELSA': (879, 699, 180, 271),
               'KLoSA': (2750, 2502, 248, 943)}

    def count(cohort: str, recovery: str, metric: str) -> int:
        """按冻结人群、恢复状态和指标取得唯一整数人数。"""
        matches = [r for r in rows if r['cohort'] == cohort and r['population_id'] == 'L2'
                   and r['recovery_status'] == recovery and r['stratum'] == 'all'
                   and r['metric'] == metric]
        assert len(matches) == 1, (cohort, recovery, metric)
        row = matches[0]
        assert row['unit'] == 'participant' and row['computability_status'] == 'CALCULABLE'
        assert row['small_cell_internal_review_only'] == 'False'
        value = Decimal(row['numerator'])
        assert value == value.to_integral_value()
        return int(value)

    result = []
    for cohort, (total, observed, missing, primary_recovery) in anchors.items():
        q = next(c for c in qa['cohorts'] if c['cohort'] == cohort)
        assert (q['L2'], q['Primary_participants'], q['L2_no_first_FI']) == (total, observed, missing)
        by_recovery = {}
        for recovery in ['substantial_recovery', 'persistent_or_limited_recovery']:
            n = count(cohort, recovery, 'population_n')
            yes = count(cohort, recovery, 'has_valid_post_FI')
            no = count(cohort, recovery, 'valid_FI_count_0')
            assert yes + no == n
            assert count(cohort, recovery, 'has_timed_post_FI') == yes
            by_recovery[recovery] = {'total': n, 'observed': yes, 'no_observed': no}
        assert sum(g['observed'] for g in by_recovery.values()) == observed
        assert sum(g['no_observed'] for g in by_recovery.values()) == missing
        assert sum(g['total'] for g in by_recovery.values()) == total
        recovered = by_recovery['substantial_recovery']
        assert recovered['observed'] == primary_recovery
        with localcontext() as context:
            context.prec = 40
            p_obs = Decimal(recovered['observed']) / Decimal(observed)
            p_no = Decimal(recovered['no_observed']) / Decimal(missing)
            scale = ((p_obs * (1 - p_obs) + p_no * (1 - p_no)) / 2).sqrt()
            assert scale > 0
            smd = (p_obs - p_no) / scale
            result.append({'cohort': cohort, 'L2_total': total,
                           'observed_n': observed, 'no_observed_n': missing,
                           'substantial_recovery_observed_n': recovered['observed'],
                           'substantial_recovery_no_observed_n': recovered['no_observed'],
                           'observed_recovery_proportion': str(p_obs),
                           'no_observed_recovery_proportion': str(p_no),
                           'signed_SMD': str(smd),
                           'observed_display': f"{recovered['observed']:,} ({p_obs * 100:.1f})",
                           'no_observed_display': f"{recovered['no_observed']:,} ({p_no * 100:.1f})",
                           'SMD_display': f'{smd:+.3f}'})
    return {'status': 'NEW_DESCRIPTIVE_DERIVATION_NOT_IN_ORIGINAL_FREEZE',
            'frozen_evidence': 'L2 recovery-status by subsequent-FI cross-counts',
            'new_results': 'Within-FI-group recovery percentages and signed binary SMD',
            'release': str(RELEASE.relative_to(ROOT)), 'freeze_sha256': sha256(RELEASE / 'freeze.json'),
            'source_sha256': hashes,
            'population': 'Original L2; no participant-level reconstruction or event reselection',
            'formula': '(p_observed-p_noFI)/sqrt((p_observed*(1-p_observed)+p_noFI*(1-p_noFI))/2)',
            'definition': 'Recovery=1 substantial recovery; Recovery=0 persistent/limited recovery',
            'direction': 'Observed FI minus No observed FI',
            'rounding': 'Percentages 1 decimal; signed SMD 3 decimals; full precision retained below',
            'microdata_read': False, 'model_or_test_calls': 0, 'cohorts': result}


if __name__ == '__main__':
    print(json.dumps(summary(), ensure_ascii=False, indent=2))
