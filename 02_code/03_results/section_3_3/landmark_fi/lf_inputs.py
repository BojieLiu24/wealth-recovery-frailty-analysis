"""冻结输入、时点、同样本与来源链核验；不输出个人面板。"""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from lf_common import (ROOT, RB_CODE, COHORTS, read_json, sha, rel, require,
                       source_code_paths)
from lf_spec import (COHORT_ANCHORS, SINGLE_COUNTS, LANDMARK_CONSTANTS,
                     LM_MAIN, LM_TIME, CONTINUOUS)
from lf_design import parameterize, delta_equivalence
from prepare_inputs import verify_sources as primary_sources, load_cohort, reconstruct_primary


def verify_sources():
    audit = primary_sources()
    inputs = audit['inputs'].copy()
    def pin(path, expected=None):
        h = sha(path)
        require(expected is None or h == expected, '冻结来源哈希异常：' + rel(path))
        inputs[rel(path)] = h
    for p in source_code_paths():
        pin(p)
    fi_path = ROOT / '00_admin/panel_manifests/fi_panel/fi_panel_v2_freeze.json'
    pin(fi_path)
    freeze = read_json(fi_path)
    require(freeze['freeze_status'] == 'FROZEN' and freeze['specification']['primary_threshold']['minimum_valid'] == 27,
            'FI版本或有效条目规则错误')
    for rec in freeze['cohorts'].values():
        for key, h in [('parquet_path', 'parquet_sha256'), ('manifest_path', 'manifest_sha256')]:
            pin(ROOT/rec[key], rec[h])
    releases = []
    for rid in ('wealth_measurement_v1', 'model_robustness_round1b_v1'):
        base = ROOT / '03_output/3.3 Robustness of the Main Findings/releases' / rid
        fr = read_json(base/'freeze.json'); manifest = read_json(base/'release_manifest.json')
        require(fr['status'] == 'FROZEN_WITH_CAVEATS', '既有发布身份变化')
        pin(base/'freeze.json'); pin(base/'release_manifest.json', fr['release_manifest_sha256'])
        pin(base/'release_validation.json', fr['release_validation_sha256'])
        inventory = manifest.get('inventory', manifest.get('analysis_inventory'))
        for r in inventory:
            p = (base/r.get('path', r.get('release_path'))).resolve()
            require(p.is_relative_to(base.resolve()) and sha(p) == r['sha256'], '既有发布内容变化')
        releases.append({'release_id': rid, 'status': fr['status'], 'freeze_sha256': sha(base/'freeze.json'),
                         'inventory_files_checked': len(inventory)})
    for name in ('Editor.md', 'Reviewers.md'):
        pin(ROOT/'07_manuscript/Round1'/name)
    audit.update(inputs=inputs, prior_releases=releases, landmark_FI_source='FI Panel v2, same fi_primary as shock')
    return audit


def ordered_row_hash(frame):
    value = frame[['person_id', 'wave']].to_json(orient='values', force_ascii=False)
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def validate_landmark_rows(frame):
    require(not frame[['person_id', 'wave']].duplicated().any(), '重复person-wave')
    roles = frame.groupby(['person_id', 'fi_time_role']).size().unstack(fill_value=0)
    require(set(roles) == {'landmark', 'later'} and roles.landmark.eq(1).all() and roles.later.ge(1).all(),
            '个人必须有唯一landmark和至少一条later')
    lm = frame.loc[frame.fi_time_role.eq('landmark')].copy()
    post = frame.loc[frame.fi_time_role.eq('later')].copy()
    require(lm.time_years.eq(0).all() and post.time_years.gt(0).all(), '禁止重置原点或非正later时间')
    require(lm.wave.eq(lm.landmark_wave).all() and lm.landmark_wave.eq(lm.index_shock_wave+1).all(), '冲击与landmark波次不符')
    require(set(lm.person_id) == set(post.person_id), '删除landmark后参与者改变')
    require(np.isfinite(lm.fi_primary.astype(float)).all() and np.isfinite(post.fi_primary.astype(float)).all(), 'FI缺失')
    post['fi_landmark'] = post.person_id.map(lm.set_index('person_id').fi_primary.astype(float))
    require(post.fi_landmark.notna().all(), 'landmark FI连接丢失；禁止另取完整案例')
    return lm, post


def support(post):
    counts = post.groupby('person_id').size()
    rows = []
    for group, block in post.groupby('recovery_status'):
        n = block.groupby('person_id').size()
        rows.append({'group': str(group), 'participants': len(n), 'rows': len(block),
            'single_post_n': int(n.eq(1).sum()), 'single_post_percent': float(100*n.eq(1).mean()),
            'min_time': float(block.time_years.min()), 'max_time': float(block.time_years.max()),
            'max_followup_quartiles': block.groupby('person_id').time_years.max().quantile([.25,.5,.75]).tolist(),
            'two_year_bins': [{'lower_exclusive': float(a), 'upper_inclusive': float(a+2),
                'participants': int(block.loc[block.time_years.gt(a)&block.time_years.le(a+2), 'person_id'].nunique()),
                'rows': int((block.time_years.gt(a)&block.time_years.le(a+2)).sum())}
                for a in np.arange(0, np.ceil(float(block.time_years.max())/2)*2, 2)],
            'tail_over_8_participants': int(block.loc[block.time_years.gt(8), 'person_id'].nunique()),
            'tail_over_8_rows': int(block.time_years.gt(8).sum())})
    return {'single_post_n': int(counts.eq(1).sum()), 'single_post_percent': float(100*counts.eq(1).mean()),
        'repeated_post_participants': int(counts.ge(2).sum()),
        'within_person_post_pairs': int((counts*(counts-1)/2).sum()),
        'observation_count_distribution': {str(k): int(v) for k,v in counts.value_counts().sort_index().items()},
        'groups': rows, 'counts_are_observation_support_not_survival_risk_sets': True}


def prepare_cohort(cohort):
    frame, lm, x_primary, beta, covariance, diagnostics, original_anchor = load_cohort(cohort)
    reconstruction = reconstruct_primary(frame, x_primary, beta, diagnostics)
    lm, post = validate_landmark_rows(frame)
    constants = LANDMARK_CONSTANTS[cohort]
    actual = (float(lm.fi_primary.astype(float).mean()), float(lm.fi_primary.astype(float).std(ddof=1)))
    require([x.hex() for x in actual] == [x.hex() for x in constants], 'Landmark标准化完整精度漂移')
    post[LM_MAIN] = (post.fi_landmark-constants[0])/constants[1]
    fi_freeze = read_json(ROOT/'00_admin/panel_manifests/fi_panel/fi_panel_v2_freeze.json')
    rec = fi_freeze['cohorts'][cohort]
    require(sha(ROOT/rec['parquet_path']) == rec['parquet_sha256'], 'FI Panel物理SHA错误')
    fi = pd.read_parquet(ROOT/rec['parquet_path'], columns=['person_id','wave','fi_primary','fi_primary_n_valid'])
    fi.person_id = fi.person_id.astype(str)
    for role, wave, observed in [('shock','index_shock_wave','fi_t'), ('landmark','landmark_wave','fi_primary')]:
        joined = lm[['person_id',wave,observed]].merge(fi.rename(columns={'wave':wave,'fi_primary':'source_fi'}),
            on=['person_id',wave], how='left', validate='one_to_one')
        require(len(joined) == len(lm) and joined.source_fi.notna().all(), '时点连接或FI缺失')
        np.testing.assert_array_equal(joined[observed].to_numpy(float), joined.source_fi.to_numpy(float))
        require(joined.fi_primary_n_valid.ge(27).all(), 'FI不满足冻结条目规则')
    anchors = {**original_anchor, 'primary_rows': original_anchor['rows'], 'rows': len(post),
        'ordered_person_wave_sha256': ordered_row_hash(post), 'landmark_in_outcome': False}
    require(len(post) == COHORT_ANCHORS[cohort]['later'], '后续行数漂移')
    s = support(post); require(s['single_post_n'] == SINGLE_COUNTS[cohort], '单次观察人数漂移')
    original = x_primary.loc[post.index].copy()
    lm_original = original.copy(); lm_original[LM_MAIN] = post[LM_MAIN]
    lm_original[LM_TIME] = post[LM_MAIN]*post.time_years
    models = {}
    for spec, matrix in [('PO01',original), ('LM01',lm_original)]:
        basis, mapping = parameterize(matrix, post.time_years.to_numpy(float), cohort, spec)
        full_start = beta if spec == 'PO01' else np.r_[beta,0.,0.]
        models[spec] = {'x': basis, 'mapping': mapping, 'start_beta': np.asarray(mapping['A'])@full_start}
    # 从冻结常数取shock尺度；不能因post-only窗口重新标准化。
    table = pd.read_csv(ROOT/'03_output/3.2 Wealth Recovery and Subsequent Frailty Progression/Primary Models/primary_lmm_standardization_constants.csv',
                        encoding='utf-8', float_precision='round_trip')
    row = table.loc[table.cohort.eq(cohort)&table.variable.eq('fi_t')].iloc[0]
    equivalence = delta_equivalence(lm_original, post, float(row['mean']), float(row.sample_sd), *constants)
    person = lm[['fi_t','fi_primary']].astype(float).rename(columns={'fi_primary':'fi_landmark'})
    person['delta_fi'] = person.fi_landmark-person.fi_t
    profile = {v: {'mean':float(person[v].mean()), 'sample_sd':float(person[v].std(ddof=1)),
                  'min':float(person[v].min()), 'max':float(person[v].max()),
                  'quartiles':person[v].quantile([.25,.5,.75]).tolist()} for v in person}
    report = {'cohort':cohort, 'anchor':anchors, 'primary_anchor':original_anchor,
        'primary_reconstruction':reconstruction, 'support':s, 'landmark_constants':{'mean':constants[0],'sample_sd':constants[1]},
        'FI_source_equality':'EXACT', 'FI_profile':profile, 'FI_correlation':person.corr().to_dict(),
        'models': {spec: model['mapping'] for spec,model in models.items()}, 'delta_equivalence':equivalence,
        'PO01_LM01_row_identity_equal':True, 'model_fit_calls':0}
    return post, models, report, beta, covariance, diagnostics
