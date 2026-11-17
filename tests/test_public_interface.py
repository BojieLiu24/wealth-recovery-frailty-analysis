"""聚合验证与可移植路径的失败/成功行为测试。"""
import json
import pytest
from conftest import ROOT
import portable_data_paths as paths
from verify_public import verify


def test_final_supplement_values_and_numerical_exceptions():
    from verify_supplement import verify as verify_tables
    import pandas as pd
    assert verify_tables()['status'] == 'PASS'
    for file, spec in [('landmark_S11.csv','PO01'), ('shock_wave_S12.csv','SP01')]:
        data = pd.read_csv(ROOT/'outputs/published_aggregates'/file)
        row = data[(data.cohort=='HRS') & (data.specification==spec)].iloc[0]
        assert row.original_stage2_status == 'HOLD_NOT_ESTIMABLE'
        assert row.accepted_status == 'PASS_WITH_CAVEAT'


def test_public_aggregates():
    result = verify()
    assert result['status'] == 'PASS'
    assert result['independent_model_reestimation'] is False


def test_mode_b_missing_configuration(monkeypatch):
    monkeypatch.delenv('WF_PATHS_FILE', raising=False)
    with pytest.raises(FileNotFoundError, match='WF_PATHS_FILE'):
        paths.resolve_asset('hrs_rand_longitudinal')


def test_unknown_source_never_falls_back():
    with pytest.raises(KeyError):
        paths.asset_record('different_cohort_or_version')


def test_config_relative_paths_and_hash_mismatch(tmp_path, monkeypatch):
    source = tmp_path/'synthetic.dta'
    source.write_bytes(b'SYNTHETIC TEST ONLY - NOT COHORT DATA')
    cfg = tmp_path/'paths.json'
    cfg.write_text(json.dumps({'data_root':'.','assets':{'hrs_rand_longitudinal':'synthetic.dta'}}), encoding='utf-8')
    monkeypatch.setenv('WF_PATHS_FILE', str(cfg))
    assert paths.resolve_asset('hrs_rand_longitudinal') == source.resolve()
    with pytest.raises(ValueError, match='SHA-256'):
        paths.verify_asset('hrs_rand_longitudinal')


def test_missing_asset_clear_error(tmp_path, monkeypatch):
    cfg = tmp_path/'paths.json'
    cfg.write_text(json.dumps({'data_root':'.','assets':{}}), encoding='utf-8')
    monkeypatch.setenv('WF_PATHS_FILE', str(cfg))
    with pytest.raises(FileNotFoundError, match='no fallback'):
        paths.resolve_asset('hrs_rand_longitudinal')


def test_inputs_inside_git_tree_rejected(tmp_path, monkeypatch):
    cfg = tmp_path/'paths.json'
    cfg.write_text(json.dumps({'assets':{'hrs_rand_longitudinal':str(ROOT/'README.md')}}), encoding='utf-8')
    monkeypatch.setenv('WF_PATHS_FILE', str(cfg))
    with pytest.raises(ValueError, match='outside'):
        paths.resolve_asset('hrs_rand_longitudinal')
