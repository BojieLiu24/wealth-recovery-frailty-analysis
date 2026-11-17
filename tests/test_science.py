"""复用已审阅的原始合成案例，扩展公共验证入口。"""
import numpy as np
import pytest
from conftest import module_from_file
import fi_panel
from outcome_components import COMMON_22, PRIMARY_ITEMS

exposure = module_from_file('public_exposure_fixtures', '02_code/01_panel_build/tests/test_exposure_panel.py')
frailty = module_from_file('public_frailty_fixtures', '02_code/01_panel_build/tests/test_fi_panel.py')
primary = module_from_file('public_primary_fixtures', '02_code/03_results/section_3_2/tests/test_section_3_2.py')
math = module_from_file('public_sensitivity_math', '02_code/03_results/section_3_3/model_robustness/tests/test_preflight.py')


def test_shock_and_recovery_thresholds():
    exposure.test_recovery_boundaries_raw_tails_missing_and_first_shock()


def test_first_shock_selection_has_no_future_information():
    exposure.test_pass_a_cannot_read_tplus1_or_fi_and_is_invariant_to_tplus1()


@pytest.mark.parametrize('cohort', ['HRS','ELSA','KLoSA'])
def test_cohort_shock_windows(cohort):
    exposure.test_cohort_window_boundaries_and_outside_major_shocks(cohort)


def test_recovery_sensitivities():
    exposure.test_sensitivity_indicators_are_dynamic_and_audit_threshold_is_rejected()


def test_exposure_locked_register():
    exposure.test_specification_and_design_register_are_locked()


def test_fi_item_and_completeness_specification():
    assert fi_panel.PRIMARY_THRESHOLD == 27 and fi_panel.COMMON22_THRESHOLD == 20
    assert len(COMMON_22) == 22
    assert all(len(PRIMARY_ITEMS[c]) == 30 for c in ['HRS','ELSA','KLoSA'])
    frailty.test_design_register_has_exact_locked_v2_decisions()


def test_primary_fi_synthetic_boundaries():
    frailty.test_build_boundaries_and_opportunity_mask(fi_panel.V2_SPEC, 'v2', 27, 20, 26, 19)


def test_fi_flags_and_missing_interviews():
    frailty.test_complete_flags_are_dynamic_specification_aware_and_not_physical()


def test_model_matrix():
    primary.test_primary_formula_has_42_columns_when_all_levels_are_present()


def test_meta_equal_effects():
    primary.test_meta_analysis_equal_effects_has_zero_heterogeneity()


def test_meta_metafor_reference():
    primary.test_meta_analysis_matches_locked_metafor_fixture()


def test_random_slope_scaling_math():
    math.test_scale_equivalence_without_fit()


@pytest.mark.parametrize('correlation', [-.8, 0., .8])
def test_random_slope_covariance_math(correlation):
    math.test_scale_variance_covariance_and_correlation(correlation)


@pytest.mark.parametrize('knots', [(0.,2.,8.), (0.,3.1,11.)])
def test_nonlinear_spline_math(knots):
    math.test_rcs_natural_basis_and_derivative(knots)


def test_nonlinear_invalid_knots():
    math.test_rcs_rejects_invalid_knots()
