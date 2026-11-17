"""唯一Stage0–2规格；科学定义不从稿件或结果显著性推导。"""
from __future__ import annotations

COHORT_ANCHORS = {
    "HRS": {"participants": 5114, "rows": 23411, "later": 18297, "recovery": 1703, "reference": 3411},
    "ELSA": {"participants": 699, "rows": 2584, "later": 1885, "recovery": 271, "reference": 428},
    "KLoSA": {"participants": 2502, "rows": 10762, "later": 8260, "recovery": 943, "reference": 1559},
}
KNOTS = {
    "HRS": [0.0, 5.75, 12.666666666666666],
    "ELSA": [0.0, 4.083333333333333, 9.550000000000011],
    "KLoSA": [0.0, 4.083333333333333, 10.0],
}
CONTINUOUS = ("age", "household_size", "household_income_rank", "wealth_loss_fraction", "fi_t")
CATEGORIES = {
    "sex": ("Male", "Female"), "education": ("High", "Medium", "Low"),
    "marital_status": ("Married/partnered", "Separated/divorced", "Widowed", "Never married"),
    "labour_force_status": ("Working", "Unemployed", "Retired or partly retired", "Other economically inactive"),
    "current_smoking": ("False", "True"), "wealth_quintile": ("5", "1", "2", "3", "4"),
}
FORMULA = (
    "fi_primary ~ recovery * time_years + age_z + household_size_z + household_income_rank_z + wealth_loss_fraction_z + fi_t_z "
    "+ C(sex, Treatment(reference='Male')) + C(education, Treatment(reference='High')) "
    "+ C(marital_status, Treatment(reference='Married/partnered')) "
    "+ C(labour_force_status, Treatment(reference='Working')) "
    "+ C(current_smoking, Treatment(reference='False')) "
    "+ C(wealth_quintile, Treatment(reference='5')) "
    "+ age_z:time_years + household_size_z:time_years + household_income_rank_z:time_years + wealth_loss_fraction_z:time_years + fi_t_z:time_years "
    "+ C(sex, Treatment(reference='Male')):time_years "
    "+ C(education, Treatment(reference='High')):time_years "
    "+ C(marital_status, Treatment(reference='Married/partnered')):time_years "
    "+ C(labour_force_status, Treatment(reference='Working')):time_years "
    "+ C(current_smoking, Treatment(reference='False')):time_years "
    "+ C(wealth_quintile, Treatment(reference='5')):time_years"
)
POLICY = {
    "maxiter": 2000, "loglik_optimizer_atol": 1e-4, "contrast_optimizer_atol": 1e-6,
    "core_se_optimizer_rtol": 1e-3, "interior_scaled_gradient_tolerance": 1e-3,
    "near_singular_eigenvalue_ratio": 1e-6, "correlation_review_threshold": 0.99,
    "primary_reconstruction_atol": 1e-10, "scale_test_rtol": 1e-10,
    "scale_test_atol": 1e-12, "scale_test_loglik_atol": 1e-8,
    "random_time_divisor": 8.0, "rs_start_slope_unscaled_variance": 0.1,
    "psd_relative_roundoff_tolerance": 1e-12,
}


def scientific_spec():
    return {
        "specifications": {
            "RS01": {"formula": FORMULA, "random_design": ["1", "time_years / 8"],
                     "random_covariance": "unstructured_2x2", "fixed_columns": 42},
            "NL01": {"formula": FORMULA + " + rcs_time + recovery:rcs_time",
                     "random_design": ["1"], "random_covariance": "random_intercept",
                     "fixed_columns": 44, "knots": KNOTS,
                     "basis": "restricted_cubic_explicit_linear_plus_one_nonlinear",
                     "covariate_time_policy": "original_covariate_x_linear_time_unchanged"},
        },
        "sample": "unchanged_primary_complete_case_eligible",
        "outcome": "fi_primary", "fit": "unweighted_ML", "reml": False,
        "standardization": "frozen_unique_primary_participant_mean_and_sample_sd",
        "time": "actual_interview_elapsed_years; landmark=0; all_frozen_later_rows",
        "anchors": COHORT_ANCHORS, "policy": POLICY,
        "optimizer": {"sequence": ["bfgs", "lbfgs"], "use_sqrt": True,
                      "lbfgs_start": "bfgs.params_object; do_not_reinterpret_packed_covariance_as_Cholesky",
                      "initial_covariance": "primary_tau2/sigma2; RS01 slope=0.1, covariance=0",
                      "automatic_retry_or_simplification": False},
        "contrasts": {"grid": {"start": 0.0, "stop": 8.0, "step": 0.25},
                      "objects": ["mu_0", "mu_1", "D", "C", "S", "A_0_8", "group_average_slopes_0_8"],
                      "ci": "model_based_normal_Wald_pointwise_95",
                      "nonlinear_tests": [["rcs_time", "recovery:rcs_time"],
                                          ["recovery:rcs_time"], ["recovery:time_years", "recovery:rcs_time"]]},
        "failure_policy": "HOLD_NOT_ESTIMABLE_or_BOUNDARY_REVIEW; no automatic PASS_WITH_CAVEAT",
        "stage3_required_review": ["nonlinearity_and_residual_shape", "A_0_8_and_CI_changes",
                                   "time_support_and_sparse_tail", "primary_interpretation", "boundary_and_identifiability"],
        "scope": {"six_models_only": True, "primary_refit": False, "meta": False,
                  "combined_model": False, "freeze_or_release_code_in_stage0_2": False,
                  "candidate_tables_figures_code_in_stage0_2": False, "participant_level_exports": False},
    }
