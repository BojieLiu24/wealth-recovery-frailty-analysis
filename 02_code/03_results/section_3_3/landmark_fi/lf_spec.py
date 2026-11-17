"""唯一科学规格及解释边界；不依据新结果选模型。"""
from lf_common import COHORTS, SPECS
from specification import FORMULA, CONTINUOUS, CATEGORIES, COHORT_ANCHORS

LANDMARK_CONSTANTS = {
    'HRS': (0.27590343448876653, 0.18036326417461707),
    'ELSA': (0.24084156700517512, 0.16453790549710362),
    'KLoSA': (0.14472603690446656, 0.12056811682128088),
}
SINGLE_COUNTS = {'HRS': 1196, 'ELSA': 207, 'KLoSA': 503}
LM_MAIN = 'fi_landmark_z'
LM_TIME = 'fi_landmark_z:time_years'
Q1 = "C(wealth_quintile, Treatment(reference='5'))[T.1]"
Q1_TIME = Q1 + ':time_years'
Q1_COMBINED = 'Q1_effect_at_23_over_12'
LM_STATEMENT = ('LM01 is a post-landmark conditional association analysis, not an alternative '
                'estimate of a causally identified effect of wealth recovery.')
POLICY = {'maxiter': 2000, 'bfgs_gtol': 1e-8, 'lbfgs_pgtol': 1e-10, 'lbfgs_factr': 100.,
          'loglik_optimizer_atol_exclusive': 1e-4, 'core_beta_optimizer_atol_exclusive': 1e-6,
          'core_se_optimizer_rtol': 1e-3, 'interior_scaled_gradient_tolerance': 1e-3,
          'primary_reconstruction_atol': 1e-10, 'equivalence_rtol': 1e-10,
          'equivalence_atol': 1e-12, 'synthetic_loglik_atol': 1e-8,
          'boundary_unscaled_variance_ratio': 1e-6, 'automatic_retry': False}


def interpretation(spec):
    statements = {
        'Primary': '在原调整条件下，使用landmark及后续FI估计的组间平均FI年变化速度差。',
        'PO01': '在相同原调整条件下，仅使用landmark之后FI观察估计的组间年变化速度差。',
        'LM01': '进一步条件于landmark FI及其与时间的关联后的组间年变化速度差。',
    }
    return {'core_estimand': statements[spec], 'units': 'FI units/year',
            'direction': 'substantial_recovery_minus_persistent_or_limited_recovery',
            'conditional_association_statement': LM_STATEMENT if spec == 'LM01' else None,
            'causally_identified': False,
            'intercept_role': 'extrapolated_nuisance_not_observed_landmark_mean' if spec != 'Primary' else 'frozen_reference',
            'recovery_main_role': 'extrapolated_nuisance_not_observed_landmark_difference' if spec != 'Primary' else 'frozen_reference',
            'landmark_level_claims_allowed': False, 'level_trajectory_plots_allowed': False,
            'coefficient_difference_is_mediation_or_reverse_causality_contribution': False,
            'can_be_called_more_confounder_control': False if spec == 'LM01' else None}


def scientific_spec():
    return {'specs': {s: {'formula': FORMULA if s == 'PO01' else FORMULA + ' + ' + LM_MAIN + ' + ' + LM_TIME,
                         'nominal_columns': 42 if s == 'PO01' else 44,
                         'basis_columns': {c: (41 if s == 'PO01' else 43) if c == 'KLoSA' else (42 if s == 'PO01' else 44) for c in COHORTS},
                         'interpretation': interpretation(s)} for s in SPECS},
            'primary_reference_interpretation': interpretation('Primary'),
            'sample': 'unchanged_primary_participants_including_single_post_observation',
            'rows': 'fi_time_role=later AND time_years>0; PO01 and LM01 identical ordered person-wave keys',
            'time': 'actual elapsed years since original landmark; no recentering',
            'outcome': 'fi_primary from frozen Analysis Panel v1', 'reml': False,
            'random_design': '[1]', 'random_effects': 'participant_random_intercept', 'weighting': 'unweighted',
            'original_standardization': 'frozen_primary_unique_person_means_and_sample_SDs',
            'landmark_standardization': {c: {'mean': m, 'sample_sd': sd, 'mean_hex': m.hex(),
                'sample_sd_hex': sd.hex(), 'ddof': 1, 'unique_primary_participants': COHORT_ANCHORS[c]['participants']} for c, (m, sd) in LANDMARK_CONSTANTS.items()},
            'categories': CATEGORIES, 'continuous': list(CONTINUOUS),
            'parameterization': {'KLoSA': 'Q1*T=(23/12)*Q1; kappa=gamma+(23/12)*eta; retain all rows and covariate information',
                                 'extra_rank_deficiency': 'STOP_AND_REPORT', 'independent_Q1_coefficients_allowed': False},
            'optimizer': {'sequence': ['bfgs', 'lbfgs'], 'policy': POLICY, 'science_change': False,
                          'start': 'frozen Primary beta mapped to basis; added LM beta=0; frozen tau2/sigma2',
                          'lbfgs_start': 'bfgs.params_object.copy(); preserve covariance parameterization'},
            'comparison': {'order': ['Primary', 'PO01', 'LM01'], 'cross_window_loglik_AIC_comparison': False,
                           'model_selection_by_significance': False, 'coefficient_difference_inference': 'descriptive_only'},
            'stage3_required_diagnostics': {'time_bin_years': 2, 'tail_years': 8,
                'tail_is_model_truncation': False, 'residual_and_support_summaries': True,
                'same_window_LM01_vs_PO01_nested_loglik_tolerance': 1e-4,
                'fixed_covariance_influence_screen': 'e_RT.T @ inv(sum(Xi.T Vi^-1 Xi)) @ Xi.T Vi^-1 ri; no variance refit',
                'influence_screen_is_final_LMM_SE': False},
            'later_stages': {'freeze_requires_specific_result_approval': True,
                'candidate_table_rows': 9, 'candidate_language': 'English', 'new_plots': False},
            'additional_delta_model': False, 'additional_RS_NL_or_meta': False}
