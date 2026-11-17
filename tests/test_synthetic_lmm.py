"""使用明确合成数据测试原42列ML随机截距拟合；不代表原队列复估。"""
import numpy as np
import pandas as pd
import patsy
import run_section_3_2 as primary


def test_primary_fit_on_synthetic_longitudinal_data():
    rng = np.random.default_rng(4603924)
    rows = []
    for i in range(160):
        person = {'person_id': f'SYNTHETIC_{i}', 'recovery': i % 2,
                  'wealth_loss_fraction': .8 + rng.random()}
        for col in primary.CONTINUOUS:
            person[col+'_z'] = rng.normal()
        for col, levels in primary.CATEGORICAL.items():
            person[col] = str(rng.choice(levels))
        random_intercept = rng.normal(0,.035)
        for time in [0.,2.,4.,6.]:
            rows.append({**person, 'time_years':time, 'fi_primary':.2,
                         '_random_intercept':random_intercept})
    frame = pd.DataFrame(rows)
    _, design = patsy.dmatrices(primary.FORMULA, frame, return_type='dataframe')
    beta = np.zeros(design.shape[1])
    beta[design.columns.get_loc('Intercept')] = .22
    beta[design.columns.get_loc('time_years')] = .002
    beta[design.columns.get_loc('recovery:time_years')] = .0013
    frame['fi_primary'] = design.to_numpy() @ beta + frame['_random_intercept'] + rng.normal(0,.015,len(frame))
    _, _, estimates, _, diagnostics = primary.fit_model(frame, 'SYNTHETIC')
    row = estimates.loc[estimates.is_primary_estimand].iloc[0]
    assert diagnostics['design_matrix_rank'] == 42
    assert diagnostics['bfgs_converged'] and diagnostics['lbfgs_converged']
    assert row.ci_lower < .0013 < row.ci_upper
    assert diagnostics['n_participants'] == 160
