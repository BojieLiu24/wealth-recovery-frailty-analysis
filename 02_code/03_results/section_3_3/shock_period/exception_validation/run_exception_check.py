"""只验证HRS/SP01已存候选；保持原始HOLD，不实施完整Stage3。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datetime import datetime, UTC
import json
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import norm
from independent_numerics import (Blocks, inverse_information, finite_difference_profile,
    log_variance_coordinates, forbid_fitting)
from sp_contract import (ROOT, CONTRACT, read_json, write_json, sha, rel,
    verify_contract, content_sha, prepare)
from sp_audit import verify_records
from common import check_protected, provenance


def run(folder):
    folder = Path(folder).resolve()
    if (folder/'check_manifest.json').exists():
        raise ValueError('拒绝覆盖已经完成的检查')
    policy = read_json(folder/'validation_policy.json')
    tol = policy['independent_checks']; limits = policy['inherited_contract']
    runroot = ROOT/'03_output/3.3 Robustness of the Main Findings/Shock Calendar Period/runs/round1e_stage2_20261015_01'
    unit = runroot/'HRS/SP01'
    m = read_json(runroot/'execution_manifest.json')
    verify_records(m['outputs'])
    assert sha(CONTRACT) == 'c2a3ba0b75047f52a7bd4e3377a38f7368cc830b3fa3082d364fd0655573a5ce'
    contract = verify_contract()
    assert contract['preflight']['manifest_sha256'] == '9cbab81f997aa43cd59ccc8af1b1d7671a708c1042805c997abbd0da0cdeb986'
    assert content_sha(contract['scientific_specification']) == 'ce120d99e278aef4b6c0e33881fee682c643637ccae830138a90ddc0a8e72306'
    tests = ET.parse(folder/'necessary_tests.xml').getroot()
    suites = tests.findall('.//testsuite') if tests.tag != 'testsuite' else [tests]
    testcounts = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    assert testcounts['tests'] > 0 and all(testcounts[k] == 0 for k in ('failures', 'errors', 'skipped'))
    original = read_json(unit/'result_object.json')
    assert original['status'] == 'HOLD_NOT_ESTIMABLE' and original['failure_reason_codes'] == ['OPTIMIZER_NOT_CONVERGED']
    checkrows, arrays, outcomes = [], {}, {}
    counter = {'forbidden_attempts': 0}
    def record(name, passed, details):
        checkrows.append({'check': name, 'passed': bool(passed), 'details': details})
    def near(name, actual, target, rtol, atol):
        actual, target = np.asarray(actual), np.asarray(target)
        record(name, np.allclose(actual, target, rtol=rtol, atol=atol),
            {'maximum_absolute_error': float(np.max(abs(actual-target))), 'rtol': rtol, 'atol': atol})
    with forbid_fitting(counter):
        stage0 = read_json(ROOT/'03_output/3.3 Robustness of the Main Findings/Shock Calendar Period/Audit/round1e_stage0_20261014_01/design_diagnostics.json')
        frame, x, design, *_ = prepare('HRS', stage0)
        expected = contract['scientific_specification']['cohorts']['HRS']
        assert design['anchor'] == expected['anchor'] == original['sample']
        assert design['anchor']['ordered_person_wave_sha256'] == 'faa44fe754ce69f5ba0852d75c28f5a241d7d4362c0e1258d7af7b6c0425bf43'
        assert design['columns'] == expected['columns'] and design['rank'] == 64
        assert design['levels'] == list(range(3, 15)) and design['reference'] == 3
        record('sample_model_identity', True, {'anchor': design['anchor'], 'rank': 64,
            'original42_columns': design['primary_columns'], 'columns': design['columns'],
            'standardization_sha256': contract['scientific_specification']['standardization_constants_sha256'],
            'time': contract['scientific_specification']['time'], 'weighted': False, 'reml': False,
            'random_effects': 'participant_random_intercept', 'additional_panel_saved': False,
            'shared_design_builder_used': True, 'numerical_likelihood_derivatives_independent_of_builder': True})
        blocks = Blocks(frame.fi_primary.to_numpy(float), x.to_numpy(float), frame.person_id.to_numpy())
        names = list(x); p = len(names); j = names.index('recovery:time_years')
        for method in ('bfgs', 'lbfgs'):
            # Stage2候选的列名为object数组；先验证原始文件SHA再读取此既有元数据。
            saved = np.load(unit/(method+'_candidate.npz'), allow_pickle=True)
            reference = np.load(unit/('bfgs_numerics.npz' if method == 'bfgs' else 'model_parameters.npz'), allow_pickle=False)
            logs = read_json(unit/(method+'_optimization.json'))
            beta = saved['fixed_effects']; q = float(saved['random_covariance_unscaled'][0, 0])
            sigma = float(saved['residual_variance']); tau = float(saved['random_covariance'][0, 0])
            assert saved['fixed_effect_names'].tolist() == names
            np.testing.assert_array_equal(saved['parameters'][:-1], beta)
            assert float(saved['parameters'][-1]) == q
            profile = blocks.profile_from_physical(beta, q)
            physical = blocks.physical(beta, tau, sigma, whiten=True)
            cov_profile, spectrum_profile = inverse_information(profile['hessian'])
            cov_full, spectrum_full = inverse_information(physical['hessian'])
            _, Hlog = log_variance_coordinates(physical['score'], physical['hessian'], tau, sigma)
            cov_log, spectrum_log = inverse_information(Hlog)
            near(method+'_profile_score_vs_saved', profile['score'], reference['score'], tol['score_rtol'], tol['score_atol'])
            near(method+'_profile_Hessian_vs_saved', profile['hessian'], reference['hessian'], tol['hessian_rtol'], tol['hessian_atol'])
            for tag, cov in (('profile', cov_profile), ('physical_full', cov_full), ('log_variance', cov_log)):
                near(method+'_'+tag+'_fixed_covariance_vs_saved', cov[:p, :p], reference['fixed_effect_covariance'], tol['fixed_covariance_rtol'], tol['fixed_covariance_atol'])
            ll_woodbury = blocks.woodbury_profile(beta, q)[0]
            llerror = max(abs(physical['loglik']-logs['log_likelihood']), abs(ll_woodbury-logs['log_likelihood']))
            record(method+'_independent_likelihood', llerror <= tol['loglik_absolute_error_max'],
                {'dense_loglik': physical['loglik'], 'Woodbury_profile_loglik': ll_woodbury,
                'saved_loglik': logs['log_likelihood'], 'max_abs_error': llerror, 'threshold': tol['loglik_absolute_error_max']})
            near(method+'_sigma_profile', profile['sigma'], sigma, 0., tol['variance_profile_absolute_error_max'])
            near(method+'_scaled_variance_mapping', tau, q*sigma, 1e-12, 1e-14)
            Q, R = np.linalg.qr(physical['wx'], mode='reduced')
            beta_qr = solve_triangular(R, Q.T@physical['wy'])
            u, s, vh = np.linalg.svd(physical['wx'], full_matrices=False)
            # 满秩SVD只是线性方程求解，不为不可逆信息构造伪逆。
            assert np.min(s) > 0
            beta_svd = vh.T@((u.T@physical['wy'])/s)
            near(method+'_QR_vs_saved_beta', beta_qr, beta, 0., tol['QR_saved_beta_absolute_error_max'])
            near(method+'_QR_vs_SVD_beta', beta_qr, beta_svd, 0., tol['QR_SVD_beta_absolute_error_max'])
            fd_results = []
            for step in tol['finite_difference_scaled_steps']:
                FD, error = finite_difference_profile(blocks, np.r_[beta, q], profile['hessian'], step)
                record(method+f'_finite_difference_step_{step}', error <= tol['finite_difference_scaled_Hessian_absolute_error_max'],
                    {'balanced_Hessian_max_abs_error': error, 'step': step, 'threshold': tol['finite_difference_scaled_Hessian_absolute_error_max']})
                fd_results.append(error); arrays[method+f'_FD_scaled_Hessian_{step}'] = FD
            scaled = float(np.max(abs(profile['score'])/np.sqrt(np.diag(-profile['hessian']))))
            scaled_full = float(np.max(abs(physical['score'])/np.sqrt(np.diag(-physical['hessian']))))
            record(method+'_first_second_order', max(scaled, scaled_full) <= limits['interior_scaled_gradient_max'],
                {'profile_scaled_gradient': scaled, 'physical_full_scaled_gradient': scaled_full,
                'threshold': limits['interior_scaled_gradient_max'], 'profile_and_full_information_PD': True,
                'Hessian_sign': 'strictly_negative_at_interior_maximum'})
            for tag, spec in (('profile', spectrum_profile), ('physical_full', spectrum_full), ('log_variance', spectrum_log)):
                record(method+'_'+tag+'_inverse_backward_error', spec['inverse_backward_error'] <= tol['information_inverse_backward_error_max'], spec)
            se = float(np.sqrt(cov_full[j, j])); savedse = float(np.sqrt(reference['fixed_effect_covariance'][j, j]))
            record(method+'_core_SE', abs(se-savedse)/savedse <= tol['core_se_relative_error_max'],
                {'independent': se, 'saved': savedse, 'relative_error': abs(se-savedse)/savedse})
            core = {'estimate': float(beta_qr[j]), 'se': se, 'ci_lower': float(beta_qr[j]-norm.ppf(.975)*se),
                'ci_upper': float(beta_qr[j]+norm.ppf(.975)*se), 'p_value': float(2*norm.sf(abs(beta_qr[j]/se)))}
            terminal = logs['terminal']
            record(method+'_native_termination_identity', terminal['complete'] and (
                terminal['status'] == 2 and 'precision loss' in terminal['message'] if method == 'bfgs' else
                terminal['status'] == 0 and terminal['task'] == 'CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH'), terminal)
            native = next(e for e in logs['events'] if e['kind'] == 'scipy_internal')
            require_settings = native['effective_arguments']['maxiter'] == 2000 and native['effective_arguments']['gtol'] == (1e-8 if method == 'bfgs' else 1e-10)
            if method == 'lbfgs': require_settings &= native['effective_arguments']['ftol'] == 100*np.finfo(float).eps
            record(method+'_effective_settings', require_settings, native['effective_arguments'])
            outcomes[method] = {'core': core, 'loglik': physical['loglik'], 'sigma2': sigma, 'tau2': tau,
                'q': q, 'parameter_vector': saved['parameters'].tolist(), 'physical_parameters': np.r_[beta, tau, sigma].tolist(),
                'profile_scaled_gradient': scaled, 'physical_scaled_gradient': scaled_full,
                'profile_information': spectrum_profile, 'full_physical_information': spectrum_full,
                'log_variance_information': spectrum_log, 'whitened_design_condition': float(s[0]/s[-1]),
                'full_fixed_covariance_eigenvalues': np.linalg.eigvalsh(cov_full[:p, :p]).tolist(),
                'finite_difference_errors': fd_results, 'terminal': terminal, 'warnings': logs['warnings'],
                'saved_LMM_covariance_includes_variance_nuisance': True}
            for tag, value in {'profile_score': profile['score'], 'profile_Hessian': profile['hessian'],
                'profile_covariance': cov_profile, 'physical_score': physical['score'],
                'physical_Hessian': physical['hessian'], 'physical_covariance': cov_full,
                'log_variance_Hessian': Hlog, 'log_variance_covariance': cov_log,
                'QR_beta': beta_qr, 'SVD_beta': beta_svd}.items(): arrays[method+'_'+tag] = value
        a, b = outcomes['bfgs'], outcomes['lbfgs']
        comparison = {'loglik_absolute_difference': abs(a['loglik']-b['loglik']),
            'core_beta_absolute_difference': abs(a['core']['estimate']-b['core']['estimate']),
            'core_se_relative_difference': abs(a['core']['se']-b['core']['se'])/a['core']['se'],
            'max_packed_parameter_absolute_difference': float(np.max(abs(np.array(a['parameter_vector'])-b['parameter_vector']))),
            'max_fixed_beta_absolute_difference': float(np.max(abs(np.array(a['parameter_vector'][:-1])-b['parameter_vector'][:-1]))),
            'q_absolute_difference': abs(a['q']-b['q']), 'tau2_absolute_difference': abs(a['tau2']-b['tau2']),
            'sigma2_absolute_difference': abs(a['sigma2']-b['sigma2']), 'first_converged': False, 'second_converged': True,
            'BFGS_start_then_LBFGS_continuation_not_independent_multistart': True}
        record('optimizer_candidate_agreement', comparison['loglik_absolute_difference'] < limits['loglik_optimizer_absolute_difference_strict'] and
            comparison['core_beta_absolute_difference'] < limits['core_beta_absolute_difference_strict'] and
            comparison['core_se_relative_difference'] <= limits['core_se_relative_difference_max'], comparison)
    assert counter['forbidden_attempts'] == 0
    verify_records(m['outputs']); verify_records(contract['sources'])
    protection = check_protected(read_json(folder/'protected_assets_before.json'))
    decision = 'NUMERICAL_EXCEPTION_SUPPORTABLE' if all(r['passed'] for r in checkrows) else 'ADDITIONAL_VERIFICATION_REQUIRED'
    result = {'check_id': folder.name, 'decision': decision, 'checks': checkrows, 'candidates': outcomes,
        'fit_calls': 0, 'optimizer_calls': 0, 'forbidden_attempts': counter['forbidden_attempts'],
        'independence': {'source_design_loader': 'shared frozen-data/design interfaces',
            'likelihood_derivatives_covariance': 'independent NumPy/SciPy linear algebra; no statsmodels likelihood/score/Hessian calls',
            'finite_differences': 'Woodbury score distinct from dense block analytic derivatives',
            'QR_SVD': 'fixed saved covariance linear solve used for beta check only; GLS covariance never substitutes final SE'},
        'original_HOLD_preserved': True, 'author_numerical_exception_approval': 'PENDING',
        'scientific_review_and_full_Stage3': 'NOT_EXECUTED', 'protected_check': protection,
        'tests': testcounts, 'policy_sha256': sha(folder/'validation_policy.json')}
    write_json(folder/'independent_numeric_validation.json', result)
    write_json(folder/'optimizer_comparison.json', {'candidates': outcomes, 'comparison': comparison,
        'original_status': 'HOLD_NOT_ESTIMABLE', 'author_exception_approval': 'PENDING'})
    with (folder/'independent_numeric_evidence.npz').open('xb') as f:
        np.savez_compressed(f, fixed_effect_names=np.array(names), **arrays)
    failed = [r['check'] for r in checkrows if not r['passed']]
    lb = outcomes['lbfgs']; core = lb['core']
    report = ['# HRS/SP01定向数值例外核查', '', f'检查ID：{folder.name}；结论：**{decision}**。',
        '原Stage2 HOLD保留；本核查不构成作者数值例外批准或科学批准。', '',
        '## 独立数值结果', '', f"L-BFGS候选：β={core['estimate']:.17g}；SE={core['se']:.17g}；95%CI=[{core['ci_lower']:.17g}, {core['ci_upper']:.17g}]；P={core['p_value']:.17g}。",
        '', '| 候选 | 独立loglik | 剖面标准化梯度 | 全物理参数标准化梯度 | 信息原条件数 | 均衡后条件数 |',
        '|---|---:|---:|---:|---:|---:|']
    for method, o in outcomes.items():
        info = o['profile_information']
        report.append(f"| {method} | {o['loglik']:.15g} | {o['profile_scaled_gradient']:.7g} | {o['physical_scaled_gradient']:.7g} | {info['raw_condition']:.8g} | {info['balanced_condition']:.8g} |")
    report += ['', '## 检查证据及误差', '', '| 检查 | 通过 | 数值证据 |', '|---|---|---|']
    for r in checkrows:
        detail = r['details']
        if 'maximum_absolute_error' in detail: evidence = f"最大绝对误差={detail['maximum_absolute_error']:.9g}; rtol={detail['rtol']}; atol={detail['atol']}"
        elif 'balanced_Hessian_max_abs_error' in detail: evidence = f"尺度化Hessian误差={detail['balanced_Hessian_max_abs_error']:.9g}"
        elif 'max_abs_error' in detail: evidence = f"似然绝对误差={detail['max_abs_error']:.9g}"
        elif 'inverse_backward_error' in detail: evidence = f"求逆后向误差={detail['inverse_backward_error']:.9g}"
        else: evidence = '完整精度见independent_numeric_validation.json'
        report.append(f"| {r['check']} | {r['passed']} | {evidence} |")
    report += ['', f'失败检查：{failed}。', '', '## 异常性质与判断', '',
        'BFGS原生status=2，准确消息为Desired error not necessarily achieved due to precision loss；L-BFGS status=0，按RELATIVE REDUCTION OF F <= FACTR*EPSMCH终止。两者未被混称为双收敛。',
        f"独立双候选似然差={comparison['loglik_absolute_difference']:.9g}，核心β差={comparison['core_beta_absolute_difference']:.9g}，核心SE相对差={comparison['core_se_relative_difference']:.9g}。",
        '两优化器沿同一起点接续，候选一致不是独立多起点或全局最优的证明；独立矩阵证据只支持保存候选附近的内点驻点与局部最大值。',
        '完整观察信息为正定，故最大化似然的Hessian为负定；不能把这种负特征值误称为异常。信息正定、合法协方差和无伪逆均以完整矩阵及谱保存。条件数须结合参数尺度判断；没有为原1.24e5设置事后通过线。',
        '参数坐标分别为优化器内部sqrt(q)、保存的(β,q)、物理(β,τ²,σ²)和对数方差坐标。q=τ²/σ²，剖面σ²=Q/n；坐标变换保留非零score的二阶曲率项。',
        '完整物理信息包含两项方差参数，通过完整矩阵求逆得到固定效应协方差；不使用固定方差GLS协方差作最终SE，不裁剪或伪逆。',
        '密集逐人Cholesky、Woodbury似然与score、两个预定步长的五点有限差分、QR/SVD点估计及对数方差坐标核查相互补充。所有独立门槛均在真实核查前保存，没有看结果后放宽。', '',
        '## 样本、独立性及保护', '',
        'HRS 5,114人、23,411行，SP01 64/64满秩；原42列保留，波次W3–W14、参照W3；时间、FI、恢复定义、冻结常数、无权重ML与随机截距结构保持。',
        '有序person-wave SHA：faa44fe754ce69f5ba0852d75c28f5a241d7d4362c0e1258d7af7b6c0425bf43。',
        '来源和设计重建共享既有只读加载器；数值似然、导数、信息和协方差未调用statsmodels实现。本核查不构成原始变量独立重建或完整Stage3科学QA。',
        f"必要测试{testcounts['tests']}项通过；核查真实fit=0、优化器=0、禁用调用尝试=0。测试中另有一次被守卫拦截的合成调用，用于验证封装恢复；没有执行底层优化器。",
        f"执行前后{protection['files_checked']}项既有资产SHA不变；合同来源{len(contract['sources'])}项复核通过。原始候选、日志、HOLD、合同、Stage0/1及Round1A–1D资产保持。", '',
        '## 停止点', '', '可支持与否以本报告结论为准；作者接受例外仍须新建独立批准记录，不改写原HOLD。未重新拟合，未开展ELSA/KLoSA QA，未进入Stage4/5，未修改论文或commit/push。', '']
    with (folder/'exception_check_report.md').open('x', encoding='utf-8', newline='\n') as f: f.write('\n'.join(report))
    decision_text = f'# 数值例外准备判断\n\n建议：**{decision}**。\n\n失败检查：{failed}。\n\n'
    if decision == 'NUMERICAL_EXCEPTION_SUPPORTABLE':
        decision_text += '独立似然、score、完整观察信息及协方差与两候选一致；现有问题局限于BFGS precision loss终止状态。已有证据支持向作者提交保存L-BFGS候选的限定数值例外，不需要为了本异常重新拟合。\n\n'
    else:
        decision_text += '现有核查尚不足以支持数值例外。最低必要追加工作是定位列出的失败检查及独立实现误差；不得直接放宽门槛或重新拟合。\n\n'
    decision_text += '作者批准状态：PENDING。原HOLD_NOT_ESTIMABLE保持。数值例外不替代三队列后续独立QA、晚期支持和科学审阅，不授权冻结或论文应用。\n'
    with (folder/'exception_readiness_decision.md').open('x', encoding='utf-8', newline='\n') as f: f.write(decision_text)
    snapshots = folder/'code_snapshot'; snapshots.mkdir(exist_ok=False)
    for code in Path(__file__).parent.glob('*.py'): shutil.copyfile(code, snapshots/code.name)
    sources = {**contract['sources'], rel(CONTRACT): sha(CONTRACT),
        contract['preflight']['path']: contract['preflight']['manifest_sha256'],
        rel(runroot/'execution_manifest.json'): sha(runroot/'execution_manifest.json'), **m['outputs']}
    write_json(folder/'check_manifest.json', {'schema': 'round1e_HRS_SP01_exception_check_v1',
        'check_id': folder.name, 'decision': decision, **provenance(), 'recorded_utc': datetime.now(UTC).isoformat(),
        'sources': sources, 'code': {rel(p): sha(p) for p in Path(__file__).parent.glob('*.py')},
        'outputs': {rel(p): sha(p) for p in folder.rglob('*') if p.is_file()},
        'protected_check': protection, 'fit_calls': 0, 'optimizer_calls': 0,
        'forbidden_attempts': 0, 'original_HOLD_unchanged': True, 'author_approval': 'PENDING',
        'stage3_full_qa_executed': False, 'freeze_executed': False, 'git_commit_or_push': False})
    print(json.dumps({'decision': decision, 'failed_checks': failed, 'core': core,
        'comparison': comparison, 'check_id': folder.name, 'protected_files': protection['files_checked']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8'); sys.stderr.reconfigure(encoding='utf-8')
    run(sys.argv[1])
