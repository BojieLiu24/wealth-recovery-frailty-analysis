"""Round1C Stage3独立入口；拒绝覆盖，禁止拟合、优化、Freeze及制表。"""
import argparse
import hashlib
from pathlib import Path
import sys
import traceback
import numpy as np
import patsy
from lf_common import ROOT,CODE,PROGRESS,RUNS,CONTRACT,TERM,read_json,write_json,write_csv,sha,rel,require,assert_hashes,check_protected,provenance
from run_landmark_fi import verify_lock,safe_id
from lf_inputs import verify_sources
from lf_spec import LM_MAIN,LM_TIME,Q1,Q1_TIME,Q1_COMBINED,LM_STATEMENT
from prepare_inputs import load_cohort,reconstruct_primary
from resolve_hrs_po01_exception import prohibit_optimization
from stage3_numerics import validate_anchor,validate_basis,validate_candidate,wald
from stage3_diagnostics import support_and_residuals,influence_screen
from stage3_review import compare_rows,round1b_reference,reports

EXPECTED_CONTRACT='57fabb54d896cd0ea6ca693f0e5d8396cce58eaead3a095bdd45491691e46c5d'
EXPECTED_PREFLIGHT='4ecdb7805f67e6f7c923d3fff635e22a83609686b86515e14fc5e0d998e0249d'
EXPECTED_EXECUTION='5229242a00e4d1acad70af7f34e7bb97e58a719c6f1f9abb97478b464e9773e5'
QA_ROOT=ROOT/'03_output/3.3 Robustness of the Main Findings/Landmark FI Sensitivity/QA'


def scope_guard(action):
    require(action in {'validate','diagnose','review'},'Stage3禁止拟合、Freeze或候选制表')


def verify_digest(path,digest):
    require(path.is_file() and sha(path)==digest,'输入SHA不符，停止')


def row_hash(frame):
    return hashlib.sha256(frame[['person_id','wave']].to_json(orient='values',force_ascii=False).encode('utf-8')).hexdigest()


def check_exception(approval,contract,run,check):
    require(approval['disposition']=='AUTHOR_ACCEPTED_NUMERICAL_EXCEPTION' and
            approval['cohort']=='HRS' and approval['specification']=='PO01' and
            approval['accepted_for_stage3'] is True and approval['original_status_changed'] is False,
            'HRS数值例外范围不符')
    require(approval['contract_sha256']==sha(CONTRACT) and approval['preflight_manifest_sha256']==contract['audit']['manifest_sha256'],
            '例外未绑定当前合同及Preflight')
    require(approval['original_run_id']==run.name and approval['scientific_result_approval']=='PENDING' and
            approval['freeze_authorization']=='NOT_GRANTED','数值例外不构成科学或冻结批准')
    assert_hashes(approval['source_hashes'])
    require(check['BFGS_and_final_parameters_identical'] and check['original_status']=='HOLD_NOT_ESTIMABLE', '独立例外证据身份不符')


def rebuild(cohort,contract):
    # 只读复用已核验的冻结Panel加载器；固定矩阵另按合同公式重新生成。
    frame,lm,source_x,beta,cov,diag,anchor=load_cohort(cohort)
    primary_check=reconstruct_primary(frame,source_x,beta,diag)
    primary_x=patsy.dmatrix(contract['scientific_specification']['specs']['PO01']['formula'].split('~',1)[1],
                           frame,NA_action='raise',return_type='dataframe')
    require(list(primary_x)==list(source_x),'Primary公式列名漂移')
    np.testing.assert_array_equal(primary_x,source_x)
    post=frame.loc[frame.fi_time_role.eq('later')].copy()
    require(post.time_years.gt(0).all() and lm.time_years.eq(0).all(),'时点或原点错误')
    require(set(post.person_id)==set(lm.person_id) and not post[['person_id','wave']].duplicated().any(),'样本或键错误')
    constants=contract['scientific_specification']['landmark_standardization'][cohort]
    require(float(lm.fi_primary.mean()).hex()==constants['mean_hex'] and
            float(lm.fi_primary.std(ddof=1)).hex()==constants['sample_sd_hex'],'Landmark中心尺度漂移')
    post[LM_MAIN]=(post.person_id.map(lm.set_index('person_id').fi_primary)-constants['mean'])/constants['sample_sd']
    actual={**anchor,'primary_rows':anchor['rows'],'rows':len(post),'ordered_person_wave_sha256':row_hash(post),'landmark_in_outcome':False}
    validate_anchor(actual,contract['cohort_anchors'][cohort])
    matrices={}
    for spec in ['PO01','LM01']:
        formula=contract['scientific_specification']['specs'][spec]['formula']
        original=patsy.dmatrix(formula.split('~',1)[1],post,NA_action='raise',return_type='dataframe')
        if cohort=='KLoSA':
            basis=original.drop(columns=[Q1_TIME]).rename(columns={Q1:Q1_COMBINED})
        else:basis=original.copy()
        matrices[spec]=(original,basis)
    return post,matrices,actual,wald(beta,cov,list(primary_x)),anchor,diag,primary_check


def write_text(path,text):
    with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)


def load_parameters(path):
    # parameter_names为历史对象数组，不参与核查，也不启用pickle反序列化。
    keys=['fixed_effect_names','fixed_effects','fixed_effect_covariance','parameters',
          'random_covariance_scaled','random_covariance_unscaled','residual_variance','hessian']
    with np.load(path,allow_pickle=False) as values:
        return {key:values[key].copy() for key in keys}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--qa-id',required=True)
    parser.add_argument('--instruction',type=Path,required=True);args=parser.parse_args()
    scope_guard('validate');qa_id=safe_id(args.qa_id)
    instruction=args.instruction.read_text(encoding='utf-8-sig')
    require('本次作者明确授权执行Round 1C Stage 3' in instruction and '不调用优化器' in instruction,'缺少本轮Stage3明确授权')
    require(not (QA_ROOT/qa_id).exists() and not (PROGRESS/'Stage3'/qa_id).exists(),'拒绝覆盖已有QA')
    contract=verify_lock()
    verify_digest(CONTRACT,EXPECTED_CONTRACT)
    require(contract['audit']['manifest_sha256']==EXPECTED_PREFLIGHT,'锁定身份不匹配')
    sources=verify_sources()
    run=RUNS/'round1c_stage2_20261014_01'
    execution=read_json(run/'execution_manifest.json')
    verify_digest(run/'execution_manifest.json',EXPECTED_EXECUTION)
    assert_hashes(execution['outputs']);assert_hashes(execution['process_files'])
    accept=PROGRESS/'Stage2_Exception_Resolution/HRS_PO01/acceptance_v1'
    checkdir=accept.parent/'check_v1'
    for path in [accept/'acceptance_manifest.json',checkdir/'manifest.json']:
        manifest=read_json(path)
        for key in ['source_hashes','inputs','outputs']:
            if key in manifest:assert_hashes(manifest[key])
    approval=read_json(accept/'author_numeric_exception_approval.json');oldcheck=read_json(checkdir/'exception_check.json')
    check_exception(approval,contract,run,oldcheck)
    protected=read_json(run/'protected_assets_before.json')
    prior_validation=check_protected(protected)
    source_hashes={**contract['inputs'],**contract['code'],**execution['outputs'],**execution['process_files']}
    for folder in [accept,checkdir]:
        for path in folder.iterdir():
            if path.is_file():source_hashes[rel(path)]=sha(path)
    source_hashes[rel(CONTRACT)]=sha(CONTRACT)
    source_hashes[rel(run/'execution_manifest.json')]=sha(run/'execution_manifest.json')
    for rid in ['wealth_measurement_v1','model_robustness_round1b_v1']:
        base=ROOT/'03_output/3.3 Robustness of the Main Findings/releases'/rid
        source_hashes.update({rel(p):sha(p) for p in base.rglob('*') if p.is_file()})
    output=QA_ROOT/qa_id;process=PROGRESS/'Stage3'/qa_id
    output.mkdir(parents=True,exist_ok=False);process.mkdir(parents=True,exist_ok=False)
    write_json(process/'stage3_execution_authorization.json',{'schema':'round1c_stage3_execution_authorization_v1',
       'authorization':'GRANTED_FOR_STAGE3_ONLY','human_instruction':instruction,
       'instruction_source':str(args.instruction),'instruction_sha256':sha(args.instruction),
       'contract_sha256':sha(CONTRACT),'preflight_manifest_sha256':contract['audit']['manifest_sha256'],
       'original_run_id':run.name,'author_scientific_result_approval':'PENDING','stages4_5':'NOT_AUTHORIZED',**provenance()})
    validations=[];matrix=[];comparisons={};guard={'forbidden_calls':0}
    with prohibit_optimization(guard):
        for cohort in ['HRS','ELSA','KLoSA']:
            post,designs,anchor,primary,pa,pdiag,pcheck=rebuild(cohort,contract)
            models=[{'cohort':cohort,'specification':'Primary',**primary,'Participants':pa['participants'],
                     'FI_observations':pa['rows'],'ordered_key_sha256':pa['ordered_person_wave_sha256'],
                     'status':'FROZEN_REFERENCE','Primary_reconstruction':pcheck}]
            for spec in ['PO01','LM01']:
                unitout=output/cohort/spec;unitout.mkdir(parents=True,exist_ok=False)
                folder=run/spec/cohort;obj=read_json(folder/'result_object.json')
                v={'cohort':cohort,'specification':spec,'original_stage2_status':obj['status'],
                   'author_numeric_exception':'AUTHOR_ACCEPTED_NUMERICAL_EXCEPTION' if (cohort,spec)==('HRS','PO01') else None,
                   'scientific_author_approval':'PENDING','release_eligible':False}
                try:
                    validate_anchor(anchor,obj['sample']);validate_anchor(anchor,read_json(folder/'sample_anchor.json'))
                    original,basis=designs[spec];names=list(basis);mapping=read_json(folder/'design_parameterization.json')
                    require(list(original)==mapping['original_columns'],'原设计列名漂移')
                    bcheck=validate_basis(original,basis,names,mapping,cohort,post.time_years)
                    require(names==obj['fixed_effect_names'],'保存固定参数标签漂移')
                    saved=load_parameters(folder/'model_parameters.npz')
                    require(saved['fixed_effect_names'].tolist()==names,'NPZ标签漂移')
                    np.testing.assert_array_equal(saved['fixed_effects'],obj['fixed_effects'])
                    np.testing.assert_array_equal(saved['fixed_effect_covariance'],obj['fixed_effect_covariance'])
                    numeric,arrays,block=validate_candidate(post.fi_primary.to_numpy(float),basis.to_numpy(float),post.person_id.to_numpy(),
                          names,saved,read_json(folder/'diagnostics.json'),read_json(folder/'optimization_log.json'),obj['status'],
                          approval if (cohort,spec)==('HRS','PO01') else None)
                    if spec=='LM01':
                        require(obj['interpretation']['conditional_association_statement']==LM_STATEMENT,'LM01估计对象改写')
                    require(not obj['interpretation']['landmark_level_claims_allowed'],'禁止landmark水平解释')
                    core=numeric['core'];supplement=read_json(folder/'stage2_core_inference.json')
                    prior=supplement['raw_candidate'] if obj['status']=='HOLD_NOT_ESTIMABLE' else supplement['validated_core']
                    for k in ['estimate','se','ci_lower','ci_upper']:
                        require(abs(core[k]-prior[k])<=1e-12,'核心Wald计算与保存值不一致')
                    prior_p=supplement['raw_candidate_p_value'] if obj['status']=='HOLD_NOT_ESTIMABLE' else supplement['p_value']
                    require(abs(core['p_value']-prior_p)<=1e-12,'P值复核失败')
                    residual=support_and_residuals(post,block,numeric['residual_variance'],numeric['random_variance'])
                    influence=influence_screen(post.fi_primary.to_numpy(float),basis.to_numpy(float),post.person_id.to_numpy(),
                         post.time_years.to_numpy(float),post.recovery.to_numpy(int),saved['fixed_effects'],
                         numeric['random_variance'],numeric['residual_variance'],names.index(TERM),core['se'])
                    v.update(numerical=numeric,design=bcheck,sample=anchor,residuals=residual,influence=influence,
                             status='PASS_WITH_CAVEAT_RECOMMENDED',
                             reasons=['条件残差及时间支持限制待作者接受']+
                                     (['原L-BFGS未收敛且准确终止信息缺失；数值例外仅限定验收'] if obj['status']=='HOLD_NOT_ESTIMABLE' else []))
                    np.savez_compressed(unitout/'independent_parameter_evidence.npz',**arrays)
                    write_json(unitout/'support_residual_diagnostics.json',residual)
                    write_json(unitout/'influence_screen.json',influence)
                    write_json(unitout/'design_parameterization_validation.json',bcheck)
                    model={'cohort':cohort,'specification':spec,**core,'Participants':anchor['participants'],
                      'FI_observations':anchor['rows'],'ordered_key_sha256':anchor['ordered_person_wave_sha256'],
                      'status':v['status'],'loglik':numeric['loglik_profile']}
                    models.append(model)
                    matrix.append({'Cohort':cohort,'Specification':spec,'Participants':anchor['participants'],
                      'FI_observations':anchor['rows'],**core,'Stage2_original_status':obj['status'],
                      'Stage3_numerical_status':numeric['numerical_status'],'Stage3_review_recommendation':v['status'],
                      'scaled_gradient_max':numeric['scaled_gradient_max'],'numerical_exception':obj['status']=='HOLD_NOT_ESTIMABLE',
                      'boundary_candidate':numeric['boundary_candidate'],'residual_skewness':residual['conditional_residual_distribution']['skewness'],
                      'residual_excess_kurtosis':residual['conditional_residual_distribution']['excess_kurtosis'],
                      'maximum_influence_over_SE':influence['max_abs_influence_over_formal_SE'],
                      'scientific_approval':'PENDING','release_eligible':False})
                except Exception as error:
                    v.update(status='HOLD',failure=str(error),traceback=traceback.format_exc())
                    matrix.append({'Cohort':cohort,'Specification':spec,'Stage2_original_status':obj['status'],
                                   'Stage3_numerical_status':'HOLD','Stage3_review_recommendation':'HOLD','reason':str(error),
                                   'scientific_approval':'PENDING','release_eligible':False})
                write_json(unitout/'independent_validation.json',v);validations.append(v)
                print(f"{cohort}/{spec}: {v['status']}",flush=True)
            complete=len(models)==3
            comparisons[cohort]={'models':models,'complete_chain':complete,
               'steps':[compare_rows(models[0],models[1]),compare_rows(models[1],models[2],True)] if complete else []}
    require(guard['forbidden_calls']==0,'Stage3尝试调用拟合或优化器')
    overall='QA_HOLD' if any(v['status']=='HOLD' for v in validations) else 'QA_APPROVAL_WITH_CAVEATS_RECOMMENDED'
    write_csv(output/'model_qa_matrix.csv',matrix)
    write_json(output/'independent_validation.json',{'status':overall,'units':validations,'source_checks':sources,
      'original_statuses_unchanged':True,'author_scientific_result_approval':'PENDING','new_model_fits':0,
      'new_optimizer_calls':0,'optimization_guard':guard,'contains_participant_records':False})
    write_json(output/'sequential_comparison.json',comparisons)
    reference=round1b_reference();write_json(output/'round1b_reference.json',reference)
    text=reports(validations,comparisons,reference,overall)
    report=['# Round 1C — Stage3独立统计QA总报告','',f'QA ID：`{qa_id}`；总体建议：`{overall}`。',
      '','Stage3技术与科学审查完成，等待作者决定。未批准科学结果、未执行Stage4/5。','',
      '已核对合同、活动Preflight、63项来源、九项锁定代码、六模型结果及HRS/PO01限定批准链。',
      '','独立路径：按合同重新构建设计；逐人Cholesky白化与QR/SVD点估计；Woodbury解析剖面score/Hessian；再以(β,τ²,σ²)完整观察信息核查核心SE。没有使用条件GLS协方差作最终推断，没有伪逆、裁剪或参数更新。','',
      '|队列/规格|原Stage2|独立数值QA|科学审阅建议|','|---|---|---|---|']
    for v in validations:
        report.append(f"|{v['cohort']}/{v['specification']}|{v['original_stage2_status']}|{v.get('numerical',{}).get('numerical_status','HOLD')}|{v['status']}|")
    report.extend(['','## 数值证据与推断支持',''])
    for v in validations:
        if 'numerical' not in v:
            report.append(f"{v['cohort']}/{v['specification']}：HOLD，{v['failure']}");continue
        n=v['numerical'];r=v['residuals'];i=v['influence']
        report.extend([f"{v['cohort']}/{v['specification']}：Cholesky似然绝对误差{n['Cholesky_loglik_abs_error']:.3g}；标准化梯度{n['scaled_gradient_max']:.3g}；完整物理参数信息核心SE相对误差{n['core_SE_full_physical_relative_error']:.3g}；随机方差{n['random_variance']:.6g}，残差方差{n['residual_variance']:.6g}。",'',
          f"条件残差偏度{r['conditional_residual_distribution']['skewness']:.3f}、超额峰度{r['conditional_residual_distribution']['excess_kurtosis']:.3f}；残差与时间相关{r['residual_time_correlation']:.3f}，与条件拟合值相关{r['residual_fitted_correlation']:.3f}。实际FI分布与残差另存，不能由FI偏态直接推断模型失效。",'',
          f"单人最大固定协方差影响约为正式SE的{i['max_abs_influence_over_formal_SE']:.3f}倍；前10人绝对影响占比{next(z['absolute_influence_share'] for z in i['top_concentration'] if z['top_participants']==10):.1%}。这是一阶筛查，不是leave-one-out或正式SE。",''])
        for g in r['groups']:
            report.append(f"- {g['recovery_status']}：{g['participants']:,}人、{g['observations']:,}行；>8年{g['over_8_year_participants']:,}人/{g['over_8_year_rows']:,}行；条件拟合越界{g['conditional_fitted_outside_0_1_rows']}行。")
        report.append('')
    report.extend(['## 解释限制与停止点','',
      'HRS/PO01原HOLD保持；独立检查支持的候选只推荐限定使用，原SciPy准确终止原因仍不可恢复。其余单元数值通过不等于线性均值、Gaussian分布或随机截距结构已证明正确。',
      '','两年分箱及8年后支持是实际观察支持，不能称为在险/生存人数。残差、极端残差时间分布和长随访影响汇总均保存在各单元机器结果。稀疏远期及一阶筛查的不完备性需保留，不能宣称已排除截短敏感性。',
      '',LM_STATEMENT,'','本轮只报告后续进展，不解释landmark水平差，不重绘Figure3，不新增联合模型或meta。',
      '','请作者结合scientific_interpretation.md、round1b_consistency_review.md与exceptions_and_decisions.md决定是否接受具体结果及限制、是否进入Stage4。没有新Release、freeze或候选投稿表。'])
    text['stage3_qa_report.md']='\n'.join(report)+'\n'
    for name,value in text.items():write_text(process/name,value)
    assert_hashes(source_hashes)
    protected_validation=check_protected(protected)
    write_json(output/'protected_assets_validation.json',protected_validation)
    qa_sources=[CODE/name for name in ['run_stage3_landmark_qa.py','stage3_numerics.py','stage3_diagnostics.py','stage3_review.py','test_stage3_landmark_qa.py',
        'exception_checks.py','resolve_hrs_po01_exception.py']]
    qa_sources.append(CODE.parent/'model_robustness/stage3/independent_checks.py')
    manifest={'schema':'round1c_stage3_manifest_v1','status':'STAGE3_COMPLETE_AWAITING_AUTHOR_DECISION','recommendation':overall,
      'qa_id':qa_id,'source_hashes':source_hashes,'qa_code':{rel(p):sha(p) for p in qa_sources},
      'outputs':{rel(p):sha(p) for p in output.rglob('*') if p.is_file()},
      'reports':{rel(p):sha(p) for p in process.rglob('*') if p.is_file()},
      'protected_assets_validation':protected_validation,'new_model_fits':0,'new_optimizer_calls':0,
      'stages4_5_executed':False,'author_scientific_result_approval':'PENDING',**provenance()}
    write_json(process/'stage3_manifest.json',manifest)
    assert_hashes(manifest['outputs']);assert_hashes(manifest['reports']);assert_hashes(manifest['qa_code'])
    print(overall,rel(process),flush=True)


if __name__=='__main__':main()
