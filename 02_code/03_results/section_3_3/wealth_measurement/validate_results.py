"""Phase 2初步文件和数值完整性核验；不代替Phase 3科学QA。"""
import numpy as np
import pandas as pd
from prepare_inputs import ROOT,OUTPUT,CONTRACT_PATH,EXPECTED_CONTRACT_SHA,sha,require,read_json
from run_models import TERM,Z

def validate(contract,objects,diagnostics):
 require(sha(CONTRACT_PATH)==EXPECTED_CONTRACT_SHA,'合同已变化')
 require(len(objects)==15 and len(diagnostics)==15,'规格数不是15')
 require(len({(o['analysis_id'],o['cohort']) for o in objects})==15,'拟合规格重复')
 require(sum(o['status']=='SKIP_IDENTICAL_TO_PRIMARY' for o in objects)==1,'跳过数不正确')
 descriptors=pd.read_csv(OUTPUT/'Descriptives/WM00_descriptives.csv')
 require(len(descriptors)==36,'WM00行数不符')
 primary=read_json(ROOT/'03_output/3.2 Wealth Recovery and Subsequent Frailty Progression/Primary Models/primary_lmm_result_object.json')['primary_estimand_rows']
 for o,d in zip(objects,diagnostics):
  anchor=next(a for a in contract['validated_sample_and_design_anchors'] if a['cohort']==o['cohort'] and a['analysis_id']==o['analysis_id'])
  for k in ['participants','fi_observations','recovery_n','comparison_n']:require(o['sample'][k]==anchor[k],'样本锚点不符')
  folder=OUTPUT/'Model Results'/o['analysis_id']/o['cohort']
  for n in ['result_object.json','diagnostics.json','sample_anchor.json','design_matrix.json','optimization_log.json']:require((folder/n).is_file(),'单元文件缺失：'+n)
  if o['status']=='SUCCESS':
   require(d['design']['rank']==d['design']['n_columns']==anchor['expected_design_columns'],'矩阵异常')
   require(d['bfgs_converged'] and d['lbfgs_converged'],'优化未收敛')
   require(d['beta3_optimizer_absolute_difference']<=1e-6 and d['loglik_optimizer_absolute_difference']<=1e-4,'优化器差异')
   for f in o['fixed_effects']:
    require(np.isfinite([f['estimate'],f['se'],f['ci_lower'],f['ci_upper']]).all() and f['se']>0,'系数异常')
    require(np.isclose(f['ci_lower'],f['estimate']-Z*f['se'],rtol=1e-12,atol=1e-14) and np.isclose(f['ci_upper'],f['estimate']+Z*f['se'],rtol=1e-12,atol=1e-14),'Wald区间错误')
   cov=np.asarray(o['fixed_effect_covariance']);index=o['fixed_effect_names'].index(TERM)
   require(np.isclose(cov[index,index],o['estimand']['se']**2,rtol=1e-12,atol=1e-16),'SE与协方差不符')
   require(len(o['fixed_effects'])==anchor['expected_design_columns'],'系数行数不符')
   with np.load(folder/'model_parameters.npz',allow_pickle=False) as p:require(set(p.files)=={'parameter_names','parameters','fixed_effect_covariance','hessian','residual_variance','random_intercept_variance'},'参数资产含未知字段')
  elif o['status']=='SKIP_IDENTICAL_TO_PRIMARY':
   require(o['cohort']=='KLoSA' and o['analysis_id']=='WM04','跳过错误单元')
   original=next(p for p in primary if p['cohort']=='KLoSA')
   for k in ['estimate','se','ci_lower','ci_upper','p_value']:require(o['estimand'][k]==original[k],'跳过值与主模型不符')
  else:require(o['failure_reasons'],'失败无原因')
 estimates=pd.read_csv(OUTPUT/'cohort_estimates.csv')
 require(len(estimates)==18,'汇总估计行数不符')
 for p in primary:
  r=estimates.loc[estimates.analysis_id.eq('PRIMARY') & estimates.cohort.eq(p['cohort'])].iloc[0]
  for k in ['estimate','se','ci_lower','ci_upper','p_value']:require(np.isclose(r[k],p[k],rtol=1e-12,atol=1e-16),'主结果引用变化')
 return {'status':'PASS','scope':'Phase 2 preliminary output integrity only; Phase 3 QA pending','specifications':15,'real_fit_units':14,'wm00_rows':36,'estimate_rows':18,'successful_fits':sum(o['status']=='SUCCESS' for o in objects),'failed_or_not_estimable':sum(o['status'] in ['FAILED','NOT_ESTIMABLE'] for o in objects)}

def validate_manifest(manifest):
 for a in manifest['outputs']:require(sha(ROOT/a['path'])==a['sha256'],'输出哈希不符：'+a['path'])
 for a in manifest['code']:require(sha(ROOT/a['path'])==a['sha256'],'代码哈希不符')
 return {'status':'PASS','output_files_checked':len(manifest['outputs']),'code_files_checked':len(manifest['code'])}
