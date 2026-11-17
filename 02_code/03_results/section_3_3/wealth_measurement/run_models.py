"""独立拟合单元；不调用正式meta或旧42列拟合入口。"""
import time
import traceback
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from prepare_inputs import write_json,clean,require

TERM='recovery:time_years'
Z=1.959963984540054

def acceptance(bfgs,lbfgs,beta_diff,ll_diff,hessian_singular,hessian,cov,scale,tau,policy):
 reasons=[]
 for label,result in [('bfgs',bfgs),('lbfgs',lbfgs)]:
  if not result.converged:reasons.append(label+'未收敛')
  if not np.isfinite(np.asarray(result.fe_params)).all() or not np.isfinite(np.asarray(result.cov_params())).all():reasons.append(label+'估计或协方差非有限')
 if not np.isfinite(beta_diff) or beta_diff>policy['beta3_absolute_tolerance']:reasons.append('恢复×时间优化器差异超门槛')
 if not np.isfinite(ll_diff) or ll_diff>policy['loglik_absolute_tolerance']:reasons.append('loglik优化器差异超门槛')
 if hessian_singular or not np.isfinite(hessian).all():reasons.append('Hessian不可用')
 if not np.isfinite(cov).all() or (np.diag(cov)<=0).any():reasons.append('固定效应协方差或SE异常')
 if not np.isfinite(scale) or scale<=0 or not np.isfinite(tau) or tau<0:reasons.append('方差分量异常')
 return reasons

def optimize(model,method,policy,start=None):
 began=time.perf_counter();result=None;error=None
 with warnings.catch_warnings(record=True) as records:
  warnings.simplefilter('always')
  try:
   kwargs={'reml':False,'method':method,'maxiter':policy['maxiter'],'disp':False,'full_output':True}
   if start is not None:kwargs['start_params']=start
   result=model.fit(**kwargs)
  except Exception as exc:error=type(exc).__name__+': '+str(exc)
 log={'optimizer':method,'elapsed_seconds':time.perf_counter()-began,'warnings':[{'category':r.category.__name__,'message':str(r.message),'source_file':r.filename.replace('\\','/').split('/')[-1],'line':r.lineno} for r in records],'error':error}
 if result is not None:
  log.update({'converged':bool(result.converged),'log_likelihood':float(result.llf),'parameters':np.asarray(result.params).tolist(),'parameter_names':list(result.params.index),'optimization_history':getattr(result,'hist',None),'recovery_time':float(result.fe_params[TERM])})
 return result,clean(log)

def diagnostic_proxies(frame,x,result):
 # 固定设计矩阵杠杆只作为筛查代理，不冒充LMM删除影响分析。
 u=np.linalg.svd(np.asarray(x),full_matrices=False)[0];leverage=(u*u).sum(axis=1)
 residual=np.asarray(result.resid);fitted=np.asarray(result.fittedvalues)
 temporary=pd.DataFrame({'person_id':frame.person_id.to_numpy(),'loss':frame.wealth_loss_fraction.to_numpy(float),'leverage':leverage,'residual2':residual**2})
 persons=temporary.groupby('person_id',observed=True).agg(loss=('loss','first'),leverage=('leverage','sum'))
 groups=[]
 for threshold in [10,100,1000]:
  mask=persons.loss.gt(threshold);rowmask=frame.wealth_loss_fraction.gt(threshold).to_numpy()
  groups.append({'loss_threshold':threshold,'participants':int(mask.sum()),'participant_percent':float(mask.mean()*100),'design_leverage_share':float(persons.loc[mask,'leverage'].sum()/persons.leverage.sum()),'conditional_residual_rmse':float(np.sqrt(np.mean(residual[rowmask]**2))) if mask.sum()>=5 else None,'small_group_residual_summary_suppressed':bool(mask.sum()<5)})
 return {'evidence':'fixed_design_leverage_proxy_and_conditional_residual_descriptives; NOT deletion/refit influence','formal_individual_influence_analysis':'NOT_PERFORMED_REQUIRES_SEPARATE_AUTHORIZATION','extreme_loss_groups':groups,'residual_mean':float(residual.mean()),'residual_sd':float(residual.std(ddof=1)),'residual_rmse':float(np.sqrt(np.mean(residual**2))),'residual_time_correlation':float(np.corrcoef(residual,frame.time_years)[0,1]),'residual_fitted_correlation':float(np.corrcoef(residual,fitted)[0,1])}

def fit_unit(contract,cohort,analysis_id,frame,x,design,counts,out,primary_se):
 out.mkdir(parents=True,exist_ok=False)
 write_json(out/'design_matrix.json',design);write_json(out/'sample_anchor.json',counts)
 diagnostics={'cohort':cohort,'analysis_id':analysis_id,'design':design,'sample':counts,'status':'FAILED','failure_reasons':[]}
 obj={'cohort':cohort,'analysis_id':analysis_id,'sample':counts,'status':'FAILED','source_kind':'NEW_LMM_FIT','inference':'unweighted ML, random intercept, model-based normal Wald','estimand':None}
 logs=[];lbfgs=None;coefficients=[]
 try:
  model=sm.MixedLM(endog=frame.fi_primary.astype(float),exog=x,groups=frame.person_id.to_numpy(),exog_re=np.ones((len(frame),1)))
  bfgs,log=optimize(model,'bfgs',contract['optimizer']);logs.append(log);write_json(out/'optimization_log.json',logs)
  require(bfgs is not None,'BFGS未返回结果：'+str(log['error']))
  lbfgs,log=optimize(model,'lbfgs',contract['optimizer'],bfgs.params);logs.append(log);write_json(out/'optimization_log.json',logs)
  require(lbfgs is not None,'L-BFGS未返回结果：'+str(log['error']))
  names=list(lbfgs.fe_params.index);cov=lbfgs.cov_params().loc[names,names].to_numpy(float)
  with warnings.catch_warnings(record=True) as hs_warnings:
   warnings.simplefilter('always');hessian,singular=model.hessian(lbfgs.params_object)
  tau=float(np.asarray(lbfgs.cov_re)[0,0]);scale=float(lbfgs.scale)
  beta_diff=abs(float(lbfgs.fe_params[TERM]-bfgs.fe_params[TERM]));ll_diff=abs(float(lbfgs.llf-bfgs.llf))
  failures=acceptance(bfgs,lbfgs,beta_diff,ll_diff,singular,hessian,cov,scale,tau,contract['optimizer'])
  diagnostics.update({'bfgs_converged':bool(bfgs.converged),'lbfgs_converged':bool(lbfgs.converged),'beta3_optimizer_absolute_difference':beta_diff,'loglik_optimizer_absolute_difference':ll_diff,'log_likelihood':float(lbfgs.llf),'random_intercept_variance':tau,'residual_variance':scale,'random_intercept_to_residual_variance_ratio':tau/scale,'hessian_singular':bool(singular),'hessian_eigenvalues':np.linalg.eigvalsh(hessian).tolist(),'hessian_condition_number':float(np.linalg.cond(hessian)),'hessian_warnings':[str(w.message) for w in hs_warnings],'fixed_covariance_eigenvalues':np.linalg.eigvalsh(cov).tolist(),'fixed_covariance_condition_number':float(np.linalg.cond(cov)),'fixed_covariance_max_asymmetry':float(np.abs(cov-cov.T).max()),'failure_reasons':failures})
  for i,name in enumerate(names):
   variance=cov[i,i];se=float(np.sqrt(variance)) if variance>0 else float('nan');estimate=float(lbfgs.fe_params[name])
   coefficients.append({'cohort':cohort,'analysis_id':analysis_id,'term':name,'estimate':estimate,'se':se,'ci_lower':estimate-Z*se,'ci_upper':estimate+Z*se,'p_value':float(2*norm.sf(abs(estimate/se))) if se>0 else None,'is_primary_estimand':name==TERM})
  pd.DataFrame(coefficients).to_csv(out/'fixed_effects.csv',index=False,encoding='utf-8')
  pd.DataFrame([{'term_i':i,'term_j':j,'covariance':cov[a,b]} for a,i in enumerate(names) for b,j in enumerate(names)]).to_csv(out/'fixed_effect_covariance.csv',index=False,encoding='utf-8')
  hnames=list(lbfgs.params.index)
  pd.DataFrame(hessian,index=hnames,columns=hnames).to_csv(out/'hessian.csv',encoding='utf-8')
  # 仅保存可复用参数；不序列化包含个人数据的完整拟合对象。
  np.savez_compressed(out/'model_parameters.npz',parameter_names=np.asarray(hnames,dtype=str),parameters=np.asarray(lbfgs.params),fixed_effect_covariance=cov,hessian=hessian,residual_variance=np.asarray(scale),random_intercept_variance=np.asarray(tau))
  estimand=next(z for z in coefficients if z['term']==TERM)
  diagnostics['SE_ratio_to_frozen_primary']=estimand['se']/primary_se
  try:diagnostics['extreme_loss_and_residual_diagnostics']=diagnostic_proxies(frame,x,lbfgs)
  except Exception as exc:diagnostics['diagnostic_proxy_error']=type(exc).__name__+': '+str(exc)
  obj.update({'fixed_effects':coefficients,'fixed_effect_covariance':cov.tolist(),'fixed_effect_names':names,'estimand':estimand,'parameter_serialization':'parameter-only NPZ; not a MixedLMResults object and contains no participant records'})
  if not failures:obj['status']=diagnostics['status']='SUCCESS'
 except Exception as exc:
  diagnostics['failure_reasons'].append(type(exc).__name__+': '+str(exc))
  # 堆栈不包含个人级数据，保留失败定位证据。
  (out/'failure_trace.txt').write_text(traceback.format_exc(),encoding='utf-8')
 obj['failure_reasons']=diagnostics['failure_reasons'];diagnostics['optimization']=logs
 write_json(out/'optimization_log.json',logs);write_json(out/'diagnostics.json',diagnostics);write_json(out/'result_object.json',obj)
 return obj,diagnostics
