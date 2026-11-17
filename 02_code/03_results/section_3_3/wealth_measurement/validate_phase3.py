"""Phase 3独立QA：固定保存参数作等价复核，禁止拟合和优化。"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import ast
import hashlib
import inspect
import io
import json
import platform
import subprocess
import sys
import traceback
import unittest
import uuid
import time
import numpy as np
import pandas as pd
import patsy
import scipy
from scipy.linalg import lstsq,cho_factor,cho_solve
from scipy.stats import norm
import statsmodels
from statsmodels.regression.mixed_linear_model import MixedLM

CODE=Path(__file__).resolve().parent
ROOT=CODE.parents[3]
R1=ROOT/'06_project_progress/round1_extra_analyses_v1'
P2=R1/'Phase2';QA=R1/'Phase3'
OUT=ROOT/'03_output/3.3 Robustness of the Main Findings/Wealth Measurement'
CONTRACT=R1/'Contracts/wealth_measurement_round1_v1_contract.json'
CONTRACT_SHA='a9d6769d479c80d6d0d886440e5f66ff6033eae5810959bb3c5eea18fdf86f4c'
PRIMARY=ROOT/'03_output/3.2 Wealth Recovery and Subsequent Frailty Progression/Primary Models/primary_lmm_result_object.json'
TERM='recovery:time_years';Z=1.959963984540054
HIGH={('WM05','HRS'),('WM01','ELSA'),('WM05','ELSA')}
COHORTS=['HRS','ELSA','KLoSA']
TOL={'file_rtol':1e-10,'file_atol':1e-12,'amount_rtol':1e-12,'amount_atol':1e-8,'core_rtol':1e-6,'core_atol':1e-10,'loglik_atol':1e-6}

def require(value,message):
 if not bool(value):raise ValueError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def rel(path):return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sanitize(value):
 if isinstance(value,dict):return {str(k):sanitize(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [sanitize(v) for v in value]
 if isinstance(value,np.ndarray):return sanitize(value.tolist())
 if isinstance(value,np.generic):return sanitize(value.item())
 if isinstance(value,float) and not np.isfinite(value):return None
 return value
def save(path,value):
 temporary=Path(path).with_name(Path(path).name+'.'+uuid.uuid4().hex+'.tmp')
 temporary.write_text(json.dumps(sanitize(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 for i in range(10):
  try:temporary.replace(path);return
  except OSError:
   if i==9:raise
   time.sleep(.1*(i+1))
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,encoding='utf-8').strip()
def close(a,b,rtol=1e-10,atol=1e-12):return bool(np.allclose(a,b,rtol=rtol,atol=atol,equal_nan=False))
def core_close(a,b):return close(a,b,TOL['core_rtol'],TOL['core_atol'])
def verify_digest(actual,expected):require(actual==expected,'SHA-256不符')
def csv(path):return pd.read_csv(path,float_precision='round_trip',encoding='utf-8')

def snapshot():
 assets={}
 for name in ['00_admin','01_data','02_code','03_output','04_collaboration','06_project_progress','07_manuscript']:
  for p in (ROOT/name).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and not p.is_relative_to(QA) and p.resolve()!=Path(__file__).resolve():assets[rel(p)]=sha(p)
 for name in ['AGENTS.md','pyproject.toml','uv.lock','renv.lock','.gitignore','.gitattributes','.Rprofile']:assets[name]=sha(ROOT/name)
 for cohort,entry in read(ROOT/'06_project_progress/foundation_audit_v1/audit_manifest.json')['cohorts'].items():assets['external_raw/'+cohort]=sha(entry['source']['path'])
 return assets

def verify_assets():
 contract=read(CONTRACT);verify_digest(sha(CONTRACT),CONTRACT_SHA);require(contract['status']=='LOCKED','合同非LOCKED')
 for name,key in [('manifest.json','manifest_sha256'),('preflight_results.json','results_sha256')]:verify_digest(sha(R1/'Preflight'/name),contract['preflight'][key])
 for item in read(R1/'delivery_validation.json')['outputs']:verify_digest(sha(ROOT/item['path']),item['sha256'])
 manifest=read(OUT/'manifest.json')
 for item in manifest['outputs']+manifest['code']:verify_digest(sha(ROOT/item['path']),item['sha256'])
 verify_digest(sha(OUT/'manifest.json'),read(P2/'manifest_validation.json')['manifest_sha256'])
 for item in contract['source_freezes']:
  verify_digest(sha(ROOT/item['path']),item['sha256']);verify_digest(sha(ROOT/item['freeze_path']),item['freeze_sha256'])
  require(read(ROOT/item['freeze_path'])['freeze_status']=='FROZEN','冻结记录状态不符')
 historical=read(ROOT/'06_project_progress/foundation_audit_addendum_v1/foundation_audit_addendum_v1_freeze.json')
 for path,h in historical['historical_model_assets_sha256'].items():verify_digest(sha(ROOT/path),h)
 lineage=contract['historical_and_current_source_lineage']
 verify_digest(sha(ROOT/lineage['historical_source_backup']),lineage['historical_source_sha256'])
 verify_digest(sha(ROOT/lineage['current_repaired_source_path']),lineage['current_repaired_source_sha256'])
 history=read(P2/'execution_interruption_attempt1.json')
 for item in history['completed_assets']:verify_digest(sha(ROOT/item['path']),item['sha256'])
 for item in history['code']:verify_digest(sha(ROOT/item['backup_path']),item['sha256'])
 old=P2/'source_versions/attempt1'
 for name in ['run_models.py','build_descriptives.py']:verify_digest(sha(old/name),sha(CODE/name))
 def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
 a=functions(old/'prepare_inputs.py');b=functions(CODE/'prepare_inputs.py')
 for name in ['verify_benchmark','load_cohort','prepare_spec','design_matrix']:require(a[name]==b[name],'科学函数AST变化：'+name)
 units=read(P2/'execution_units.json')['units'];require(len(units)==14 and sum(u['reused_without_refit'] for u in units)==9,'中断续跑数量不符')
 require(len(history['completed_assets'])==81,'中断前81份文件记录不完整')
 require(manifest['runtime']==runtime(),'当前依赖版本与实际运行环境不一致')
 return contract,manifest,{'status':'PASS','phase2_output_files':len(manifest['outputs']),'phase2_code_files':len(manifest['code']),'frozen_parquets':9,'preserved_attempt1_model_files':81,'attempt1_models':9,'attempt2_models':5,'scientific_functions_identical':True,'historical_primary_source_distinct_from_KH_repaired_source':True}

def runtime():return {'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'patsy':patsy.__version__,'statsmodels':statsmodels.__version__}

def independently_load(contract,cohort):
 def load(component):
  asset=next(x for x in contract['source_freezes'] if x['cohort']==cohort and x['component']==component)
  cols=list(asset['columns_read'])
  if component=='analysis_panel' and 'fi_primary' not in cols:cols.append('fi_primary')
  return pd.read_parquet(ROOT/asset['path'],columns=cols)
 panel=load('analysis_panel');exposure=load('exposure_panel');wealth=load('wealth_panel')
 require(not panel.duplicated(['person_id','wave']).any(),'Analysis键重复')
 panel=panel.loc[panel.primary_complete_case_eligible.eq(True)].copy()
 landmark=panel.loc[panel.fi_time_role.eq('landmark')].copy();require(not landmark.person_id.duplicated().any(),'landmark重复')
 require(landmark.time_years.eq(0).all() and landmark.wave.eq(landmark.landmark_wave).all(),'landmark时间或波次不符')
 require(set(panel.person_id)==set(landmark.person_id),'有参与者缺少唯一landmark')
 frozen_fields=list(contract['common_model']['continuous_covariates'])+list(contract['common_model']['categorical_covariates'])+['recovery_status','tminus1_wave','index_shock_wave','landmark_wave']
 require(panel.groupby('person_id',observed=True)[frozen_fields].nunique(dropna=False).eq(1).all().all(),'个人固定字段行间变化')
 ref=exposure.loc[exposure.is_index_shock.eq(True)].copy();require(not ref.person_id.duplicated().any(),'首次索引冲击键重复')
 ref['absolute_loss']=ref.wealth_tminus1.astype(float)-ref.wealth_t.astype(float)
 threshold=contract['analyses'][2]['thresholds'][cohort]
 require(len(ref)==threshold['reference_n'],'P25参考人数不符')
 p25=float(np.quantile(ref.absolute_loss.to_numpy(float),.25,method='linear'))
 require(close(p25,threshold['value_full_precision']),'独立P25不符')
 require(int(ref.absolute_loss.eq(threshold['value_full_precision']).sum())==threshold['reference_equal_n'],'参考并列人数不符')
 person=landmark[['person_id','recovery_status','wealth_quintile','tminus1_wave','index_shock_wave','landmark_wave']].merge(ref[['person_id','candidate_shock_wave','tplus1_wave','wealth_tminus1','wealth_t','wealth_tplus1','absolute_loss','wealth_loss_fraction','recovery_fraction_raw','recovery_status']].rename(columns={'recovery_status':'exposure_recovery_status'}),on='person_id',how='left',validate='one_to_one')
 require(person.recovery_status.eq(person.exposure_recovery_status).all(),'恢复状态连接不符')
 require(person.index_shock_wave.eq(person.candidate_shock_wave).all() and person.landmark_wave.eq(person.tplus1_wave).all(),'固定索引时点连接不符')
 require(not wealth.duplicated(['person_id','wave']).any(),'Wealth键重复')
 for role,wave,amount in [('pre','tminus1_wave','wealth_tminus1'),('shock','index_shock_wave','wealth_t'),('landmark','landmark_wave','wealth_tplus1')]:
  rhs=wealth.rename(columns={'wave':wave,'wealth_equiv_divisor':'div_'+role,'wealth_total_real_equiv':'joined_amount'})
  person=person.merge(rhs,on=['person_id',wave],how='left',validate='many_to_one')
  require(close(person.joined_amount,person[amount],TOL['amount_rtol'],TOL['amount_atol']),'冻结财富金额连接不符')
  person=person.drop(columns='joined_amount')
 require(person[['div_pre','div_shock','div_landmark']].notna().all().all(),'等值化除数缺失')
 linked=landmark[['person_id','wealth_loss_fraction']].merge(person[['person_id','wealth_loss_fraction']],on='person_id',validate='one_to_one',suffixes=('_analysis','_exposure'))
 require(np.array_equal(linked.wealth_loss_fraction_analysis.to_numpy(),linked.wealth_loss_fraction_exposure.to_numpy()),'分析/暴露损失值不符')
 if cohort=='KLoSA':require(person[['div_pre','div_shock','div_landmark']].eq(1).all().all(),'KLoSA除数不是1')
 for variable in contract['common_model']['categorical_covariates']:panel[variable]=panel[variable].astype(str)
 panel['recovery']=panel.recovery_status.eq('substantial_recovery').astype(int)
 for z in contract['common_model']['frozen_standardization_constants'][cohort]:
  v=landmark[z['variable']].to_numpy(float)
  require(close(np.mean(v),z['mean']) and close(np.std(v,ddof=1),z['sample_sd']),'冻结标准化常数不符')
  panel[z['variable']+'_z']=(panel[z['variable']].astype(float)-z['mean'])/z['sample_sd']
 log_values=np.log1p(person.wealth_loss_fraction.to_numpy(float));logz=contract['analyses'][3]['new_constants'][cohort]
 require(close(log_values.mean(),logz['mean']) and close(log_values.std(ddof=1),logz['sample_sd']),'WM03每人一次常数不符')
 return panel,person,{'reference_n':len(ref),'independent_P25':p25,'locked_P25':threshold['value_full_precision'],'reference_equal_n':int(ref.absolute_loss.eq(threshold['value_full_precision']).sum()),'primary_equal_n':int(person.absolute_loss.eq(threshold['value_full_precision']).sum()),'primary_equal_recovery_n':int((person.absolute_loss.eq(threshold['value_full_precision']) & person.recovery_status.eq('substantial_recovery')).sum()),'primary_equal_comparison_n':int((person.absolute_loss.eq(threshold['value_full_precision']) & person.recovery_status.eq('persistent_or_limited_recovery')).sum()),'WM03_mean':float(log_values.mean()),'WM03_sample_SD':float(log_values.std(ddof=1))}

def restriction(person,aid,threshold):
 if aid=='WM01':return person.wealth_quintile.ne(1)
 if aid=='WM02':return person.absolute_loss.ge(threshold)
 if aid=='WM03':return pd.Series(True,index=person.index)
 if aid=='WM04':return person.div_pre.eq(person.div_shock)&person.div_shock.eq(person.div_landmark)
 if aid=='WM05':return person.wealth_t.gt(0)
 raise ValueError('未知规格')

def omissions_allowed(columns,original_columns,aid,cohort):
 removed=[x for x in original_columns if x not in columns]
 expected=["C(wealth_quintile, Treatment(reference='5'))[T.1]","C(wealth_quintile, Treatment(reference='5'))[T.1]:time_years"] if aid=='WM01' or (cohort=='KLoSA' and aid in ['WM02','WM05']) else []
 return removed==expected and columns==[x for x in original_columns if x not in expected]

def independently_prepare(contract,cohort,aid,panel,person,preflight):
 threshold=contract['analyses'][2]['thresholds'][cohort]['value_full_precision']
 mask=restriction(person,aid,threshold);selected=person.loc[mask].copy();frame=panel.loc[panel.person_id.isin(set(selected.person_id))].copy()
 if aid=='WM03':
  z=contract['analyses'][3]['new_constants'][cohort]
  frame['wealth_loss_fraction_z']=(np.log1p(frame.wealth_loss_fraction.astype(float))-z['mean'])/z['sample_sd']
 anchor=next(a for a in contract['validated_sample_and_design_anchors'] if a['analysis_id']==aid and a['cohort']==cohort)
 sample={'participants':len(selected),'fi_observations':len(frame),'recovery_n':int(selected.recovery_status.eq('substantial_recovery').sum()),'comparison_n':int(selected.recovery_status.eq('persistent_or_limited_recovery').sum())}
 require(all(sample[k]==anchor[k] for k in sample),'参与者或FI行数不符')
 require(len(frame.loc[frame.fi_time_role.eq('landmark')])==len(selected),'限制后每人landmark不唯一')
 refs=contract['common_model']['categorical_reference_levels'];require(all(str(r) in set(frame[v]) for v,r in refs.items()),'参考类别缺失')
 rhs=contract['common_model']['original_formula'].split('~',1)[1]
 x=patsy.dmatrix(rhs,frame,return_type='dataframe',NA_action='raise')
 original=preflight['primary_design']['columns'];require(omissions_allowed(list(x.columns),original,aid,cohort),'非法类别省略或列次序')
 fullframe=frame.copy()
 for variable in contract['common_model']['categorical_covariates']:fullframe[variable]=pd.Categorical(frame[variable],categories=sorted(panel[variable].unique()))
 full=patsy.dmatrix(rhs,fullframe,return_type='dataframe',NA_action='raise')
 zero=[name for name in full if full[name].eq(0).all()]
 require(zero==anchor['omitted_columns'],'省略列没有结构性全零证据')
 u,s,vh=np.linalg.svd(x.to_numpy(float),full_matrices=False);tol=float(s[0]*max(x.shape)*np.finfo(float).eps);rank=int((s>tol).sum())
 require(rank==x.shape[1]==anchor['expected_design_columns'],'矩阵降秩或列数不符')
 expected=next(a for a in preflight['models'] if a['analysis_id']==aid)['design']['actual_basis']
 require(list(x.columns)==expected['columns'],'矩阵列名与Preflight不符')
 matrix={'n_columns':x.shape[1],'rank':rank,'rank_tolerance':tol,'condition_number':float(s[0]/s[-1]),'min_singular_value':float(s[-1]),'omitted_columns':zero,'columns':list(x.columns)}
 modes=[]
 if (aid,cohort) in HIGH:
  for k in [-1,-2,-3]:
   vector=vh[k];top=np.argsort(abs(vector))[-5:][::-1]
   modes.append({'singular_value':float(s[k]),'core_loading':float(vector[list(x.columns).index(TERM)]),'largest_loadings':[{'term':list(x.columns)[i],'loading':float(vector[i])} for i in top]})
 matrix['weak_directions']=modes
 matrix['restricted_participant_standardized_loss_SD']=float(frame.loc[frame.fi_time_role.eq('landmark')].wealth_loss_fraction_z.std(ddof=1))
 return frame,selected,x,sample,matrix

def grouped(y,x,ids):
 _,index=np.unique(ids,return_inverse=True);sizes=np.bincount(index)
 sx=np.zeros((len(sizes),x.shape[1]));sy=np.zeros(len(sizes));np.add.at(sx,index,x);np.add.at(sy,index,y)
 return index,sizes,sx,sy

def equivalent_linear_checks(y,x,ids,beta,tau,sigma,hessian,cov):
 index,sizes,sx,sy=grouped(y,x,ids)
 delta=1/np.sqrt(sigma+sizes*tau)-1/np.sqrt(sigma)
 wx=x/np.sqrt(sigma)+(sx/sizes[:,None])[index]*delta[index,None]
 wy=y/np.sqrt(sigma)+(sy/sizes)[index]*delta[index]
 column_norm=np.linalg.norm(wx,axis=0)
 qr=lstsq(wx/column_norm,wy,lapack_driver='gelsy')[0]/column_norm
 svd=lstsq(wx/column_norm,wy,lapack_driver='gelsd')[0]/column_norm
 residual=wy-wx@beta
 logdet=float(np.sum((sizes-1)*np.log(sigma)+np.log(sigma+sizes*tau)))
 ll=-.5*(len(y)*np.log(2*np.pi)+logdet+float(residual@residual))
 info=-np.asarray(hessian,float);require(np.isfinite(info).all() and (np.diag(info)>0).all(),'负Hessian对角不可用')
 balance=1/np.sqrt(np.diag(info));balanced=balance[:,None]*info*balance[None,:]
 factor=cho_factor(balanced,lower=True,check_finite=True)
 solved=balance[:,None]*cho_solve(factor,np.eye(len(balance)))*balance[None,:]
 backward=float(np.linalg.norm(info@solved-np.eye(len(info)),ord=np.inf)/(np.linalg.norm(info,ord=np.inf)*np.linalg.norm(solved,ord=np.inf)+1))
 return qr,svd,ll,solved,{'observed_information_condition':float(np.linalg.cond(info)),'balanced_information_condition':float(np.linalg.cond(balanced)),'balanced_inverse_backward_error':backward,'hessian_max_asymmetry':float(np.max(np.abs(info-info.T))),'information_min_eigenvalue':float(np.linalg.eigvalsh(info).min()),'fixed_covariance_condition':float(np.linalg.cond(cov)),'fixed_covariance_min_eigenvalue':float(np.linalg.eigvalsh(cov).min()),'fixed_covariance_stable_inverse_max_absolute_difference':float(np.max(np.abs(solved[:len(beta),:len(beta)]-cov))),'qr_svd_max_beta_difference':float(np.max(np.abs(qr-svd))),'all_fixed_beta_max_absolute_difference':float(np.max(np.abs(qr-beta))),'fitted_value_max_absolute_difference':float(np.max(np.abs(x@(qr-beta))))}

def optimizer_scale_and_ll(y,x,ids,parameters):
 beta=parameters[:-1];rho=float(parameters[-1]);require(rho>0,'优化器随机方差比异常')
 residual=y-x@beta;_,sizes,_,sums=grouped(residual,np.ones((len(y),1)),ids)
 sigma=float((residual@residual-np.sum(rho/(1+sizes*rho)*sums**2))/len(y));tau=rho*sigma
 require(sigma>0 and tau>0,'优化器方差分量异常')
 ll=-.5*(len(y)*(np.log(2*np.pi)+np.log(sigma))+float(np.sum(np.log1p(sizes*rho)))+len(y))
 return sigma,tau,ll

def grade(ok,caveats):return 'HOLD' if not ok else ('PASS_WITH_CAVEAT' if caveats else 'PASS')

class SyntheticTests(unittest.TestCase):
 def test_hash_tampering(self):
  with self.assertRaises(ValueError):verify_digest(hashlib.sha256(b'changed').hexdigest(),hashlib.sha256(b'original').hexdigest())
 def test_threshold_boundary(self):
  threshold=1173.0949517238053;v=np.array([np.nextafter(threshold,-np.inf),threshold,np.nextafter(threshold,np.inf)])
  self.assertEqual((v>=threshold).tolist(),[False,True,True]);self.assertNotEqual((v>threshold).tolist(),[False,True,True])
 def test_duplicate_join(self):
  a=pd.DataFrame({'person_id':[1]});b=pd.DataFrame({'person_id':[1,1]})
  with self.assertRaises(pd.errors.MergeError):a.merge(b,on='person_id',validate='one_to_one')
 def test_wrong_log_constant(self):
  values=np.log1p(np.array([.75,1.,10.]));mean=values.mean();sd=values.std(ddof=1)
  self.assertFalse(close((values-mean)/sd,(values-mean)/(sd*2)))
 def test_illegal_omission(self):
  original=['Intercept',"C(wealth_quintile, Treatment(reference='5'))[T.1]",TERM,"C(wealth_quintile, Treatment(reference='5'))[T.1]:time_years",'age_z']
  good=[x for x in original if '[T.1]' not in x]
  self.assertTrue(omissions_allowed(good,original,'WM01','ELSA'));self.assertFalse(omissions_allowed(good[:-1],original,'WM01','ELSA'))
 def test_CI_P_inconsistency(self):
  beta=.001;se=.0003;self.assertFalse(close(beta+Z*se,beta+2.58*se));self.assertFalse(close(2*norm.sf(abs(beta/se)),.5))
 def test_high_condition_core_stability(self):
  rng=np.random.default_rng(50);n=200;t=np.tile([0.,1.],100);g=np.repeat(np.arange(100)%2,2);x=np.column_stack([np.ones(n),t,g*t,rng.normal(size=n)*1e-5]);beta=np.array([.2,.002,.001,.03]);ids=np.repeat(np.arange(100),2)
  y=x@beta;tau=.005;sigma=.01;index,sizes,sx,sy=grouped(y,x,ids);delta=1/np.sqrt(sigma+sizes*tau)-1/np.sqrt(sigma);wx=x/np.sqrt(sigma)+(sx/sizes[:,None])[index]*delta[index,None]
  information=wx.T@wx;h=-information;cov=np.linalg.solve(information,np.eye(4))
  qr,svd,ll,stable,diag=equivalent_linear_checks(y,x,ids,beta,tau,sigma,h,cov)
  self.assertGreater(np.linalg.cond(x),1e4);self.assertTrue(core_close(qr[2],beta[2]));self.assertEqual(grade(True,['高条件数']),'PASS_WITH_CAVEAT')
  self.assertFalse(core_close(qr[2]+1e-4,beta[2]));self.assertEqual(grade(False,['高条件数']),'HOLD')
 def test_no_fit_call(self):
  tree=ast.parse(Path(__file__).read_text(encoding='utf-8'))
  self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in {'fit','optimize','minimize'} for n in ast.walk(tree)))

def wm00_qa(contract,cohort,person,saved_rows,saved_csv):
 require(not person.person_id.duplicated().any(),'WM00重复参与者')
 person=person.copy();person['absolute_recovery']=person.wealth_tplus1-person.wealth_t
 fields=['wealth_tminus1','wealth_t','wealth_tplus1','absolute_loss','absolute_recovery','recovery_fraction_raw'];verified=[]
 for group in ['substantial_recovery','persistent_or_limited_recovery']:
  subset=person.loc[person.recovery_status.eq(group)]
  for variable in fields:
   values=subset[variable].to_numpy(float);require(np.isfinite(values).all(),'WM00非有限值')
   units='ratio' if variable=='recovery_fraction_raw' else contract['analyses'][0]['amount_units'][cohort]
   match=[r for r in saved_rows if r['cohort']==cohort and r['recovery_status']==group and r['variable']==variable];require(len(match)==1,'WM00单元缺失或重复')
   actual=match[0];tab=saved_csv.loc[saved_csv.cohort.eq(cohort)&saved_csv.recovery_status.eq(group)&saved_csv.variable.eq(variable)];require(len(tab)==1,'WM00 CSV单元缺失或重复')
   expected={'n':len(values),'median':float(np.quantile(values,.5,method='linear')),'q1':float(np.quantile(values,.25,method='linear')),'q3':float(np.quantile(values,.75,method='linear')),'negative_n':int((values<0).sum()),'negative_percent':float((values<0).mean()*100),'zero_n':int((values==0).sum()),'zero_percent':float((values==0).mean()*100)}
   for field,value in expected.items():
    if field in ['n','negative_n','zero_n']:require(actual[field]==value==tab.iloc[0][field],'WM00人数不符')
    else:
     amount=field in ['median','q1','q3'] and variable!='recovery_fraction_raw'
     require(close(actual[field],value,TOL['amount_rtol'] if amount else TOL['file_rtol'],TOL['amount_atol'] if amount else TOL['file_atol']),'WM00独立分位数或比例不符')
     require(close(actual[field],tab.iloc[0][field]),'WM00 JSON/CSV不符')
   require(actual['unit']==units==tab.iloc[0]['unit'],'WM00单位不符')
   verified.append({'cohort':cohort,'recovery_status':group,'variable':variable,'status':'PASS','participants':len(values),'unit':units})
 return verified

def model_qa(contract,cohort,aid,frame,selected,x,sample,matrix,aggregate,diagnostic_map,estimates,asset_index,warning_evidence):
 folder=OUT/'Model Results'/aid/cohort;obj=read(folder/'result_object.json');diag=read(folder/'diagnostics.json');design=read(folder/'design_matrix.json');anchor=read(folder/'sample_anchor.json');logs=read(folder/'optimization_log.json')
 require(obj['status']=='SUCCESS' and diag['status']=='SUCCESS','Phase 2没有成功状态')
 require(obj==aggregate[(aid,cohort)] and diag==diagnostic_map[(aid,cohort)],'单元与汇总对象不符')
 require(sample==obj['sample']==anchor==diag['sample'],'实际样本与保存锚点不符')
 for key in ['columns','omitted_columns','n_columns','rank']:require(matrix[key]==design[key],'实际矩阵与保存设计不符')
 for key in ['condition_number','min_singular_value','rank_tolerance']:require(close(matrix[key],design[key]),'矩阵数值诊断不符')
 rows=estimates.loc[estimates.analysis_id.eq(aid)&estimates.cohort.eq(cohort)];require(len(rows)==1 and rows.iloc[0]['status']=='SUCCESS','估计汇总缺失或状态不符')
 idxrow=asset_index.loc[asset_index.analysis_id.eq(aid)&asset_index.cohort.eq(cohort)];require(len(idxrow)==1,'资产索引重复或缺失')
 require(idxrow.iloc[0].contract_sha256==CONTRACT_SHA and idxrow.iloc[0].result_object_path==rel(folder/'result_object.json'),'资产索引合同或路径错误')
 require(int(idxrow.iloc[0].input_population_n)==sample['participants'] and int(idxrow.iloc[0].model_fi_rows)==sample['fi_observations'],'资产索引人数不符')
 names=list(x.columns);require(obj['fixed_effect_names']==names,'固定效应列名错误')
 coefficients=obj['fixed_effects'];require(len(coefficients)==len(names) and [c['term'] for c in coefficients]==names,'固定效应对象不完整')
 coefcsv=csv(folder/'fixed_effects.csv');require(coefcsv.term.tolist()==names,'固定效应CSV列名不符')
 cov=np.asarray(obj['fixed_effect_covariance'],float);require(cov.shape==(len(names),len(names)),'协方差维度不符')
 require(np.isfinite(cov).all() and (np.diag(cov)>0).all() and close(cov,cov.T),'协方差非有限/非对称/SE异常')
 beta=np.asarray([c['estimate'] for c in coefficients],float);require(np.isfinite(beta).all(),'非有限固定效应')
 for i,c in enumerate(coefficients):
  se=float(np.sqrt(cov[i,i]));expected={'estimate':beta[i],'se':se,'ci_lower':beta[i]-Z*se,'ci_upper':beta[i]+Z*se,'p_value':float(2*norm.sf(abs(beta[i]/se)))}
  for field,value in expected.items():require(close(c[field],value) and close(coefcsv.iloc[i][field],c[field]),'β/SE/CI/P或CSV不一致：'+field)
  require(bool(c['is_primary_estimand'])==(c['term']==TERM),'estimand标签错误')
 covcsv=csv(folder/'fixed_effect_covariance.csv');require(len(covcsv)==len(names)**2 and not covcsv.duplicated(['term_i','term_j']).any(),'协方差CSV单元不完整')
 covtable=covcsv.pivot(index='term_i',columns='term_j',values='covariance').reindex(index=names,columns=names).to_numpy(float);require(close(covtable,cov),'协方差CSV不符')
 with np.load(folder/'model_parameters.npz',allow_pickle=False) as raw:parameters={k:np.array(raw[k],copy=True) for k in raw.files}
 require(set(parameters)=={'parameter_names','parameters','fixed_effect_covariance','hessian','residual_variance','random_intercept_variance'},'NPZ含未识别字段')
 pnames=parameters['parameter_names'].tolist();require(pnames[:-1]==names and len(pnames)==len(names)+1,'参数维度或顺序不符')
 require(close(parameters['parameters'][:-1],beta) and close(parameters['fixed_effect_covariance'],cov),'NPZ参数或协方差不符')
 hessian=parameters['hessian'];hcsv=pd.read_csv(folder/'hessian.csv',index_col=0,float_precision='round_trip',encoding='utf-8')
 require(hcsv.index.tolist()==pnames and hcsv.columns.tolist()==pnames and close(hcsv.to_numpy(float),hessian),'Hessian CSV/NPZ不符')
 require(not diag['hessian_singular'] and np.isfinite(hessian).all(),'Hessian不充分')
 sigma=float(parameters['residual_variance']);tau=float(parameters['random_intercept_variance']);require(sigma>0 and tau>0,'方差分量非正')
 require(close(sigma,diag['residual_variance']) and close(tau,diag['random_intercept_variance']) and close(parameters['parameters'][-1],tau/sigma),'方差参数化不符')
 require([l['optimizer'] for l in logs]==['bfgs','lbfgs'] and logs==diag['optimization'],'优化顺序或日志不符')
 core=names.index(TERM);require(obj['estimand']==coefficients[core],'核心参数对象不符')
 for key in ['estimate','se','ci_lower','ci_upper','p_value']:require(close(rows.iloc[0][key],coefficients[core][key]),'汇总核心估计不符')
 y=frame.fi_primary.to_numpy(float);values=x.to_numpy(float);ids=frame.person_id.to_numpy()
 warning_rows=[]
 for log in logs:
  require(log['converged'] and not log['error'],'优化器未收敛或报错')
  require(log['parameter_names']==pnames and np.isfinite(log['parameters']).all(),'优化器参数顺序或数值不符')
  stage_sigma,stage_tau,stage_ll=optimizer_scale_and_ll(y,values,ids,np.asarray(log['parameters'],float))
  require(abs(stage_ll-log['log_likelihood'])<=TOL['loglik_atol'],'优化器固定参数loglik不符')
  require(close(log['parameters'][core],log['recovery_time']),'优化器核心估计记录不符')
  for w in log['warnings']:
   require(w['message']=='The MLE may be on the boundary of the parameter space.' and w['category']=='ConvergenceWarning','出现尚未解释的warning')
   require(stage_tau<.01,'boundary warning触发条件无法复现')
   warning_rows.append({'optimizer':log['optimizer'],'message':w['message'],'saved_callsite':w['source_file']+':'+str(w['line']),'source_evidence':warning_evidence['path'],'reconstructed_profiled_residual_variance_at_stored_parameters':stage_sigma,'reconstructed_random_intercept_variance':stage_tau,'trigger_abs_random_variance_lt_0_01':True})
 beta_diff=abs(logs[0]['parameters'][core]-logs[1]['parameters'][core]);ll_diff=abs(logs[0]['log_likelihood']-logs[1]['log_likelihood'])
 require(beta_diff<=contract['optimizer']['beta3_absolute_tolerance'] and ll_diff<=contract['optimizer']['loglik_absolute_tolerance'],'双优化器超门槛')
 require(close(beta_diff,diag['beta3_optimizer_absolute_difference']) and close(ll_diff,diag['loglik_optimizer_absolute_difference']),'保存优化差异不符')
 require(close(logs[1]['parameters'],parameters['parameters']),'L-BFGS/NPZ参数不一致')
 qr,svd,ll,stable,numeric=equivalent_linear_checks(y,values,ids,beta,tau,sigma,hessian,cov)
 stable_se=float(np.sqrt(stable[core,core]));saved_se=coefficients[core]['se']
 require(core_close(qr[core],beta[core]) and core_close(svd[core],beta[core]),'核心β等价复核未通过')
 require(core_close(stable_se,saved_se),'核心SE等价复核未通过')
 require(abs(ll-diag['log_likelihood'])<=TOL['loglik_atol'],'独立Gaussian loglik未通过')
 require(numeric['balanced_inverse_backward_error']<=1e-10 and numeric['fitted_value_max_absolute_difference']<=1e-8,'线性代数误差放大')
 require(numeric['fixed_covariance_min_eigenvalue']>0 and numeric['information_min_eigenvalue']>0,'信息或协方差正定性未通过')
 primary=next(p for p in read(PRIMARY)['primary_estimand_rows'] if p['cohort']==cohort)
 require(close(diag['SE_ratio_to_frozen_primary'],saved_se/primary['se']),'相对主模型SE比值不符')
 caveats=['高条件数主要涉及损失调整项及其时间交互；核心项等价复核通过，但需报告弱识别限制'] if (aid,cohort) in HIGH else []
 status=grade(True,caveats)
 numeric.update({'qr_core_beta':float(qr[core]),'svd_core_beta':float(svd[core]),'saved_core_beta':float(beta[core]),'qr_core_beta_absolute_difference':float(abs(qr[core]-beta[core])),'svd_core_beta_absolute_difference':float(abs(svd[core]-beta[core])),'stable_core_SE':stable_se,'saved_core_SE':saved_se,'stable_core_SE_relative_difference':float(abs(stable_se-saved_se)/saved_se),'gaussian_loglik':ll,'loglik_absolute_difference':float(abs(ll-diag['log_likelihood']))})
 detail={'analysis_id':aid,'cohort':cohort,'status':status,'automated_checks_status':'PASS','scientific_author_review':'PENDING','sample':sample,'estimand':coefficients[core],'matrix':matrix,'equivalent_checks':numeric,'warnings':warning_rows,'random_intercept_variance':tau,'residual_variance':sigma,'optimizer_beta_difference':beta_diff,'optimizer_loglik_difference':ll_diff,'SE_ratio_to_primary':saved_se/primary['se'],'caveats':caveats,'source_kind':'NEW_LMM_FIT','limitations':['未序列化包含微观记录的完整模型对象；复核依据冻结输入、保存参数和矩阵','未进行个体删除影响分析、随机结构或时间函数替代模型'],'trace':{'contract_path':rel(CONTRACT),'contract_sha256':CONTRACT_SHA,'phase2_result_path':rel(folder/'result_object.json'),'phase2_result_sha256':sha(folder/'result_object.json'),'phase2_diagnostic_path':rel(folder/'diagnostics.json'),'actual_code_version':idxrow.iloc[0].execution_code_version,'model_code_sha256':sha(CODE/'run_models.py'),'input_assets':[{'path':a['path'],'sha256':a['sha256'],'freeze_path':a['freeze_path'],'freeze_sha256':a['freeze_sha256']} for a in contract['source_freezes'] if a['cohort']==cohort]}}
 return detail

def write_memos(details,primary,thresholds,wm00,overall,issues):
 success=[d for d in details if d['status']!='HOLD'];primary_map={p['cohort']:p for p in primary['primary_estimand_rows']}
 counts={status:sum(d['status']==status for d in details) for status in ['PASS','PASS_WITH_CAVEAT','HOLD']}
 lines=['# Healthcare-4603924：Phase 3财富测量QA报告','',f"建议状态：**{overall}**。作者科学审查：**PENDING**。本报告是自动核验与AI辅助统计审查，不构成人工批准，也不授权Phase 4。",'',f"14个真实模型：PASS {counts['PASS']}，PASS_WITH_CAVEAT {counts['PASS_WITH_CAVEAT']}，HOLD {counts['HOLD']}。KLoSA WM04冻结引用单列。",'', '## P3.1 结果资产与复现链','', '合同、Preflight、九份冻结Parquet及freeze、冻结Primary和Phase 2 manifest的164份输出、8份代码逐项核对。第一次执行9个模型的81份文件哈希及代码备份再次核验，科学模块哈希与输入/设计函数AST相同；续跑5个模型没有更改科学逻辑。文件持久化和续跑控制的差异单独保留在Phase 2过程记录。','', '历史Primary仍对应KH修复前源码；当前KH修复源码为不同身份，历史freeze没有改写。逐模型合同、输入、实际代码版本、结果路径及SHA记录于qa_validation.json的trace字段。','', '| 队列 | WM02完整精度P25 | 参考N | 参考并列N | 主要样本并列N（恢复/参照） |','|---|---:|---:|---:|---:|']
 for c,t in thresholds.items():lines.append(f"| {c} | {t['locked_P25']!r} | {t['reference_n']} | {t['reference_equal_n']} | {t['primary_equal_n']}（{t['primary_equal_recovery_n']}/{t['primary_equal_comparison_n']}） |")
 lines+=['','P25采用参考人群全部冻结首次索引冲击者，严格>=；主要样本的全部并列者保留。KLoSA参考24人中21人本来属于主要样本，另外3人不因WM02被排除，区别来自原主样本资格。WM03独立核实原主要样本每人一次的log1p均值和样本SD，其他标准化沿用冻结常数。40列模型的原Q1及Q1×time均为结构性零列，实际矩阵保留Q5参照和其他调整项。','', '## P3.2 数值与诊断','', '| 规格 | 队列 | QA | N（恢复/参照） | FI行 | β恢复×时间 | SE | 95%CI |','|---|---|---|---:|---:|---:|---:|---|']
 for d in sorted(details,key=lambda x:(x['analysis_id'],COHORTS.index(x['cohort']))):
  s=d.get('sample',{});e=d.get('estimand');values=f"{e['estimate']:.8f} | {e['se']:.8f} | [{e['ci_lower']:.8f}, {e['ci_upper']:.8f}]" if e else '— | — | —'
  lines.append(f"| {d['analysis_id']} | {d['cohort']} | {d['status']} | {s.get('participants','待核')}（{s.get('recovery_n','待核')}/{s.get('comparison_n','待核')}） | {s.get('fi_observations','待核')} | {values} |")
 lines+=['','核心β、SE、全部固定效应Wald CI/P、CSV/JSON/NPZ、双优化器、方差参数化及矩阵标签逐项检查。固定已保存方差对白化设计进行QR和SVD求解；独立评价保存参数下的Gaussian loglik；负Hessian对角平衡后Cholesky求解核验观察信息协方差。没有调用拟合或优化入口，GLS条件方差没有被当成原模型Wald SE。','', '### 三个高条件数模型','', '| 规格/队列 | κ(X) | κ固定效应协方差 | 平衡后κ信息矩阵 | β等价绝对误差 | SE相对误差 | 最弱方向核心载荷 |','|---|---:|---:|---:|---:|---:|---:|']
 weak_notes=[]
 for d in success:
  if (d['analysis_id'],d['cohort']) in HIGH:
   m=d['matrix'];n=d['equivalent_checks'];lines.append(f"| {d['analysis_id']}/{d['cohort']} | {m['condition_number']:.1f} | {n['fixed_covariance_condition']:.3g} | {n['balanced_information_condition']:.1f} | {n['qr_core_beta_absolute_difference']:.3g} | {n['stable_core_SE_relative_difference']:.3g} | {m['weak_directions'][0]['core_loading']:.3g} |")
   labels='、'.join(z['term'] for z in m['weak_directions'][0]['largest_loadings'][:4]);weak_notes+=['',f"{d['analysis_id']}/{d['cohort']}最弱方向主要由{labels}组成。限制后每人一次的标准化损失SD={m['restricted_participant_standardized_loss_SD']:.6g}。核心项等价复核通过，但损失相关干扰参数不能据此视为充分精确；等级为{d['status']}。",'']
 lines+=weak_notes
 if success:
  lines+=['',f"全部模型复核的最大核心β绝对误差={max(d['equivalent_checks']['qr_core_beta_absolute_difference'] for d in success):.6g}；最大核心SE相对误差={max(d['equivalent_checks']['stable_core_SE_relative_difference'] for d in success):.6g}；最大Gaussian loglik绝对误差={max(d['equivalent_checks']['loglik_absolute_difference'] for d in success):.6g}。全部固定效应的最大QR复核绝对误差={max(d['equivalent_checks']['all_fixed_beta_max_absolute_difference'] for d in success):.6g}，对应预测值最大误差={max(d['equivalent_checks']['fitted_value_max_absolute_difference'] for d in success):.6g}。",'']
 lines+=['','最弱方向载荷依赖当前编码与尺度，只作几何诊断；核心可靠性结论同时依赖等价β、观察信息SE、loglik及优化一致性，不能仅凭小载荷判断。','', '### Warning与诊断边界','', '每个优化器的boundary警告都按保存参数独立重现profiled scale及随机截距方差，并追溯statsmodels 0.15.0源码的abs(random variance)<0.01规则；没有发现未解释的新warning。该绝对尺度触发器本身不证明方差贴近0，也不证明随机截距足够或线性假设成立。源码摘录、哈希和逐warning触发数值另存。','', '未完成随机斜率、非线性、个体删除影响模型或新增meta。Phase 2的设计杠杆/残差摘要仍只是筛查代理；本轮没有升级其证据等级。','', '## P3.3 描述结果与分母','', f'WM00独立核验{len(wm00)}个单元，计数单位是每名原主要参与者一次；金额单位分别为2022等值化USD、2020等值化GBP和2022价格万KRW。零与合法负值保留，分位数线性插值。15个规格分母及两组构成与合同和Preflight完全一致。','', 'ELSA WM01：306人（88/218），1,163行；KLoSA WM05：801人（282/519），3,412行。汇总18行包括3条冻结Primary引用、14条新拟合及1条KLoSA WM04引用，来源不混淆。全精度机器结果保持原样；显示舍入/负零检查单列，未改写Phase 2表。','', '## P3.4–P3.5 科学解释及覆盖','', '科学解释见scientific_interpretation.md；审稿意见逐条映射见reviewer_issue_mapping.md。技术可用性与科学含义分开判断，测量误差、回归均值和流动财富组成问题没有被本轮彻底解决。','', '## Phase 4建议与停止点','', f'建议：{overall}。三个高条件数模型如上列示非阻断数值限制；未发现新的构造或模型输入错误时，作者可在确认解释及限制后另行授权冻结。当前作者状态PENDING，Phase 4未获执行授权，本轮没有创建freeze。', '', '所有已存在文件前后哈希、Git HEAD和tracked diff核验；仅新增QA脚本和Phase3文件。不修改合同、历史源码、冻结数据、Phase 2、正文、补充材料或S6，不commit/push。']
 if issues:lines+=['','## 阻断/异常','']+['- '+i for i in issues]
 (QA/'phase3_qa_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 science=['# Scientific Interpretation Memo：财富测量稳健性','', '证据角色：AI_ASSISTED_REVIEW；作者确认：PENDING。用于作者和统计专家审阅，不是可直接粘贴投稿稿件的Results/Discussion。','', '## HRS','']
 hrs=[d for d in success if d['cohort']=='HRS'];elsa=[d for d in success if d['cohort']=='ELSA'];klosa=[d for d in success if d['cohort']=='KLoSA']
 if len(hrs)==5 and all(d['estimand']['ci_lower']>0 for d in hrs):science+=['五个规格的恢复×时间估计均为正且95%CI未跨零，支持“在已检验的财富定义处理和限制人群中，调整后关联相对稳定”的限定表述。WM03与Primary使用同一主要人群，β从约0.001268变为0.001260；该转换下变化较小。','', 'WM01/02/04/05改变分母，不能把其系数差异视为同一样本的纯模型效应。WM05是压力测试，其高条件数主要涉及损失调整项；核心数值复核通过不等于该项估计精确。统计显著不证明因果关系或临床意义，也不能证明已排除极端个体影响。']
 else:science+=['HRS完整结论须结合模型HOLD/不确定项处理；不默认宣称五个规格均支持稳定。']
 science+=['','## ELSA','', '目前各规格核心估计仍为负，区间跨零；WM04上界接近0但不能将其解释为与其他规格存在显著差异。WM01的N由699降至306，恢复/参照为88/218；SE由0.001142增至0.001616。现有等价核验支持核心项数值可用，精度变化应首先与限制人群及信息量变化同时报告，不能全部归因于数值病态。','', '允许：“点估计在这些规格中仍为负，但估计不确定性及人群组成不同。”不允许：“恢复没有关联”“排除Q1证实原结果错误”或“跨规格差异具有统计显著性”。','', '## KLoSA','', 'WM02限制后N=1,870，β约−0.000279；Primary N=2,502，β约0.000579。两者CI均跨零，不能用方向翻转推断人群间关联存在统计学差异，也不能把它作为同样本人群内的直接规格比较。跨所有限制规格的“方向稳定”表述不成立。','', 'WM05 N=801（282/519），β约0.000768，SE约0.000955，CI跨零；精度较Primary降低。WM03保持原主要人群，β约0.000674，CI仍跨零。KLoSA WM04是固定除数下的完全相同引用，不提供额外独立稳健性证据。','', '## 测量、经济含义与证据边界','', 'WM00直接给出财富三个时点、绝对损失/恢复和恢复比例的分布，零负财富均按合同保留。财富金额不跨队列直接比较。净财富为负时，财富损失可超过100%；恢复至少损失的一半可以表现为负债减少，既不保证landmark财富为正，也不保证恢复到冲击前水平。','', '示例仅用于公式解释、不是微观数据：冲击前100、冲击时−100、landmark为0，则损失比例2，恢复比例0.5，仍未回到冲击前100。比较不同初始财富者时，相同恢复比例不代表相同绝对资源。','', 'WM01回应最低五分位集中问题；WM02是从全部首次索引冲击者得到的经验性P25限制，不是经济学上验证的实质性损失阈值，也不能消除所有低分母问题。WM03评价损失调整函数形式；WM04评价等值化除数稳定人群；WM05是冲击时正财富人群压力测试。','', '这些分析没有直接校正测量误差或回归均值，没有判断每次冲击是否是真实经济事件；没有解决反向因果、未测混杂、存活/随访选择、健康状态在shock与landmark间的变化。财富组成及流动财富问题仍需作者决定是否扩展，不能声称已解决。','', '## 作者需确认的结论','', f'技术与AI辅助审查建议为{overall}，但作者确认仍PENDING。请确认HRS的限定稳定性表述、ELSA的精度/人群解释、KLoSA方向不稳定及全部未覆盖问题；本轮不直接修改稿件或启动冻结。']
 science[science.index('## 测量、经济含义与证据边界')+1:science.index('## 测量、经济含义与证据边界')+1]=['','沿用基础审计已核实的定义：损失比例 L=(W[t−1]−W[t])/W[t−1]；恢复比例 R=(W[t+1]−W[t])/(W[t−1]−W[t])。冲击前财富必须正，合格冲击要求L≥0.75，恢复分类严格R≥0.50。这里重述既定公式，不重新识别冲击或生成暴露。']
 (QA/'scientific_interpretation.md').write_text('\n'.join(science)+'\n',encoding='utf-8')
 exceptions=['# Phase 3异常与遗留事项','',f'整体建议：{overall}；作者审查PENDING。','', '## 数值及追溯阻断项','']+(['- '+i for i in issues] if issues else ['未发现阻断项。'])
 exceptions+=['','## 非阻断限制','', '- HRS WM05、ELSA WM01和ELSA WM05高条件数涉及损失相关干扰项；核心复核通过，保留PASS_WITH_CAVEAT及弱方向证据。','- 28条boundary warning的源码触发可复现；保留解释，不能据此认证随机结构或全部模型假设。','- 历史运行未保存含微观记录的完整拟合对象或逐行输入指纹。本轮用冻结输入、确定性代码、保存参数、β/协方差/loglik数值复核建立证据；不将其描述为直接检查历史内存对象。','- 极端个体的正式删除影响分析未开展；QR/SVD仅核验既定模型的数值等价性。','- P25限制未消除全部低财富基数，误差及回归均值未直接校正。','- 限制规格比较同时改变人群，不提供正式跨规格差异检验。','- 财富组成/流动财富、随机斜率、非线性、IPCW及其他返修模块未在本轮执行。','- 内部结果含小单元；外发前独立进行披露检查，不改写当前内部资产。','', '## 最小下一步','', '作者审阅三份核心报告及限定表述后，另行授权Phase 4。若出现HOLD，先处理该模型对应的具体数值或追溯证据；不得自动改变规格或重拟合。']
 (QA/'exceptions.md').write_text('\n'.join(exceptions)+'\n',encoding='utf-8')

def reviewer_mapping():
 editor=ROOT/'07_manuscript/Round1/Editor.md';reviewers=ROOT/'07_manuscript/Round1/Reviewers.md'
 def location(path,needle):
  matches=[i+1 for i,line in enumerate(path.read_text(encoding='utf-8').splitlines()) if needle in line]
  require(matches,'审稿原文定位失败：'+needle);return rel(path)+':'+str(matches[0])
 rows=[
  ('AE-wealth-validity',editor,'1) the first is related','低财富基数、测量误差、零负值及绝对金额','部分回应','WM00金额分布、WM01排除Q1、WM02绝对损失P25、WM03函数转换、WM05正财富压力测试','未校正误差/回归均值；未验证实际经济事件或财富组成'),
  ('AE-first-decision-1',editor,'measurement error and regression','测量误差与回归均值；排除低绝对损失或Q1','部分回应','WM01与WM02直接完成要求的限制分析；WM00/03提供补充证据','分析要求已覆盖；误差机制只能讨论，不能宣称排除'),
  ('AE-KLoSA-Q1',editor,'2) the second is related','KLoSA最低五分位和参考人群','直接回应（既有基础审计）','foundation_audit_v1及addendum证明构造/流失链，本轮核对其证据哈希','不是本轮新增数据重建；后续需忠实报告参考分布'),
  ('AE-liquid-composition',editor,'the wealth composition','住房/非流动财富与流动财富恢复','未覆盖','本轮只有总净财富','留作待作者决定的可选扩展或明确限制'),
  ('R2-recovery-formula',reviewers,'Clarify the mathematical definition','恢复公式、跨零及负值处理','直接回应（证据层）','基础审计公式；WM00金额/恢复分布和科学备忘录纯示例','稿件尚未修改；50%恢复不等于正财富或恢复原水平'),
  ('R3-negative-formula',reviewers,'The manuscript defines a major wealth shock','负财富、超过100%的损失和数学示例','直接回应（证据层）','基础公式审计、WM00、WM05及纯数学示例','WM05是限制人群压力测试，不证明负值原记录错误'),
  ('R3-initial-wealth',reviewers,'whether the ability to recover','初始财富差异与残余经济资源','部分回应','WM00绝对金额、WM01/02分布限制','未正式检验效应修饰；相同恢复比例不等于相同资源'),
  ('R3-null-heterogeneity',reviewers,'The overall estimate is','区分无关联证据与关联不存在','直接回应（解释建议）','逐队列CI及科学备忘录；KLoSA WM02方向翻转','不进行新meta、等效性检验或跨规格差异检验'),
  ('R1-heterogeneity',reviewers,'Cross-cohort heterogeneity','强调各队列差异','部分回应','三队列分开估计与解释，不声称共同稳定方向','国家背景解释仍需文献和作者审查'),
  ('AE-random-slope',editor,'3) the thirs is','随机斜率','未覆盖','本轮仅冻结随机截距规格','后续独立模块；不扩展本次QA'),
  ('R2-time-assumption',reviewers,'Clarify assumptions regarding','线性与随机结构诊断','部分回应','本轮检验数值、协方差、warning来源','不能替代线性/随机斜率敏感性及残差假设审查'),
  ('AE-attrition',editor,'4) another important issue','死亡及失访','未覆盖','财富测量模块没有IPCW或新增选择模型','独立返修模块'),
  ('AE-landmark-health',editor,'5) the fifth issue','landmark健康调整','未覆盖','保留原FI调整规格','独立返修模块'),
  ('AE-context-period',editor,'7) Another one','时期和上下文','未覆盖','未新增日历期、事件或住房调整','独立决定'),
  ('AE-clinical',editor,'6) then another','实际重要性/等效性','部分回应','提供全精度效应及CI，强调非因果和非临床认证','未锁定临床等效界值或新检验'),
  ('AE-meta',editor,'Meta-analytic synthesis','小队列数下meta不稳定','未覆盖（本模块）','保持旧冻结meta并强调队列结果','KH修复已有基础收尾证据；本次无新meta'),
  ('AE-no-shock',editor,'8) finally','无冲击参照组','未覆盖','仅冻结冲击经历者两组','不新增参照样本'),
 ]
 lines=['# 审稿意见覆盖矩阵','', '覆盖状态指现有证据能回答的范围，不代表已提交回复或修改稿件。原文定位来自本地Round1材料，哈希列入QA manifest。','', '| 意见ID | 原文位置 | 问题 | 覆盖 | 现有证据 | 遗留限制 |','|---|---|---|---|---|---|']
 for key,path,needle,issue,status,evidence,limit in rows:lines.append(f'| {key} | {location(path,needle)} | {issue} | {status} | {evidence} | {limit} |')
 (QA/'reviewer_issue_mapping.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
 resume='--resume-incomplete-qa' in sys.argv
 require(not QA.exists() or (resume and (QA/'incomplete_attempt_01').is_dir() and not any(QA.glob('*.*'))),'Phase3目录已有证据，禁止覆盖，请先审阅')
 started=datetime.now(timezone(timedelta(hours=8))).isoformat();contract,phase2,asset_qa=verify_assets();before=snapshot();head=git('rev-parse','HEAD');tracked=git('status','--porcelain','--untracked-files=no');diff=git('diff','--no-ext-diff')
 QA.mkdir(parents=True,exist_ok=resume);save(QA/'protected_hashes_before.json',before)
 if resume:
  require(before==read(QA/'incomplete_attempt_01/protected_hashes_before.json'),'初次QA准备中断后受保护资产变化')
  save(QA/'qa_preparation_interruption.json',{'cause':'合同categorical_covariates是字典；首次独立QA准备阶段列表连接类型错误，未进入模型数值核验','resolution':'读取类别字段名列表；保留初次脚本和全部初次记录，不更改Phase2或科学规则','archived_evidence_path':rel(QA/'incomplete_attempt_01'),'new_fit_calls':0})
 (QA/'git_status_before.txt').write_text(git('status','--porcelain','--untracked-files=all')+'\n',encoding='utf-8')
 (QA/'git_diff_before.patch').write_text(diff+'\n',encoding='utf-8')
 stream=io.StringIO();testresult=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticTests));(QA/'synthetic_tests.log').write_text(stream.getvalue(),encoding='utf-8');require(testresult.wasSuccessful(),'合成测试失败，禁止继续数值复核')
 source_lines,startline=inspect.getsourcelines(MixedLM.fit);source=''.join(source_lines);start=next(i for i,line in enumerate(source_lines) if 'f1 = ' in line);end=start+10
 warning_excerpt=''.join(f'{startline+i}: {source_lines[i]}' for i in range(start,end));(QA/'warning_source_excerpt.txt').write_text(warning_excerpt,encoding='utf-8')
 warning_evidence={'path':rel(QA/'warning_source_excerpt.txt'),'library':'statsmodels.regression.mixed_linear_model.MixedLM.fit','statsmodels_version':statsmodels.__version__,'method_source_sha256':hashlib.sha256(source.encode('utf-8')).hexdigest(),'installed_module_sha256':sha(inspect.getsourcefile(MixedLM)),'predicate':'abs(diagonal(random_effect_covariance)) < 0.01; only one random intercept and no variance components'};save(QA/'warning_source_evidence.json',warning_evidence)
 print('ASSETS PASS; SYNTHETIC TESTS PASS; no fitting/optimization calls',flush=True)
 preflight=read(R1/'Preflight/preflight_results.json');bundle=read(OUT/'result_object.json');aggregate={(o['analysis_id'],o['cohort']):o for o in bundle['models']};diag_map={(o['analysis_id'],o['cohort']):o for o in read(OUT/'model_diagnostics.json')['models']};estimates=csv(OUT/'cohort_estimates.csv');asset_index=csv(OUT/'analysis_asset_index.csv');primary=read(PRIMARY)
 require(len(aggregate)==len(diag_map)==15 and len(estimates)==18 and len(asset_index)==18,'整体规格、引用或资产索引数不符')
 for p in primary['primary_estimand_rows']:
  row=estimates.loc[estimates.analysis_id.eq('PRIMARY')&estimates.cohort.eq(p['cohort'])];require(len(row)==1 and row.iloc[0]['status']=='REFERENCE_ONLY','冻结Primary标签不符')
  for k in ['estimate','se','ci_lower','ci_upper','p_value']:require(close(row.iloc[0][k],p[k]),'冻结Primary引用数值不符')
 wm00_saved=read(OUT/'Descriptives/WM00_descriptives.json')['rows'];wm00_csv=csv(OUT/'Descriptives/WM00_descriptives.csv');require(len(wm00_saved)==len(wm00_csv)==36,'WM00总单元数不符')
 details=[];references=[];wm00=[];thresholds={};issues=[];sample_checks=[]
 for cohort in COHORTS:
  print('QA '+cohort,flush=True)
  panel,person,thresholds[cohort]=independently_load(contract,cohort);wm00+=wm00_qa(contract,cohort,person,wm00_saved,wm00_csv)
  pf=next(c for c in preflight['cohorts'] if c['cohort']==cohort)
  for aid in ['WM01','WM02','WM03','WM04','WM05']:
   try:
    frame,selected,x,sample,matrix=independently_prepare(contract,cohort,aid,panel,person,pf)
    folder=OUT/'Model Results'/aid/cohort
    if aid=='WM02':
     included=selected.absolute_loss.eq(thresholds[cohort]['locked_P25']).sum();require(int(included)==thresholds[cohort]['primary_equal_n'],'WM02边界主要参与者未全保留');thresholds[cohort]['primary_equal_retained_n']=int(included)
    sample_checks.append({'analysis_id':aid,'cohort':cohort,'sample':sample,'matrix_columns':matrix['n_columns'],'status':'PASS'})
    if aid=='WM04' and cohort=='KLoSA':
     obj=read(folder/'result_object.json');diag=read(folder/'diagnostics.json');p=next(p for p in primary['primary_estimand_rows'] if p['cohort']==cohort)
     require(obj['status']=='SKIP_IDENTICAL_TO_PRIMARY' and obj['sample']==sample and obj['estimand']==p and obj==aggregate[(aid,cohort)],'KLoSA WM04引用错误')
     require(read(folder/'optimization_log.json')==[] and diag['optimization']==[] and not (folder/'model_parameters.npz').exists(),'KLoSA WM04存在额外拟合')
     verify_digest(sha(ROOT/obj['reference_path']),obj['reference_sha256'])
     row=estimates.loc[estimates.analysis_id.eq(aid)&estimates.cohort.eq(cohort)].iloc[0]
     for k in ['estimate','se','ci_lower','ci_upper','p_value']:require(close(row[k],p[k]),'KLoSA WM04汇总引用不符')
     references.append({'analysis_id':aid,'cohort':cohort,'status':'PASS_REFERENCE_ONLY','execution_status':'SKIP_IDENTICAL_TO_PRIMARY','sample':sample,'estimand':p,'source_path':obj['reference_path'],'source_sha256':obj['reference_sha256']})
     print('WM04 KLoSA PASS_REFERENCE_ONLY',flush=True)
    else:
     detail=model_qa(contract,cohort,aid,frame,selected,x,sample,matrix,aggregate,diag_map,estimates,asset_index,warning_evidence);details.append(detail);print(aid+' '+cohort+' '+detail['status'],flush=True)
   except Exception as exc:
    message=aid+'/'+cohort+': '+type(exc).__name__+': '+str(exc);issues.append(message)
    (QA/(aid+'_'+cohort+'_qa_failure.txt')).write_text(traceback.format_exc(),encoding='utf-8')
    if (aid,cohort)==('WM04','KLoSA'):references.append({'analysis_id':aid,'cohort':cohort,'status':'HOLD','failure':message})
    else:details.append({'analysis_id':aid,'cohort':cohort,'status':'HOLD','automated_checks_status':'FAIL','failure':message,'scientific_author_review':'PENDING'})
    print('HOLD '+message,flush=True)
  del panel,person
 overall='QA_HOLD' if issues else ('QA_APPROVED_WITH_CAVEATS' if any(d['status']=='PASS_WITH_CAVEAT' for d in details) else 'QA_APPROVED')
 negative_zero=[]
 for d in details:
  if 'estimand' in d:
   for k in ['estimate','ci_lower','ci_upper']:
    if f"{d['estimand'][k]:.8f}"=='-0.00000000':negative_zero.append({'analysis_id':d['analysis_id'],'cohort':d['cohort'],'field':k})
 validation={'schema':'wealth_measurement_phase3_qa_v1','recommended_disposition':overall,'automated_asset_checks':asset_qa,'author_scientific_review':'PENDING','review_type':'AUTOMATED_QA_PLUS_AI_ASSISTED_STATISTICAL_REVIEW','phase4_execution_authorized':False,'tolerances':TOL,'runtime':runtime(),'model_qa':details,'frozen_references':references,'WM00':{'status':'PASS','independently_verified_cells':len(wm00),'cells':wm00},'sample_checks':sample_checks,'threshold_and_standardization_checks':thresholds,'negative_zero_core_display_at_8_decimals':negative_zero,'issues':issues,'tests':{'run':testresult.testsRun,'passed':testresult.wasSuccessful()},'new_fit_calls':0,'new_meta_calls':0}
 save(QA/'qa_validation.json',validation)
 table=[]
 for d in details:
  n=d.get('equivalent_checks',{});s=d.get('sample',{});e=d.get('estimand',{});m=d.get('matrix',{})
  table.append({'analysis_id':d['analysis_id'],'cohort':d['cohort'],'qa_status':d['status'],'author_review':'PENDING',**s,**{k:e.get(k) for k in ['estimate','se','ci_lower','ci_upper','p_value']},'design_columns':m.get('n_columns'),'design_rank':m.get('rank'),'design_condition':m.get('condition_number'),'min_singular_value':m.get('min_singular_value'),'covariance_condition':n.get('fixed_covariance_condition'),'balanced_information_condition':n.get('balanced_information_condition'),'equivalent_core_beta_abs_error':n.get('qr_core_beta_absolute_difference'),'equivalent_core_SE_relative_error':n.get('stable_core_SE_relative_difference'),'gaussian_loglik_abs_error':n.get('loglik_absolute_difference'),'optimizer_beta_abs_difference':d.get('optimizer_beta_difference'),'optimizer_loglik_abs_difference':d.get('optimizer_loglik_difference'),'random_intercept_variance':d.get('random_intercept_variance'),'residual_variance':d.get('residual_variance'),'caveats_or_failure':'；'.join(d.get('caveats',[])) or d.get('failure',''),'actual_code_version':d.get('trace',{}).get('actual_code_version')})
 pd.DataFrame(table).sort_values(['analysis_id','cohort']).to_csv(QA/'model_qa_matrix.csv',index=False,encoding='utf-8')
 save(QA/'high_condition_evidence.json',{'models':[d for d in details if (d['analysis_id'],d['cohort']) in HIGH]})
 write_memos(details,primary,thresholds,wm00,overall,issues);reviewer_mapping()
 after=snapshot();save(QA/'protected_hashes_after.json',after)
 require(before==after,'QA期间受保护文件发生变化')
 require(head==git('rev-parse','HEAD') and tracked==git('status','--porcelain','--untracked-files=no') and diff==git('diff','--no-ext-diff'),'HEAD或既有tracked变更发生变化')
 (QA/'git_status_after.txt').write_text(git('status','--porcelain','--untracked-files=all')+'\n',encoding='utf-8')
 save(QA/'boundary_validation.json',{'status':'PASS','protected_assets':len(before),'all_existing_hashes_unchanged':True,'HEAD_unchanged':True,'tracked_diff_unchanged':True,'new_fits':0,'new_meta':0,'result_freeze_created':False,'author_confirmation':'PENDING'})
 input_paths=set(before)-{p for p in before if p.startswith('external_raw/')}
 inputs=[{'path':p,'sha256':before[p]} for p in sorted(input_paths)]
 outputs=[{'path':rel(p),'sha256':sha(p)} for p in sorted(QA.rglob('*')) if p.is_file() and p.name not in ['qa_manifest.json','delivery_validation.json']]
 manifest={'schema':'wealth_measurement_phase3_qa_manifest_v1','started':started,'ended':datetime.now(timezone(timedelta(hours=8))).isoformat(),'git_HEAD':head,'git_dirty':True,'contract_sha256':CONTRACT_SHA,'phase2_manifest_path':rel(OUT/'manifest.json'),'phase2_manifest_sha256':sha(OUT/'manifest.json'),'QA_code':{'path':rel(Path(__file__)),'sha256':sha(__file__)},'inputs_and_protected_repository_assets':inputs,'external_raw_hashes':{k:before[k] for k in before if k.startswith('external_raw/')},'outputs':outputs,'recommended_disposition':overall,'author_scientific_review':'PENDING','runtime':runtime(),'boundary_validation':'PASS','new_fits':0,'new_meta':0,'phase4_authorized':False}
 save(QA/'qa_manifest.json',manifest)
 for item in outputs:verify_digest(sha(ROOT/item['path']),item['sha256'])
 save(QA/'delivery_validation.json',{'status':'PASS','QA_manifest_path':rel(QA/'qa_manifest.json'),'QA_manifest_sha256':sha(QA/'qa_manifest.json'),'output_files':len(outputs),'model_rows':len(table),'real_model_qa_count':14,'reference_only_count':len(references),'author_scientific_review':'PENDING','recommended_disposition':overall})
 print('DELIVERY '+overall+'; author PENDING; no Phase4',flush=True)

if __name__=='__main__':main()
