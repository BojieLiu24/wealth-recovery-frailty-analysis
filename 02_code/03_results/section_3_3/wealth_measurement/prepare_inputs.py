"""从锁定合同和冻结面板准备Phase 2输入；不修改任何输入资产。"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import os
import time
import uuid
import numpy as np
import pandas as pd
import patsy

CODE=Path(__file__).resolve().parent
ROOT=CODE.parents[3]
ROUND1=ROOT/'06_project_progress/round1_extra_analyses_v1'
PHASE=ROUND1/'Phase2'
OUTPUT=ROOT/'03_output/3.3 Robustness of the Main Findings/Wealth Measurement'
CONTRACT_PATH=ROUND1/'Contracts/wealth_measurement_round1_v1_contract.json'
EXPECTED_CONTRACT_SHA='a9d6769d479c80d6d0d886440e5f66ff6033eae5810959bb3c5eea18fdf86f4c'
PRIMARY=ROOT/'03_output/3.2 Wealth Recovery and Subsequent Frailty Progression'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT.resolve()).as_posix()
def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def require(v,msg):
 if not bool(v):raise ValueError(msg)
def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,np.generic):return clean(v.item())
 if isinstance(v,float) and not np.isfinite(v):return None
 return v
def write_json(p,d):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 # 先写独立临时文件再原子替换，避免Windows短暂文件占用破坏有效证据。
 target=Path(p);temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.tmp')
 temporary.write_text(json.dumps(clean(d),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 for attempt in range(10):
  try:os.replace(temporary,target);return
  except OSError:
   if attempt==9:raise
   time.sleep(.1*(attempt+1))
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,encoding='utf-8').strip()

def protected_snapshot():
 result={};new=(CODE,OUTPUT,PHASE)
 for dirname in ['00_admin','01_data','02_code','03_output','04_collaboration','06_project_progress','07_manuscript']:
  for p in (ROOT/dirname).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and not any(p.is_relative_to(x) for x in new):result[rel(p)]=sha(p)
 for n in ['AGENTS.md','pyproject.toml','uv.lock','renv.lock','.gitignore','.gitattributes','.Rprofile']:result[n]=sha(ROOT/n)
 prior=read_json(ROOT/'06_project_progress/foundation_audit_v1/audit_manifest.json')
 for c,v in prior['cohorts'].items():result['external_raw/'+c]=sha(v['source']['path'])
 return result

def verify_benchmark():
 require(sha(CONTRACT_PATH)==EXPECTED_CONTRACT_SHA,'合同SHA与批准版本不符')
 contract=read_json(CONTRACT_PATH);require(contract['status']=='LOCKED','合同未锁定')
 for artifact in [ROUND1/'Preflight/manifest.json',ROUND1/'Preflight/preflight_results.json']:
  key='manifest_sha256' if artifact.name=='manifest.json' else 'results_sha256'
  require(sha(artifact)==contract['preflight'][key],'Preflight哈希不符')
 delivery=read_json(ROUND1/'delivery_validation.json');require(delivery['status']=='PASS' and delivery['contract_sha256']==EXPECTED_CONTRACT_SHA,'交付记录不符')
 for a in delivery['outputs']:require(sha(ROOT/a['path'])==a['sha256'],'Phase 0/1交付文件变化：'+a['path'])
 assets=[]
 for x in contract['source_freezes']:
  require(sha(ROOT/x['path'])==x['sha256'],'冻结Parquet哈希不符')
  require(sha(ROOT/x['freeze_path'])==x['freeze_sha256'],'面板freeze变化')
  f=read_json(ROOT/x['freeze_path']);require(f['freeze_status']=='FROZEN','面板非FROZEN')
  assets.append(x)
 source=contract['historical_and_current_source_lineage']
 require(sha(ROOT/source['current_repaired_source_path'])==source['current_repaired_source_sha256'],'KH修复源码变化')
 require(sha(ROOT/source['historical_source_backup'])==source['historical_source_sha256'],'历史源码备份变化')
 closeout=read_json(ROOT/'06_project_progress/foundation_audit_addendum_v1/foundation_audit_addendum_v1_freeze.json')
 for p,h in closeout['historical_model_assets_sha256'].items():require(sha(ROOT/p)==h,'历史主模型资产变化')
 return contract,assets

def load_cohort(contract,cohort):
 def load(component):
  a=next(a for a in contract['source_freezes'] if a['component']==component and a['cohort']==cohort)
  return pd.read_parquet(ROOT/a['path'])
 p=load('analysis_panel');e=load('exposure_panel');w=load('wealth_panel')
 require(not p.duplicated(['person_id','wave']).any(),'分析面板键重复')
 p=p.loc[p.primary_complete_case_eligible.eq(True)].copy()
 lm=p.loc[p.fi_time_role.eq('landmark')].copy()
 require(not lm.person_id.duplicated().any(),'主样本landmark重复')
 e=e.loc[e.is_index_shock.eq(True)].copy();require(not e.person_id.duplicated().any(),'索引冲击重复')
 person=lm[['person_id','recovery_status','wealth_quintile','tminus1_wave','index_shock_wave','landmark_wave']].merge(e[['person_id','candidate_shock_wave','wealth_tminus1','wealth_t','wealth_tplus1','recovery_fraction_raw','wealth_loss_fraction','recovery_status']].rename(columns={'recovery_status':'exposure_status'}),on='person_id',how='left',validate='one_to_one')
 require(len(person)==len(lm) and person.recovery_status.eq(person.exposure_status).all(),'主样本/Exposure连接不符')
 require(person.index_shock_wave.eq(person.candidate_shock_wave).all(),'索引波次不符')
 require(person.tminus1_wave.eq(person.index_shock_wave-1).all() and person.landmark_wave.eq(person.index_shock_wave+1).all(),'波次不相邻')
 require(not w.duplicated(['person_id','wave']).any(),'Wealth键重复')
 for role,key,amount in [('pre','tminus1_wave','wealth_tminus1'),('shock','index_shock_wave','wealth_t'),('landmark','landmark_wave','wealth_tplus1')]:
  wx=w[['person_id','wave','wealth_total_real_equiv','wealth_equiv_divisor']].rename(columns={'wave':key,'wealth_total_real_equiv':'amount_'+role,'wealth_equiv_divisor':'div_'+role})
  person=person.merge(wx,on=['person_id',key],how='left',validate='many_to_one')
  require(np.allclose(person[amount],person['amount_'+role],rtol=1e-12,atol=1e-8,equal_nan=False),'金额回查不符')
 require(person[['div_pre','div_shock','div_landmark']].notna().all().all(),'除数缺失')
 require(np.isfinite(p.fi_primary.to_numpy(float)).all(),'模型FI缺失或非有限')
 for z in contract['common_model']['categorical_covariates']:p[z]=p[z].astype(str)
 p['recovery']=p.recovery_status.eq('substantial_recovery').astype(int)
 for z in contract['common_model']['frozen_standardization_constants'][cohort]:p[z['variable']+'_z']=(p[z['variable']].astype(float)-z['mean'])/z['sample_sd']
 return p,person

def prepare_spec(contract,cohort,analysis_id,p,person):
 if analysis_id=='WM01':mask=person.wealth_quintile.ne(1)
 elif analysis_id=='WM02':mask=(person.wealth_tminus1-person.wealth_t).ge(contract['analyses'][2]['thresholds'][cohort]['value_full_precision'])
 elif analysis_id=='WM03':mask=pd.Series(True,index=person.index)
 elif analysis_id=='WM04':mask=person.div_pre.eq(person.div_shock)&person.div_shock.eq(person.div_landmark)
 elif analysis_id=='WM05':mask=person.wealth_t.gt(0)
 else:raise ValueError('未知分析ID')
 selected=person.loc[mask].copy();df=p.loc[p.person_id.isin(set(selected.person_id))].copy()
 if analysis_id=='WM03':
  z=contract['analyses'][3]['new_constants'][cohort]
  df['wealth_loss_fraction_z']=(np.log1p(df.wealth_loss_fraction.astype(float))-z['mean'])/z['sample_sd']
 anchor=next(x for x in contract['validated_sample_and_design_anchors'] if x['cohort']==cohort and x['analysis_id']==analysis_id)
 counts={'participants':len(selected),'fi_observations':len(df),'recovery_n':int(selected.recovery_status.eq('substantial_recovery').sum()),'comparison_n':int(selected.recovery_status.eq('persistent_or_limited_recovery').sum())}
 require(all(counts[k]==anchor[k] for k in counts),'限制样本锚点变化')
 require(set(df.person_id)==set(selected.person_id),'纵向记录人员集合不符')
 for v in ['tminus1_wave','index_shock_wave','landmark_wave']:require(df.groupby('person_id')[v].nunique().eq(1).all(),'个人时点不恒定')
 return df,selected,anchor,counts

def design_matrix(contract,frame,anchor):
 formula=contract['common_model']['original_formula'];rhs=formula.split('~',1)[1]
 references=contract['common_model']['categorical_reference_levels']
 require(all(str(ref) in set(frame[v].astype(str)) for v,ref in references.items()),'NOT_ESTIMABLE: 参考类别缺失')
 x=patsy.dmatrix(rhs,frame,NA_action='raise',return_type='dataframe')
 require(x.shape[1]==anchor['expected_design_columns'],'NOT_ESTIMABLE: 未批准的矩阵列数')
 require(np.linalg.matrix_rank(x.to_numpy())==x.shape[1],'NOT_ESTIMABLE: 矩阵降秩')
 pf=read_json(ROUND1/'Preflight/preflight_results.json')
 expected=next(m for c in pf['cohorts'] if c['cohort']==anchor['cohort'] for m in c['models'] if m['analysis_id']==anchor['analysis_id'])
 require(x.columns.tolist()==expected['design']['actual_basis']['columns'],'矩阵列名与Preflight不符')
 singular=np.linalg.svd(x.to_numpy(),compute_uv=False)
 diagnostics={'columns':x.columns.tolist(),'n_columns':x.shape[1],'rank':int(np.linalg.matrix_rank(x)),'rank_tolerance':float(singular[0]*max(x.shape)*np.finfo(float).eps),'condition_number':float(singular[0]/singular[-1]),'min_singular_value':float(singular[-1]),'max_singular_value':float(singular[0]),'singular_values':singular.tolist(),'omitted_columns':anchor['omitted_columns']}
 return x,diagnostics
