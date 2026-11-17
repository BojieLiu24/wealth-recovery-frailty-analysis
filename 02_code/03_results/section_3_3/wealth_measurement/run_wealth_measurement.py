"""财富测量Phase 2真实运行入口：14个拟合，零meta。"""
from datetime import datetime,timezone,timedelta
import sys
import platform
import time
import traceback
import subprocess
import numpy as np
import pandas as pd
import scipy
import statsmodels
import patsy
from prepare_inputs import *
from build_descriptives import summarize
from run_models import fit_unit
from validate_results import validate,validate_manifest

APPROVAL = Path(__import__('os').environ.get('WF_AUTHORIZATION_FILE', 'authorization-not-configured.json'))

def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def reference_rows(primary,contract):
 rows=[]
 for p in primary['primary_estimand_rows']:
  a=next(a for a in contract['validated_sample_and_design_anchors'] if a['cohort']==p['cohort'] and a['analysis_id']=='WM03')
  rows.append({'analysis_id':'PRIMARY','cohort':p['cohort'],'status':'REFERENCE_ONLY','source_kind':'FROZEN_PRIMARY','participants':p['participants'],'fi_observations':p['model_rows'],'recovery_n':a['recovery_n'],'comparison_n':a['comparison_n'],**{k:p[k] for k in ['estimate','se','ci_lower','ci_upper','p_value']}})
 return rows

def report(objects,primary,validation,unchanged,elapsed):
 primary_map={p['cohort']:p for p in primary['primary_estimand_rows']}
 lines=['# Healthcare-4603924：财富测量 Phase 2 执行报告','',f'运行结束：{now()}。状态：模型运行及初步输出完整性检查完成；Phase 3 QA待作者审阅。','',f"实际尝试14个LMM拟合单元（各BFGS后接L-BFGS）；SUCCESS {sum(o['status']=='SUCCESS' for o in objects)}项，失败/不可估计 {sum(o['status'] in ['FAILED','NOT_ESTIMABLE'] for o in objects)}项。KLoSA WM04按合同跳过，直接引用冻结主结果。WM00已完成36行描述统计。未运行meta。",'', '| 规格 | 队列 | N（恢复/参照） | FI行 | 状态 | β恢复×时间 | SE | 95%CI | 相对主结果Δβ |','|---|---|---:|---:|---|---:|---:|---|---:|']
 for o in sorted(objects,key=lambda o:(o['analysis_id'],['HRS','ELSA','KLoSA'].index(o['cohort']))):
  s=o['sample'];e=o['estimand'];p=primary_map[o['cohort']]
  if e:values=f"{e['estimate']:.8f} | {e['se']:.8f} | [{e['ci_lower']:.8f}, {e['ci_upper']:.8f}] | {e['estimate']-p['estimate']:+.8f}"
  else:values='— | — | — | —'
  lines.append(f"| {o['analysis_id']} | {o['cohort']} | {s['participants']}（{s['recovery_n']}/{s['comparison_n']}） | {s['fi_observations']} | {o['status']} | {values} |")
 lines+=['','失败单元若有数值，仅表示未验收候选估计，不能用作正式稳健性证据。','', '## 与冻结主结果的初步比较','']
 for c in ['HRS','ELSA','KLoSA']:
  p=primary_map[c];selected=[o for o in objects if o['cohort']==c and o['estimand']]
  text='；'.join(f"{o['analysis_id']} β={o['estimand']['estimate']:.6f}，SE={o['estimand']['se']:.6f}，N={o['sample']['participants']}，{o['status']}" for o in selected)
  lines.append(f"- {c}：冻结Primary β={p['estimate']:.6f}，SE={p['se']:.6f}，N={p['participants']}。{text}。")
 lines+=['','WM01、WM02、WM04、WM05改变参与者分母；与Primary的差异同时可能反映人群组成与调整后估计变化，不能解释为同一样本的纯规格效应。WM03保持原主要人群并仅改变损失变量及其时间交互的函数形式。WM05保持压力测试地位。WM02阈值是经验性分布限制，不能消除全部低分母情况。','', '## 诊断与Phase 3重点','', '各单元保存原始优化器警告、收敛状态、主参数与loglik差异、设计矩阵奇异值、Hessian和固定效应协方差及其谱、随机截距/残差方差、SE相对Primary的比值。高条件数本身不构成失败。','', '极端损失诊断提供固定设计杠杆份额和条件残差描述代理，未运行删除个体或leave-one-out模型；不能据此宣称已完成正式个体影响分析。残差子组不足5人时不发布残差摘要；内部总体表小单元外发前需另做披露核查。','', 'Phase 3重点核查HRS损失函数形式及压力测试、ELSA WM01的306人（88/218）与高条件数、KLoSA WM05的801人（282/519）、协方差谱和boundary警告。所有拟合失败必须保持隔离，不得静默改规格。','', '保存参数专用NPZ及完整系数/协方差/Hessian CSV/JSON，未保存包含微观记录的完整MixedLMResults对象。NPZ可作为同输入、同环境再利用的参数起点，不能单独恢复完整拟合对象。','', '## 完整性与执行边界','', f"初步数值/文件完整性：{validation['status']}；受保护资产前后哈希完全一致：{unchanged}；运行秒数：{elapsed:.1f}。",'', '合同仍为LOCKED，Phase 0仍为VALIDATED；本次仅记录新执行授权及目录布局，不修改历史合同/运行记录。原始DTA、所有冻结面板、历史模型及meta、原稿、Figure和S6未改。未创建结果freeze，未commit或push。','', '输出manifest和资产索引用仓库相对路径记录合同、九份数据/冻结记录、实际代码与结果哈希。Phase 2停止于初步完整性检查；尚未完成Phase 3科学QA，也未进入Phase 4冻结。']
 failures=[o for o in objects if o['failure_reasons']]
 if failures:
  lines+=['','## 失败单元及最小下一步','']
  for o in failures:lines.append(f"- {o['analysis_id']} {o['cohort']}：{'；'.join(o['failure_reasons'])}。保留当前输入、规格与失败证据；Phase 3先核查该单元的优化日志及数值诊断，进一步运行需另行授权。")
 (PHASE/'phase2_execution_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
 began=time.perf_counter();contract,inputs=verify_benchmark()
 resume='--resume-serialization-interruption' in sys.argv
 require(resume or (not OUTPUT.exists() and not PHASE.exists()),'输出目录已经存在，禁止覆盖；先审阅既有内容')
 print('BENCHMARK PASS; '+('resuming preserved attempt1' if resume else 'creating new Phase2 artifacts'),flush=True)
 current=protected_snapshot();head=git('rev-parse','HEAD');tracked=git('status','--porcelain','--untracked-files=no');status=git('status','--porcelain','--untracked-files=all')
 if resume:
  require(not (OUTPUT/'manifest.json').exists(),'已完成运行禁止续跑')
  interruption=read_json(PHASE/'execution_interruption_attempt1.json')
  for a in interruption['completed_assets']:require(sha(ROOT/a['path'])==a['sha256'],'既有成功资产变化')
  before=read_json(PHASE/'protected_hashes_before.json');require(before==current,'中断后受保护资产变化')
  authorization=read_json(PHASE/'execution_authorization.json');require(head==authorization['head'],'HEAD变化')
  require(git('diff','--no-ext-diff')+'\n'==(PHASE/'git_diff_before.patch').read_text(encoding='utf-8'),'既有tracked diff变化')
 else:
  before=current;PHASE.mkdir(parents=True);OUTPUT.mkdir(parents=True)
 test=subprocess.run([sys.executable,str(CODE/'tests/test_phase2.py')],cwd=ROOT,encoding='utf-8',capture_output=True)
 (PHASE/('synthetic_tests_attempt2.log' if resume else 'synthetic_tests.log')).write_text(test.stdout+test.stderr,encoding='utf-8')
 require(test.returncode==0,'合成检查失败，禁止真实拟合')
 if not resume:
  write_json(PHASE/'protected_hashes_before.json',before)
  (PHASE/'git_status_before.txt').write_text(status+'\n',encoding='utf-8')
  (PHASE/'git_diff_before.patch').write_text(git('diff','--no-ext-diff')+'\n',encoding='utf-8')
  authorization={'approved_at_source':'2026-10-13 user attached Phase2 request','attachment_sha256':sha(APPROVAL),'contract_sha256':EXPECTED_CONTRACT_SHA,'scope':'WM00 plus 14 locked LMM units, no meta','directory_override':'本轮作者指定代码02_code/03_results/section_3_3/wealth_measurement和结果03_output/3.3 Robustness of the Main Findings/Wealth Measurement；过程Phase2；覆盖早期合同目录边界，科学定义和历史合同保持原样','model_result_freeze':'NOT_AUTHORIZED','git_commit_push':'NOT_AUTHORIZED','run_started':now(),'head':head}
  write_json(PHASE/'execution_authorization.json',authorization)
 primary_path=PRIMARY/'Primary Models/primary_lmm_result_object.json';primary=read_json(primary_path)
 primary_map={p['cohort']:p for p in primary['primary_estimand_rows']}
 objects=[];diagnostics=[];descriptives=[];index=[];execution_units=[]
 for c in ['HRS','ELSA','KLoSA']:
  print('LOAD '+c,flush=True);p,person=load_cohort(contract,c);descriptives+=summarize(contract,c,person)
  for aid in ['WM01','WM02','WM03','WM04','WM05']:
   print('START '+aid+' '+c+' '+now(),flush=True)
   folder=OUTPUT/'Model Results'/aid/c
   try:
    frame,selected,anchor,counts=prepare_spec(contract,c,aid,p,person)
    x,design=design_matrix(contract,frame,anchor)
    if resume and (folder/'result_object.json').exists():
     obj=read_json(folder/'result_object.json');diag=read_json(folder/'diagnostics.json')
     require(obj['status']=='SUCCESS' and obj['sample']==counts,'续跑既有单元状态或锚点不符')
     execution_units.append({'analysis_id':aid,'cohort':c,'execution_attempt':1,'reused_without_refit':True})
    elif aid=='WM04' and c=='KLoSA':
     folder.mkdir(parents=True);obj={'cohort':c,'analysis_id':aid,'sample':counts,'status':'SKIP_IDENTICAL_TO_PRIMARY','source_kind':'FROZEN_PRIMARY_REFERENCE_NO_FIT','estimand':primary_map[c],'failure_reasons':[],'reference_path':rel(primary_path),'reference_sha256':sha(primary_path)}
     diag={'cohort':c,'analysis_id':aid,'sample':counts,'status':obj['status'],'design':design,'optimization':[],'reason':'divisor fixed at 1; same participants, rows, covariates as frozen primary'}
     for filename,data in [('result_object.json',obj),('diagnostics.json',diag),('sample_anchor.json',counts),('design_matrix.json',design),('optimization_log.json',[])]:write_json(folder/filename,data)
    else:
     obj,diag=fit_unit(contract,c,aid,frame,x,design,counts,folder,primary_map[c]['se'])
     execution_units.append({'analysis_id':aid,'cohort':c,'execution_attempt':2 if resume else 1,'reused_without_refit':False})
   except Exception as exc:
    folder.mkdir(parents=True,exist_ok=True)
    anchor=next(a for a in contract['validated_sample_and_design_anchors'] if a['cohort']==c and a['analysis_id']==aid)
    counts={k:anchor[k] for k in ['participants','fi_observations','recovery_n','comparison_n']}
    message=type(exc).__name__+': '+str(exc);state='NOT_ESTIMABLE' if 'NOT_ESTIMABLE:' in message else 'FAILED'
    obj={'cohort':c,'analysis_id':aid,'sample':counts,'sample_counts_source':'EXPECTED_ANCHOR; preparation failed, not actual certified counts','status':state,'source_kind':'PREPARATION_FAILED','estimand':None,'failure_reasons':[message]}
    diag={'cohort':c,'analysis_id':aid,'status':state,'failure_reasons':[message]}
    for filename,data in [('result_object.json',obj),('diagnostics.json',diag),('sample_anchor.json',obj),('design_matrix.json',{'status':'PREPARATION_FAILED'}),('optimization_log.json',[])]:write_json(folder/filename,data)
    (folder/'failure_trace.txt').write_text(traceback.format_exc(),encoding='utf-8')
   objects.append(obj);diagnostics.append(diag)
   print('DONE '+aid+' '+c+' '+obj['status']+' '+str(obj['failure_reasons']),flush=True)
   write_json(PHASE/'execution_checkpoint.json',{'completed_units':len(objects),'objects':objects,'updated':now()})
  del p,person
 desc=OUTPUT/'Descriptives';desc.mkdir()
 pd.DataFrame(descriptives).to_csv(desc/'WM00_descriptives.csv',index=False,encoding='utf-8');write_json(desc/'WM00_descriptives.json',{'status':'COMPLETE','rows':descriptives,'quantile_method':'linear','counting_unit':'unique primary participant','amount_comparison_between_cohorts':False})
 estimates=reference_rows(primary,contract)
 for o in objects:
  row={'analysis_id':o['analysis_id'],'cohort':o['cohort'],'status':o['status'],'source_kind':o['source_kind'],**o['sample']}
  row.update({k:o['estimand'].get(k) if o['estimand'] else None for k in ['estimate','se','ci_lower','ci_upper','p_value']});estimates.append(row)
 pd.DataFrame(estimates).to_csv(OUTPUT/'cohort_estimates.csv',index=False,encoding='utf-8')
 write_json(OUTPUT/'result_object.json',{'schema':'wealth_measurement_phase2_v1','status':'UNFROZEN_PHASE3_QA_PENDING','contract_sha256':EXPECTED_CONTRACT_SHA,'frozen_primary_reference':reference_rows(primary,contract),'models':objects})
 write_json(OUTPUT/'model_diagnostics.json',{'status':'PHASE3_QA_PENDING','models':diagnostics})
 write_json(PHASE/'execution_units.json',{'units':execution_units,'fit_units':len(execution_units),'resumed':resume,'reason':'进度JSON写入异常；仅修正序列化及续跑入口，科学构造和拟合代码未改变' if resume else None,'attempt1_code_record':rel(PHASE/'execution_interruption_attempt1.json') if resume else None})
 code=[{'path':rel(f),'sha256':sha(f)} for f in sorted(CODE.rglob('*')) if f.is_file() and '__pycache__' not in f.parts]
 code_hash=sha(CODE/'run_models.py')
 for c in ['HRS','ELSA','KLoSA']:
  panel=next(a for a in inputs if a['component']=='analysis_panel' and a['cohort']==c)
  common={'cohort':c,'contract_sha256':EXPECTED_CONTRACT_SHA,'panel_freeze_sha256':panel['freeze_sha256'],'panel_data_sha256':panel['sha256'],'source_code_sha256':code_hash,'validation_status':'PRELIMINARY_INTEGRITY_PENDING_PHASE3_QA'}
  index.append({**common,'analysis_id':'WM00','input_population_n':primary_map[c]['participants'],'model_fi_rows':0,'result_object_path':rel(desc/'WM00_descriptives.json'),'diagnostics_path':'','execution_status':'COMPLETE_DESCRIPTIVE'})
  for o in [o for o in objects if o['cohort']==c]:
   folder=OUTPUT/'Model Results'/o['analysis_id']/c
   index.append({**common,'analysis_id':o['analysis_id'],'input_population_n':o['sample']['participants'],'model_fi_rows':o['sample']['fi_observations'],'result_object_path':rel(folder/'result_object.json'),'diagnostics_path':rel(folder/'diagnostics.json'),'execution_status':o['status']})
 pd.DataFrame(index).to_csv(OUTPUT/'analysis_asset_index.csv',index=False,encoding='utf-8')
 validation=validate(contract,objects,diagnostics);write_json(PHASE/'preliminary_validation.json',validation)
 print('OUTPUT INTEGRITY '+validation['status']+'; checking protected assets',flush=True)
 after=protected_snapshot();write_json(PHASE/'protected_hashes_after.json',after)
 require(before==after,'受保护资产发生变化；立即停止并调查')
 require(git('rev-parse','HEAD')==head and git('status','--porcelain','--untracked-files=no')==tracked,'HEAD或既有tracked修改变化')
 (PHASE/'git_status_after.txt').write_text(git('status','--porcelain','--untracked-files=all')+'\n',encoding='utf-8')
 write_json(PHASE/'boundary_validation.json',{'status':'PASS','protected_assets':len(before),'hashes_unchanged':True,'head_unchanged':True,'tracked_changes_unchanged':True,'real_fit_units':14,'new_meta_runs':0,'freeze_created':False})
 report(objects,primary,validation,True,time.perf_counter()-began)
 outputs=[{'path':rel(f),'sha256':sha(f)} for root in [OUTPUT,PHASE] for f in sorted(root.rglob('*')) if f.is_file() and f.name not in ['manifest.json','manifest_validation.json']]
 manifest={'schema':'wealth_measurement_phase2_execution_v1','status':'COMPLETE_UNFROZEN' if validation['successful_fits']==14 else 'COMPLETE_WITH_FAILURES_UNFROZEN','started':authorization['run_started'],'ended':now(),'elapsed_seconds':time.perf_counter()-began,'git_head':head,'contract_sha256':EXPECTED_CONTRACT_SHA,'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'patsy':patsy.__version__,'statsmodels':statsmodels.__version__},'inputs':inputs,'historical_and_current_source_lineage':contract['historical_and_current_source_lineage'],'code':code,'outputs':outputs,'preliminary_validation':validation,'phase3_QA':'PENDING','full_model_pickle_saved':False,'new_meta_runs':0,'approval':authorization}
 write_json(OUTPUT/'manifest.json',manifest);write_json(PHASE/'manifest_validation.json',validate_manifest(manifest))
 print('COMPLETE '+manifest['status'],flush=True)

if __name__=='__main__':main()
