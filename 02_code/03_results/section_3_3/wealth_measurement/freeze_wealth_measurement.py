"""仅发布wealth_measurement_v1；不运行模型、meta或稿件生成流程。"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import ast
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from validate_freeze import validate_release,sha,read,require,RELEASE_ID,FROZEN_STATUS,CONTRACT_SHA,CAVEAT_MODELS

CODE=Path(__file__).resolve().parent;ROOT=CODE.parents[3]
ROUND1=ROOT/'06_project_progress/round1_extra_analyses_v1'
PHASE=ROUND1/'Phase4';P2=ROUND1/'Phase2';P3=ROUND1/'Phase3'
WORK=ROOT/'03_output/3.3 Robustness of the Main Findings/Wealth Measurement'
PARENT=ROOT/'03_output/3.3 Robustness of the Main Findings/releases';FINAL=PARENT/RELEASE_ID
PRIMARY=ROOT/'03_output/3.2 Wealth Recovery and Subsequent Frailty Progression'
NEW_CODE={CODE/n for n in ['freeze_wealth_measurement.py','validate_freeze.py','test_freeze.py']}

def rel(p):return Path(p).resolve().relative_to(ROOT.resolve()).as_posix()
def now():
 utc=datetime.now(timezone.utc);return {'local':utc.astimezone(timezone(timedelta(hours=8))).isoformat(),'utc':utc.isoformat(),'timezone':'Asia/Shanghai'}
def save(path,data):
 with Path(path).open('x',encoding='utf-8',newline='\n') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,encoding='utf-8')
def ensure_new_release(path):require(not Path(path).exists(),'目标目录已存在；禁止覆盖')
def event(message,**data):
 print(message,flush=True)
 with (PHASE/'execution_log.jsonl').open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps({'time':now(), 'event':message,**data},ensure_ascii=False)+'\n')
def snapshot():
 result={}
 for name in ['00_admin','01_data','02_code','03_output','04_collaboration','06_project_progress','07_manuscript']:
  for p in (ROOT/name).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and p not in NEW_CODE and not p.is_relative_to(PHASE) and not p.is_relative_to(FINAL) and not any(x.startswith('.wealth_measurement_v1_staging_') for x in p.parts):result[rel(p)]=sha(p)
 for p in ROOT.iterdir():
  if p.is_file():result[rel(p)]=sha(p)
 for c,v in read(ROOT/'06_project_progress/foundation_audit_v1/audit_manifest.json')['cohorts'].items():result['external_raw/'+c]=sha(v['source']['path'])
 return result

def source_preflight():
 contract_path=ROUND1/'Contracts/wealth_measurement_round1_v1_contract.json';require(sha(contract_path)==CONTRACT_SHA,'锁定合同SHA冲突');contract=read(contract_path);require(contract['status']=='LOCKED','合同非LOCKED')
 approval=read(P3/'author_decision.json');am=read(P3/'author_decision_manifest.json');qa=read(P3/'qa_validation.json');qm=read(P3/'qa_manifest.json');p2=read(WORK/'manifest.json')
 require(approval['phase3_author_decision']=='APPROVED_WITH_CAVEATS' and approval['unresolved_blocking_qa_items']==0 and approval['author_scientific_interpretation_accepted'],'作者批准不符')
 require(approval['phase4_authorization']=='NOT_YET_GRANTED' and approval['phase4_freeze_status']=='NOT_STARTED','历史批准边界已变化')
 require(qa['recommended_disposition']==qm['recommended_disposition']=='QA_APPROVED_WITH_CAVEATS' and not qa['issues'],'QA建议或阻断项不符')
 require({s:sum(d['status']==s for d in qa['model_qa']) for s in ['PASS','PASS_WITH_CAVEAT','HOLD']}=={'PASS':11,'PASS_WITH_CAVEAT':3,'HOLD':0},'QA计数不符')
 for item in am['source_evidence']+am['approval_outputs']+p2['outputs']+p2['code']+qm['outputs']+[qm['QA_code'],qm['QA_code_at_numeric_execution']]:require(sha(ROOT/item['path'])==item['sha256'],'来源哈希冲突：'+item['path'])
 require(sha(P3/'qa_manifest.json')==read(P3/'delivery_validation.json')['QA_manifest_sha256'],'QA manifest关系冲突')
 require(sha(WORK/'manifest.json')==read(P2/'manifest_validation.json')['manifest_sha256'],'Phase2 manifest关系冲突')
 for name,key in [('manifest.json','manifest_sha256'),('preflight_results.json','results_sha256')]:require(sha(ROUND1/'Preflight'/name)==contract['preflight'][key],'Preflight来源冲突')
 for a in contract['source_freezes']:
  require(sha(ROOT/a['path'])==a['sha256'] and sha(ROOT/a['freeze_path'])==a['freeze_sha256'],'冻结输入哈希冲突')
 history=read(P2/'execution_interruption_attempt1.json')
 for item in history['completed_assets']:require(sha(ROOT/item['path'])==item['sha256'],'首次9模型文件变化')
 for item in history['code']:require(sha(ROOT/item['backup_path'])==item['sha256'],'首次源码快照变化')
 for name in ['run_models.py','build_descriptives.py']:require(sha(P2/'source_versions/attempt1'/name)==sha(CODE/name),'科学模块变化')
 def funcs(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
 old=funcs(P2/'source_versions/attempt1/prepare_inputs.py');current=funcs(CODE/'prepare_inputs.py')
 for name in ['load_cohort','prepare_spec','design_matrix','verify_benchmark']:require(old[name]==current[name],'科学函数AST变化')
 primary_manifest=read(PRIMARY/'section_3_2_manifest.json');primary_freeze=read(PRIMARY/'section_3_2_freeze.json')
 require(primary_freeze['freeze_version']=='section_3_2_v3' and primary_freeze['manifest_sha256']==sha(PRIMARY/'section_3_2_manifest.json'),'Primary冻结链不符')
 require(primary_manifest['v2_active_asset_hashes']['Primary Models/primary_lmm_result_object.json']==sha(PRIMARY/'Primary Models/primary_lmm_result_object.json'),'Primary结果哈希不符')
 return contract,p2,qa,qm,approval,am,history

def main():
 if FINAL.exists():
  print(json.dumps({'operation':'VERIFY_EXISTING_ONLY','validation':validate_release(FINAL,ROOT,'full')},ensure_ascii=False,indent=2));return
 require(not PHASE.exists(),'Phase4过程目录已存在，禁止覆盖；先核对既有执行证据')
 contract,p2,qa,qm,approval,am,history=source_preflight();before=snapshot()
 for path,h in am['preexisting_assets_sha256_after'].items():require(before.get(path)==h,'批准后已有资产变化：'+path)
 head=git('rev-parse','HEAD').strip();tracked=git('status','--porcelain','--untracked-files=no');diff=git('diff','--no-ext-diff')
 PHASE.mkdir();save(PHASE/'protected_hashes_before.json',before)
 (PHASE/'git_status_before.txt').write_text(git('status','--porcelain','--untracked-files=all'),encoding='utf-8');(PHASE/'git_diff_before.patch').write_text(diff,encoding='utf-8')
 authorization={'schema':'wealth_measurement_phase4_authorization_v1','release_id':RELEASE_ID,'phase4_authorization':'GRANTED_FOR_WEALTH_MEASUREMENT_V1','recorded_at':now(),'basis':'本轮作者明确要求PLEASE IMPLEMENT Phase 4执行规格，并在前一条Final Approved Execution Instructions授权本模块P4.1–P4.4','scope':'仅正式发布、独立验证与冻结wealth_measurement_v1；不拟合、不meta、不制表、不修改稿件、不commit/push','phase3_author_decision_path':rel(P3/'author_decision.json'),'phase3_author_decision_sha256':sha(P3/'author_decision.json'),'historic_author_record_rewritten':False}
 save(PHASE/'phase4_authorization.json',authorization)
 event('P4.1 PRECONDITIONS PASS',HEAD=head,protected_assets=len(before))
 stage=PARENT/('.wealth_measurement_v1_staging_'+uuid.uuid4().hex);stage.mkdir()
 inventory=[];code_provenance=[];evidence_paths={};copied_sources=set()
 def copy(source,destination,role):
  source=Path(source);target=stage/destination;require(not target.exists(),'复制路径冲突')
  target.parent.mkdir(parents=True,exist_ok=True);expected=sha(source);shutil.copyfile(source,target);require(sha(target)==expected,'复制字节不一致')
  inventory.append({'release_path':destination,'sha256':expected,'bytes':target.stat().st_size,'role':role,'source_path':rel(source)});copied_sources.add(rel(source));return destination
 for item in p2['outputs']:
  source=ROOT/item['path']
  if source.is_relative_to(WORK):copy(source,source.relative_to(WORK).as_posix(),'NUMERIC_ASSET')
 require(sum(a['role']=='NUMERIC_ASSET' for a in inventory)==137,'数值复制不是137份')
 def evidence(source,destination,key=None):
  dest=copy(source,'Evidence/'+destination,'EVIDENCE_DOCUMENT')
  if key:evidence_paths[key]=dest
  return dest
 evidence(ROUND1/'Contracts/wealth_measurement_round1_v1_contract.json','Contracts/wealth_measurement_round1_v1_contract.json','contract')
 for name in ['manifest.json','preflight_results.json']:evidence(ROUND1/'Preflight'/name,'Preflight/'+name)
 evidence(WORK/'manifest.json','Phase2/original_working_manifest.json','phase2_manifest')
 for name,key in [('execution_units.json','execution_units'),('execution_interruption_attempt1.json','interruption'),('serialization_resume_code_diff.txt',None),('delivery_check.json',None),('manifest_validation.json',None)]:evidence(P2/name,'Phase2/'+name,key)
 for name,key in [('model_qa_matrix.csv','qa_matrix'),('qa_validation.json','qa_validation'),('qa_manifest.json','qa_manifest'),('author_decision.json','author_decision'),('author_decision_manifest.json','author_decision_manifest')]:evidence(P3/name,'Phase3/'+name,key)
 for path in sorted({a['freeze_path'] for a in contract['source_freezes']}):evidence(ROOT/path,'Input Freezes/'+path)
 for name in ['section_3_2_freeze.json','section_3_2_manifest.json','section_3_2_validation.json']:evidence(PRIMARY/name,'Primary Metadata/'+name)
 evidence(PHASE/'phase4_authorization.json','Publication/phase4_authorization.json','phase4_authorization')
 for name in ['pyproject.toml','uv.lock']:evidence(ROOT/name,'Environment/'+name)
 def code(source,destination,version):
  dest=copy(source,'Evidence/Code/'+destination,'EVIDENCE_CODE')
  code_provenance.append({'source_path':rel(source),'snapshot_path':dest,'sha256':sha(source),'version_role':version})
 for a in p2['code']:
  source=ROOT/a['path'];code(source,'phase2_actual/'+source.relative_to(CODE).as_posix(),'ACTUAL_PHASE2_FINAL_CODE')
 for a in history['code']:
  source=ROOT/a['backup_path'];code(source,'attempt1/'+source.relative_to(P2/'source_versions/attempt1').as_posix(),'FIRST_NINE_MODELS_ATTEMPT1')
 code(ROOT/qm['QA_code']['path'],'QA/validate_phase3_presentation.py','QA_PRESENTATION_SOURCE')
 code(ROOT/qm['QA_code_at_numeric_execution']['path'],'QA/validate_phase3_numeric.py','ACTUAL_NUMERIC_QA_SOURCE')
 code(ROOT/contract['historical_and_current_source_lineage']['historical_source_backup'],'Primary Historical/run_section_3_2.py','HISTORICAL_PRIMARY_PRE_KH_FIX')
 for name in ['freeze_wealth_measurement.py','validate_freeze.py','test_freeze.py']:code(CODE/name,'Phase4/'+name,'PHASE4_PUBLICATION_AND_VALIDATION')
 external={}
 def reference(path,h,role):
  if path not in copied_sources:
   if path in external:require(external[path]['sha256']==h,'来源SHA声明冲突');external[path]['roles'].append(role)
   else:external[path]={'path':path,'sha256':h,'roles':[role]}
 for a in p2['outputs']:reference(a['path'],a['sha256'],'PHASE2_UNCOPIED_PROCESS_EVIDENCE')
 for a in qm['outputs']:reference(a['path'],a['sha256'],'PHASE3_UNCOPIED_QA_EVIDENCE')
 for a in am['source_evidence']:reference(a['path'],a['sha256'],'AUTHOR_APPROVED_SOURCE_EVIDENCE')
 reference(rel(P3/'author_decision.md'),sha(P3/'author_decision.md'),'HUMAN_READABLE_AUTHOR_APPROVAL')
 primary_result=PRIMARY/'Primary Models/primary_lmm_result_object.json'
 reference(rel(primary_result),sha(primary_result),'HISTORICAL_PRIMARY_NUMERIC_SOURCE')
 lineage=contract['historical_and_current_source_lineage'];reference(lineage['current_repaired_source_path'],lineage['current_repaired_source_sha256'],'CURRENT_KH_REPAIRED_CODE_NOT_HISTORICAL_GENERATOR')
 qam={(d['cohort'],d['analysis_id']):d for d in qa['model_qa']};executions={(u['cohort'],u['analysis_id']):u for u in read(P2/'execution_units.json')['units']};bundle=read(stage/'result_object.json')
 index=[]
 def idx(aid,cohort,kind,result_path,qa_status,data_source,sample,diagnostics_path='',attempt=''):
  index.append({'analysis_id':aid,'cohort':cohort,'source_type':kind,'result_path':result_path,'result_sha256':sha(stage/result_path),'qa_status':qa_status,'data_source':data_source,'diagnostics_path':diagnostics_path,'diagnostics_sha256':sha(stage/diagnostics_path) if diagnostics_path else '', 'input_population_n':sample['participants'],'model_fi_rows':sample['fi_observations'],'execution_attempt':attempt,'contract_sha256':CONTRACT_SHA})
 for d in bundle['models']:
  cohort=d['cohort'];aid=d['analysis_id'];folder='Model Results/'+aid+'/'+cohort
  if d['status']=='SUCCESS':idx(aid,cohort,'NEW_FIT',folder+'/result_object.json',qam[(cohort,aid)]['status'],'Frozen Analysis/Exposure/Wealth Panels v1',d['sample'],folder+'/diagnostics.json',executions[(cohort,aid)]['execution_attempt'])
  else:idx(aid,cohort,'SKIP_IDENTICAL_TO_PRIMARY',folder+'/result_object.json','PASS_REFERENCE_ONLY','section_3_2_v3 frozen Primary',d['sample'],folder+'/diagnostics.json','NO_FIT')
 for d in bundle['frozen_primary_reference']:
  sample={k:d[k] for k in ['participants','fi_observations','recovery_n','comparison_n']};idx('PRIMARY',d['cohort'],'FROZEN_PRIMARY_REFERENCE','result_object.json','FROZEN_PRIMARY_REFERENCE_VERIFIED','section_3_2_v3 frozen Primary',sample,attempt='HISTORICAL')
  idx('WM00',d['cohort'],'DESCRIPTIVE','Descriptives/WM00_descriptives.json','PASS_DESCRIPTIVE','Original primary complete-case participants',{'participants':d['participants'],'fi_observations':0},attempt='DESCRIPTIVE_AFTER_RESUME')
 with (stage/'release_analysis_asset_index.csv').open('x',encoding='utf-8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(index[0]));writer.writeheader();writer.writerows(index)
 readme=f'''# wealth_measurement_v1

本地内部正式结果Release，发布状态由freeze.json的FROZEN_WITH_CAVEATS确定。

137份数值文件与已批准Phase 2原资产字节一致。原文件中的UNFROZEN/PENDING及工作目录路径保留为历史快照，不是当前发布状态；Phase 3作者批准的五项解释完整保存在Evidence/Phase3/author_decision.json。

## 正式读取规则

未来制表应锁定Release ID及freeze.json的SHA，先运行只读验证，再根据release_analysis_asset_index.csv读取本包内文件。该索引有18条模型结果（14 NEW_FIT、3 FROZEN_PRIMARY_REFERENCE、1 SKIP_IDENTICAL_TO_PRIMARY）及3条DESCRIPTIVE入口。保留的原analysis_asset_index.csv是工作阶段历史索引，正式制表不回退读取工作目录。

## 验证入口

使用uv run --locked python 02_code/03_results/section_3_3/wealth_measurement/validate_freeze.py --release-root "{rel(FINAL)}" --mode release-only，只验证包内完整性，不读取外部数据。--mode full额外核验九份冻结数据、历史Primary及明确列出的外部证据；首次发布必须Full PASS。

## 来源、限制与版本

11 PASS、3 PASS_WITH_CAVEAT、0 HOLD。HRS WM05、ELSA WM01、ELSA WM05仍为有条件通过，不升级为PASS。五项作者决定仅支持既定财富处理及限制人群下的观察性关联稳健性，不排除测量误差、回归均值、反向因果、未测混杂或正式个体影响；未完成事项不宣称已解决。

KLoSA WM04引用冻结Primary，未重复拟合，不构成新独立证据。第一次9个模型和续跑5个模型的代码、记录与文件身份分别保留。历史Primary由KH修复前源码生成，当前KH源码另行记录；没有修改历史freeze。

## 发布保护

拒绝覆盖同名目录，校验SHA和完整文件清单，正式发布后不再修改。Manifest覆盖所有内容资产；仅release_manifest.json、release_validation.json和freeze.json三个管理文件单独按单向哈希依赖验证。后续纠错必须另行批准新版本并记录supersedes，保留v1。Windows只读属性不是内容验收前提。

本包不包含微观数据、完整含参与者记录的拟合对象或新meta。内部汇总含小单元，公开分享前需要单独披露审查。Phase5尚未执行，无S7/S6或稿件修改，未commit/push。
'''
 (stage/'README.md').write_text(readme,encoding='utf-8')
 for path in ['release_analysis_asset_index.csv','README.md']:
  p=stage/path;inventory.append({'release_path':path,'sha256':sha(p),'bytes':p.stat().st_size,'role':'RELEASE_READING_INTERFACE','source_path':None})
 manifest={'schema':'wealth_measurement_release_manifest_v1','release_id':RELEASE_ID,'release_repo_path':rel(FINAL),'generated_at':now(),'git_HEAD':head,'git_dirty':bool(git('status','--porcelain')),'contract_sha256':CONTRACT_SHA,'source_assets':contract['source_freezes'],'code_provenance':code_provenance,'analysis_inventory':inventory,'numeric_asset_count':137,'evidence_asset_count':sum(a['role'].startswith('EVIDENCE_') for a in inventory),'evidence_paths':evidence_paths,'external_evidence_references':sorted(external.values(),key=lambda d:d['path']),'primary_reference':{'release_id':'section_3_2_v3','result_path':rel(primary_result),'result_sha256':sha(primary_result),'freeze_path':rel(PRIMARY/'section_3_2_freeze.json'),'freeze_sha256':sha(PRIMARY/'section_3_2_freeze.json'),'manifest_path':rel(PRIMARY/'section_3_2_manifest.json'),'manifest_sha256':sha(PRIMARY/'section_3_2_manifest.json'),'hash_authority':'section_3_2_v3.v2_active_asset_hashes'},'historical_and_current_source_lineage':lineage,'phase2_run_lineage':{'first_attempt_units':9,'resume_units':5,'execution_units_snapshot':evidence_paths['execution_units'],'interruption_snapshot':evidence_paths['interruption']},'qa_evidence':{'recommendation':'QA_APPROVED_WITH_CAVEATS','author_decision':'APPROVED_WITH_CAVEATS','qa_pass':11,'qa_pass_with_caveat':3,'qa_hold':0},'caveats':approval['accepted_scientific_decisions'],'model_fit_calls':0,'meta_executed':False,'management_file_exceptions':['release_manifest.json','release_validation.json','freeze.json']}
 save(stage/'release_manifest.json',manifest)
 event('P4.2 COPIES COMPLETE',numeric_assets=137,evidence_files=manifest['evidence_asset_count'],external_evidence_references=len(external))
 validation=validate_release(stage,ROOT,'full',payload_only=True);validation['validated_at']=now();save(stage/'release_validation.json',validation);save(PHASE/'staged_payload_full_validation.json',validation)
 freeze={'release_id':RELEASE_ID,'status':FROZEN_STATUS,'frozen_at':now(),'git_HEAD':head,'git_dirty':True,'phase3_author_decision':'APPROVED_WITH_CAVEATS','phase4_authorization':'GRANTED_FOR_WEALTH_MEASUREMENT_V1','fitted_lmm_units':14,'qa_pass':11,'qa_pass_with_caveat':3,'qa_hold':0,'klosa_wm04':'SKIP_IDENTICAL_TO_PRIMARY','meta_executed':False,'release_manifest_sha256':sha(stage/'release_manifest.json'),'release_validation_sha256':sha(stage/'release_validation.json'),'validation':{'status':'PASS','mode':'full','report_path':'release_validation.json'},'caveat_models':[{'cohort':c,'analysis_id':a,'status':'PASS_WITH_CAVEAT'} for c,a in sorted(CAVEAT_MODELS)],'next_phase':'READY_FOR_MANUSCRIPT_INTEGRATION','phase5_executed':False}
 save(stage/'freeze.json',freeze)
 save(PHASE/'staged_freeze_full_validation.json',validate_release(stage,ROOT,'full'))
 event('P4.3 STAGED FREEZE FULL PASS')
 test=subprocess.run([sys.executable,str(CODE/'test_freeze.py'),'--fixture',str(stage)],cwd=ROOT,encoding='utf-8',capture_output=True)
 (PHASE/'synthetic_tests.log').write_text(test.stdout+test.stderr,encoding='utf-8');require(test.returncode==0,'发布测试失败，未发布正式目录')
 require(snapshot()==before,'发布前既有资产变化')
 ensure_new_release(FINAL);os.rename(stage,FINAL)
 event('ATOMIC PUBLICATION COMPLETE',release_path=rel(FINAL))
 # 正式目录此后只读；所有收尾报告写在外部Phase4目录。
 final_full=validate_release(FINAL,ROOT,'full');final_local=validate_release(FINAL,None,'release-only')
 save(PHASE/'freeze_validation.json',{'status':'PASS','full':final_full,'release_only':final_local,'validated_at':now(),'phase4_status':FROZEN_STATUS,'phase5_status':'READY_FOR_MANUSCRIPT_INTEGRATION','phase5_executed':False})
 with (PHASE/'release_inventory.csv').open('x',encoding='utf-8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['release_path','role','bytes','sha256','source_path']);writer.writeheader();writer.writerows(inventory)
 after=snapshot();save(PHASE/'protected_hashes_after.json',after);require(after==before,'发布后既有资产变化')
 require(git('rev-parse','HEAD').strip()==head and git('status','--porcelain','--untracked-files=no')==tracked and git('diff','--no-ext-diff')==diff,'HEAD或已有tracked修改变化')
 (PHASE/'git_status_after.txt').write_text(git('status','--porcelain','--untracked-files=all'),encoding='utf-8')
 boundary={'status':'PASS','protected_assets':len(before),'all_existing_hashes_unchanged':True,'HEAD_unchanged':True,'tracked_diff_unchanged':True,'new_model_fits':0,'new_meta_runs':0,'manuscript_changed':False,'git_commit_push':False};save(PHASE/'boundary_validation.json',boundary)
 report=f'''# Healthcare-4603924：Phase 4正式发布与冻结

Release ID：{RELEASE_ID}；正式目录：{rel(FINAL)}。

最终状态：**FROZEN_WITH_CAVEATS**。Phase5状态：**READY_FOR_MANUSCRIPT_INTEGRATION**；尚未执行。

实际完整复制137份数值资产、{manifest['evidence_asset_count']}份Evidence快照，新增README及Release读取索引；三份管理文件按单向哈希关系验证。其他{len(external)}份证据按仓库相对路径、SHA和角色引用，原文件未删除。Evidence数量包括实际源码、历史及续跑版本、锁定环境、QA和作者批准，并非重新复制全部过程目录。

Manifest SHA-256：{sha(FINAL/'release_manifest.json')}。

Freeze SHA-256：{sha(FINAL/'freeze.json')}。

## 验证结果

暂存Full验证、含Freeze暂存验证、8项发布边界测试、正式目录release-only和Full验证均PASS。release-only仅证明内部完整性；Full另核验九份冻结数据、历史Primary及外部证据。137份数值副本与批准源文件字节一致；WM00共36条，18条模型来源为14个新拟合、3个冻结Primary及1个KLoSA WM04引用，另有3条描述入口。

11 PASS、3 PASS_WITH_CAVEAT、0 HOLD。HRS WM05、ELSA WM01、ELSA WM05保留条件；作者批准的五项科学解释及限制完整保留。第一次9个模型与续跑5个模型的来源未合并或抹去。历史QA PENDING及工作结果UNFROZEN标签保留为历史快照，以新授权、批准和freeze解释正式状态。

## 保护与停止点

{len(before)}项既有资产前后哈希一致，包括冻结输入、历史Primary及meta、合同、QA和稿件。HEAD保持{head}；既有KH源码未提交修改保持不变。本轮没有拟合、meta、S7/S6或稿件生成，没有commit/push。未设置Windows只读属性；通过拒绝覆盖、完整SHA清单和原子版本目录发布保护内容。

未来制表需指定Release ID及上述Freeze SHA，先验证再读取release_analysis_asset_index.csv，不回退工作目录。后续纠错另行批准新版本并保留v1。完成后停止，等待Phase5授权。
'''
 (PHASE/'phase4_freeze_report.md').write_text(report,encoding='utf-8')
 event('P4.4 FINAL FULL AND RELEASE-ONLY PASS',freeze_sha256=sha(FINAL/'freeze.json'))
 process_outputs=[{'path':rel(p),'sha256':sha(p)} for p in sorted(PHASE.rglob('*')) if p.is_file()]
 save(PHASE/'phase4_delivery_manifest.json',{'release_id':RELEASE_ID,'phase4_status':FROZEN_STATUS,'phase5_status':'READY_FOR_MANUSCRIPT_INTEGRATION','phase5_executed':False,'release_manifest_sha256':sha(FINAL/'release_manifest.json'),'freeze_sha256':sha(FINAL/'freeze.json'),'process_outputs':process_outputs,'publisher_sha256':sha(__file__),'validator_sha256':sha(CODE/'validate_freeze.py'),'git_HEAD':head})
 print(json.dumps({'release_id':RELEASE_ID,'release_path':rel(FINAL),'numeric_assets':137,'evidence_files':manifest['evidence_asset_count'],'release_manifest_sha256':sha(FINAL/'release_manifest.json'),'freeze_sha256':sha(FINAL/'freeze.json'),'status':FROZEN_STATUS,'next_phase':'READY_FOR_MANUSCRIPT_INTEGRATION','validation':'FULL AND RELEASE-ONLY PASS'},ensure_ascii=False,indent=2))

if __name__=='__main__':
 try:main()
 except Exception as exc:
  print('PUBLICATION STOPPED: '+type(exc).__name__+': '+str(exc),file=sys.stderr)
  if PHASE.exists() and not (PHASE/'publication_failure.json').exists():save(PHASE/'publication_failure.json',{'status':'STOPPED','reason':type(exc).__name__+': '+str(exc),'time':now(),'formal_release_exists':FINAL.exists(),'overwrite_performed':False})
  raise
