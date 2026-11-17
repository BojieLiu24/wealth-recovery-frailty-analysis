"""wealth_measurement_v1只读验证；不拟合、不调用meta、不修改Release。"""
from pathlib import Path,PurePosixPath
import argparse
import csv
import hashlib
import json
import math

RELEASE_ID='wealth_measurement_v1'
FROZEN_STATUS='FROZEN_WITH_CAVEATS'
CONTRACT_SHA='a9d6769d479c80d6d0d886440e5f66ff6033eae5810959bb3c5eea18fdf86f4c'
MANAGEMENT={'release_manifest.json','release_validation.json','freeze.json'}
CAVEAT_MODELS={('HRS','WM05'),('ELSA','WM01'),('ELSA','WM05')}
REPO=Path(__file__).resolve().parents[4]

def require(value,message):
 if not bool(value):raise ValueError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):
 with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def safe(base,relative):
 require(isinstance(relative,str) and relative and '\\' not in relative and ':' not in relative,'非法路径字符串')
 p=PurePosixPath(relative)
 require(not p.is_absolute() and all(x not in ['.','..',''] for x in relative.split('/')),'非法相对路径')
 result=Path(base).joinpath(*p.parts)
 require(result.resolve().is_relative_to(Path(base).resolve()),'路径逃逸')
 require(not result.is_symlink(),'禁止符号链接')
 return result
def equal(a,b):return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12)
def same_estimand(a,b):
 for key in ['estimate','se','ci_lower','ci_upper','p_value']:require(equal(a[key],b[key]),'核心数值不一致：'+key)
def same_sample(a,b):
 for key in ['participants','fi_observations','recovery_n','comparison_n']:require(int(a[key])==int(b[key]),'样本锚点不一致：'+key)

def validate_release(release,repo=None,mode='release-only',payload_only=False):
 release=Path(release).resolve();require(mode in ['release-only','full'],'未知验证模式')
 manifest=read(release/'release_manifest.json')
 require(manifest['release_id']==RELEASE_ID and manifest['contract_sha256']==CONTRACT_SHA,'Release ID或合同不符')
 inventory=manifest['analysis_inventory'];declared=[a['release_path'] for a in inventory]
 require(len(declared)==len(set(declared)) and not MANAGEMENT.intersection(declared),'清单重复或管理文件进入内容哈希清单')
 actual=set()
 for p in release.rglob('*'):
  require(not p.is_symlink() and p.resolve().is_relative_to(release),'发现链接或路径逃逸')
  if p.is_file():actual.add(p.relative_to(release).as_posix())
 require(actual-set(declared)<=MANAGEMENT,'发现未列明文件：'+str(sorted(actual-set(declared)-MANAGEMENT)))
 require(set(declared)<=actual,'内容资产缺失')
 for entry in inventory:
  p=safe(release,entry['release_path']);require(sha(p)==entry['sha256'],'内容资产SHA不符：'+entry['release_path'])
  require(p.stat().st_size==entry['bytes'],'内容资产大小不符')
 numeric=[a for a in inventory if a['role']=='NUMERIC_ASSET'];evidence=[a for a in inventory if a['role'].startswith('EVIDENCE_')]
 require(len(numeric)==manifest['numeric_asset_count']==137,'数值资产数不是137')
 require(len(evidence)==manifest['evidence_asset_count'],'Evidence数不符')
 paths=manifest['evidence_paths']
 def ev(name):return read(safe(release,paths[name]))
 contract=ev('contract');approval=ev('author_decision');qav=ev('qa_validation')
 require(contract['status']=='LOCKED' and sha(safe(release,paths['contract']))==CONTRACT_SHA,'合同快照不符')
 require(approval['phase3_author_decision']=='APPROVED_WITH_CAVEATS' and approval['author_scientific_interpretation_accepted'] is True and approval['unresolved_blocking_qa_items']==0,'作者批准不完整')
 require(len(approval['accepted_scientific_decisions'])==5 and [d['id'] for d in approval['accepted_scientific_decisions']]==list('ABCDE') and all(d['accepted'] for d in approval['accepted_scientific_decisions']),'五项科学决定未完整保留')
 accepted=approval['accepted_scientific_decisions'][3]['models']
 require({(d['cohort'],d['analysis_id']) for d in accepted}==CAVEAT_MODELS and all(d['qa_status']=='PASS_WITH_CAVEAT' for d in accepted),'作者caveat丢失')
 require(approval['phase4_authorization']=='NOT_YET_GRANTED','历史Phase3批准边界被改写')
 auth=ev('phase4_authorization');require(auth['phase4_authorization']=='GRANTED_FOR_WEALTH_MEASUREMENT_V1' and auth['release_id']==RELEASE_ID,'本轮Phase4授权不符')
 require(qav['recommended_disposition']=='QA_APPROVED_WITH_CAVEATS' and not qav['issues'],'QA建议或阻断项不符')
 qamodels=qav['model_qa'];qa_map={(d['cohort'],d['analysis_id']):d for d in qamodels}
 qa_counts={s:sum(d['status']==s for d in qamodels) for s in ['PASS','PASS_WITH_CAVEAT','HOLD']}
 require(len(qa_map)==14 and qa_counts=={'PASS':11,'PASS_WITH_CAVEAT':3,'HOLD':0},'QA计数错误')
 require({key for key,d in qa_map.items() if d['status']=='PASS_WITH_CAVEAT'}==CAVEAT_MODELS,'三个caveat模型丢失')
 matrix=rows(safe(release,paths['qa_matrix']))
 require(len(matrix)==14 and {(d['cohort'],d['analysis_id']):d['qa_status'] for d in matrix}=={key:d['status'] for key,d in qa_map.items()},'QA矩阵错误')
 bundle=read(release/'result_object.json');models=bundle['models'];diag=read(release/'model_diagnostics.json')['models'];diagnostic_map={(d['cohort'],d['analysis_id']):d for d in diag}
 require(len(models)==15 and len(diagnostic_map)==15,'模型单元数不符')
 require(sum(d['status']=='SUCCESS' for d in models)==14,'新拟合计数不符')
 skip=[d for d in models if d['status']=='SKIP_IDENTICAL_TO_PRIMARY'];require(len(skip)==1 and skip[0]['cohort']=='KLoSA' and skip[0]['analysis_id']=='WM04','Primary引用误标为新拟合')
 estimates=rows(release/'cohort_estimates.csv');estimate_map={(d['cohort'],d['analysis_id']):d for d in estimates};require(len(estimates)==len(estimate_map)==18,'18条结果计数不符')
 primary_rows=bundle['frozen_primary_reference'];require(len(primary_rows)==3,'冻结Primary数量不符')
 primary_map={d['cohort']:d for d in primary_rows};require(set(primary_map)=={'HRS','ELSA','KLoSA'},'Primary队列错误')
 for p in primary_rows:
  row=estimate_map[(p['cohort'],'PRIMARY')];require(row['source_kind']=='FROZEN_PRIMARY' and row['status']=='REFERENCE_ONLY','Primary来源误标')
  same_estimand(row,p);same_sample(row,p)
 for d in models:
  key=(d['cohort'],d['analysis_id']);folder=release/'Model Results'/d['analysis_id']/d['cohort'];unit=read(folder/'result_object.json');require(unit==d,'模型单元和汇总不符')
  row=estimate_map[key];same_estimand(row,d['estimand']);same_sample(row,d['sample'])
  if d['status']=='SUCCESS':
   require(key in qa_map,'模型无QA');same_estimand(d['estimand'],qa_map[key]['estimand']);same_sample(d['sample'],qa_map[key]['sample'])
   require(diagnostic_map[key]==read(folder/'diagnostics.json'),'诊断单元和汇总不符')
   e=d['estimand'];require(equal(e['ci_lower'],e['estimate']-1.959963984540054*e['se']) and equal(e['ci_upper'],e['estimate']+1.959963984540054*e['se']),'Wald CI不符')
   require(equal(e['p_value'],math.erfc(abs(e['estimate']/e['se'])/math.sqrt(2))),'Wald P不符')
  else:
   same_estimand(d['estimand'],primary_map['KLoSA']);same_sample(d['sample'],primary_map['KLoSA'])
   require(read(folder/'optimization_log.json')==[] and not (folder/'model_parameters.npz').exists(),'KLoSA WM04存在新拟合资产')
 wm=read(release/'Descriptives/WM00_descriptives.json')['rows'];wmcsv=rows(release/'Descriptives/WM00_descriptives.csv')
 require(len(wm)==len(wmcsv)==36 and len({(d['cohort'],d['recovery_status'],d['variable']) for d in wm})==36,'WM00单元不完整')
 expected_fields={'wealth_tminus1','wealth_t','wealth_tplus1','absolute_loss','absolute_recovery','recovery_fraction_raw'}
 for c in ['HRS','ELSA','KLoSA']:
  for group in ['substantial_recovery','persistent_or_limited_recovery']:
   cells=[d for d in wm if d['cohort']==c and d['recovery_status']==group];require({d['variable'] for d in cells}==expected_fields,'WM00指标缺失')
   for cell in cells:
    expected_n=primary_map[c]['recovery_n' if group=='substantial_recovery' else 'comparison_n'];require(cell['n']==expected_n,'WM00人数不符')
    r=next(r for r in wmcsv if (r['cohort'],r['recovery_status'],r['variable'])==(c,group,cell['variable']))
    for name in ['n','median','q1','q3','negative_n','negative_percent','zero_n','zero_percent']:require(equal(r[name],cell[name]),'WM00 CSV/JSON不符')
 index=rows(release/'release_analysis_asset_index.csv');require(len(index)==21,'Release索引不是21条')
 source_counts={s:sum(r['source_type']==s for r in index) for s in ['NEW_FIT','FROZEN_PRIMARY_REFERENCE','SKIP_IDENTICAL_TO_PRIMARY','DESCRIPTIVE']}
 require(source_counts=={'NEW_FIT':14,'FROZEN_PRIMARY_REFERENCE':3,'SKIP_IDENTICAL_TO_PRIMARY':1,'DESCRIPTIVE':3},'Release索引来源计数错误')
 require(len({(r['cohort'],r['analysis_id']) for r in index})==21,'Release索引重复')
 for r in index:
  require(sha(safe(release,r['result_path']))==r['result_sha256'],'索引结果SHA错误')
  if r['diagnostics_path']:require(sha(safe(release,r['diagnostics_path']))==r['diagnostics_sha256'],'索引诊断SHA错误')
  if r['source_type']=='NEW_FIT':require(r['qa_status']==qa_map[(r['cohort'],r['analysis_id'])]['status'],'索引caveat状态错误')
  elif r['source_type']=='FROZEN_PRIMARY_REFERENCE':require(r['analysis_id']=='PRIMARY','Primary索引ID错误')
  elif r['source_type']=='SKIP_IDENTICAL_TO_PRIMARY':require((r['cohort'],r['analysis_id'])==('KLoSA','WM04'),'跳过索引ID错误')
  else:require(r['analysis_id']=='WM00','描述索引ID错误')
 execution=ev('execution_units')['units'];require(len(execution)==14 and sum(d['execution_attempt']==1 for d in execution)==9 and sum(d['execution_attempt']==2 for d in execution)==5,'中断9+5历史关系不符')
 interruption=ev('interruption');require(len(interruption['completed_assets'])==81,'中断前81文件记录不完整')
 by_source={d.get('source_path'):d for d in inventory if d.get('source_path')}
 for a in interruption['completed_assets']:require(a['path'] in by_source and by_source[a['path']]['sha256']==a['sha256'],'中断前模型文件未字节一致保存')
 for a in manifest['code_provenance']:
  require(sha(safe(release,a['snapshot_path']))==a['sha256'],'源码快照哈希不符')
 history=manifest['historical_and_current_source_lineage'];require(history['historical_source_sha256']!=history['current_repaired_source_sha256'],'历史运行与KH修复源码身份混淆')
 external_checked=0
 if mode=='full':
  require(repo is not None,'Full验证必须指定仓库根目录');repo=Path(repo).resolve();cache={}
  def external(path,h):
   nonlocal external_checked
   if path not in cache:cache[path]=sha(safe(repo,path));external_checked+=1
   require(cache[path]==h,'外部来源SHA不符：'+path)
  for a in inventory:
   if a.get('source_path'):external(a['source_path'],a['sha256'])
  for a in manifest['source_assets']:
   external(a['path'],a['sha256']);external(a['freeze_path'],a['freeze_sha256'])
  for a in manifest['external_evidence_references']:external(a['path'],a['sha256'])
  p=manifest['primary_reference'];external(p['result_path'],p['result_sha256']);external(p['freeze_path'],p['freeze_sha256']);external(p['manifest_path'],p['manifest_sha256'])
  pf=read(safe(repo,p['freeze_path']));pm=read(safe(repo,p['manifest_path']))
  require(pf['freeze_version']=='section_3_2_v3' and pf['manifest_sha256']==p['manifest_sha256'],'历史Primary冻结链错误')
  require(pm['v2_active_asset_hashes']['Primary Models/primary_lmm_result_object.json']==p['result_sha256'],'Primary数值来源哈希错误')
  actual_primary=read(safe(repo,p['result_path']))['primary_estimand_rows']
  for r in actual_primary:same_estimand(r,primary_map[r['cohort']])
 if not payload_only:
  require(MANAGEMENT<=actual,'特殊管理文件缺失')
  validation=read(release/'release_validation.json');freeze=read(release/'freeze.json')
  require(validation['status']=='PASS' and validation['mode']=='full' and validation['validation_scope']=='PRE_FREEZE_PAYLOAD','首次完整验证不符')
  require(validation['release_manifest_sha256']==sha(release/'release_manifest.json'),'验证报告与Manifest哈希关系错误')
  require(freeze['release_id']==RELEASE_ID and freeze['status']==FROZEN_STATUS,'正式Freeze状态错误')
  require(freeze['release_manifest_sha256']==sha(release/'release_manifest.json') and freeze['release_validation_sha256']==sha(release/'release_validation.json'),'Freeze哈希关系错误')
  require(freeze['phase3_author_decision']=='APPROVED_WITH_CAVEATS' and freeze['phase4_authorization']=='GRANTED_FOR_WEALTH_MEASUREMENT_V1','Freeze批准关系错误')
  require(freeze['fitted_lmm_units']==14 and freeze['qa_pass']==11 and freeze['qa_pass_with_caveat']==3 and freeze['qa_hold']==0 and freeze['klosa_wm04']=='SKIP_IDENTICAL_TO_PRIMARY' and freeze['meta_executed'] is False,'Freeze计数或边界错误')
 return {'status':'PASS','mode':mode,'validation_scope':'PRE_FREEZE_PAYLOAD' if payload_only else 'FROZEN_RELEASE','release_id':RELEASE_ID,'release_manifest_sha256':sha(release/'release_manifest.json'),'release_content_integrity':'PASS','external_source_validation':'PASS' if mode=='full' else 'NOT_CHECKED','external_files_checked':external_checked,'numeric_assets':len(numeric),'evidence_files':len(evidence),'content_assets':len(inventory),'WM00_records':36,'new_fit_units':14,'primary_reference_rows':3,'skip_reference_rows':1,'qa_counts':qa_counts,'source_counts':source_counts,'new_model_fits':0,'new_meta_runs':0,'freeze_sha256':sha(release/'freeze.json') if (release/'freeze.json').exists() else None}

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release-root',required=True);parser.add_argument('--repo-root',default=str(REPO));parser.add_argument('--mode',choices=['release-only','full'],default='release-only');args=parser.parse_args()
 print(json.dumps(validate_release(args.release_root,args.repo_root,args.mode),ensure_ascii=False,indent=2))
