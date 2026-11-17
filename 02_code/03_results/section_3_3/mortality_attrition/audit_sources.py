"""Round1D只读来源核验；不调用面板构建或模型入口。"""
from pathlib import Path
import sys
import subprocess
import hashlib
import json
import numpy as np
import pandas as pd

CODE=Path(__file__).resolve().parent
ROOT=CODE.parents[3]
sys.path.insert(0,str(CODE.parent/'landmark_fi'))
from lf_common import read_json,write_json,sha,rel,require,assert_hashes,provenance
from lf_inputs import verify_sources as prior_sources
from validate_round1c_release import validate as validate_round1c

COHORTS=('HRS','ELSA','KLoSA')
FREEZES={'spine':'spine/spine_v2_freeze.json','candidate':'candidate_intervals/candidate_intervals_v3_freeze.json',
 'eligibility':'eligibility_base/eligibility_base_v2_freeze.json','exposure':'exposure_panel/exposure_panel_v1_freeze.json',
 'wealth':'wealth_panel/wealth_panel_v1_freeze.json','fi':'fi_panel/fi_panel_v2_freeze.json',
 'covariate':'covariate_panel/covariate_panel_v1_freeze.json','ledger':'analysis_cohort_ledger/analysis_cohort_ledger_v1_freeze.json',
 'final':'final_eligibility/final_eligibility_v1_freeze.json','analysis':'analysis_panel/analysis_panel_v1_freeze.json'}


def file_key(path):
    p=Path(path).resolve()
    return rel(p) if p.is_relative_to(ROOT) else str(p)


def pin(records,path,expected=None):
    p=Path(path);require(p.is_file(),'来源缺失：'+str(p));h=sha(p)
    require(expected is None or h==expected,'来源SHA不符：'+str(p))
    records[file_key(p)]=h
    return h


def verify_records(records):
    for name,h in records.items():
        p=ROOT/name;require(p.is_file() and sha(p)==h,'来源或受保护资产变化：'+name)


def snapshot():
    # 固定全部非缓存研究资产，微观数据另按冻结输入登记，不遍历环境或.git。
    records={}
    for name in ['00_admin','02_code','03_output','04_collaboration','05_literature','06_project_progress','07_manuscript']:
        base=ROOT/name
        if not base.exists():continue
        for p in base.rglob('*'):
            if p.is_file() and not any(s in p.parts for s in ['__pycache__','.pytest_cache','node_modules']):
                records[rel(p)]=sha(p)
    for name in ['AGENTS.md','README.md','uv.lock','pyproject.toml','renv.lock','.gitignore']:
        if (ROOT/name).is_file():records[name]=sha(ROOT/name)
    return records


def verify_sources():
    previous=prior_sources();records=previous['inputs'].copy();paths={}
    for kind,name in FREEZES.items():
        frpath=ROOT/'00_admin/panel_manifests'/name;pin(records,frpath)
        fr=read_json(frpath);require(fr['freeze_status']=='FROZEN','上游未冻结')
        paths[kind]={}
        for cohort,r in fr['cohorts'].items():
            p=ROOT/r.get('parquet_path',r.get('path'));h=r.get('parquet_sha256',r.get('sha256'))
            pin(records,p,h);paths[kind][cohort]=p
            mp=r.get('manifest_path',r.get('active_manifest_path',r.get('build_manifest_path')))
            mh=r.get('manifest_sha256',r.get('active_manifest_sha256',r.get('build_manifest_sha256')))
            if mp:pin(records,ROOT/mp,mh)
    release=ROOT/'03_output/3.3 Robustness of the Main Findings/releases/landmark_fi_round1c_v1'
    round1c=validate_round1c(release,'full')
    for p in release.rglob('*'):
        if p.is_file():pin(records,p)
    registry=pd.read_csv(ROOT/'00_admin/cohort_registry.csv',encoding='utf-8');pin(records,ROOT/'00_admin/cohort_registry.csv')
    books=pd.read_csv(ROOT/'00_admin/codebooks/codebook_manifest.csv',encoding='utf-8')
    bookpaths={};rawpaths={}
    spine_fr=read_json(ROOT/'00_admin/panel_manifests/spine/spine_v2_freeze.json')
    for row in registry.itertuples():
        require(str(row.version_pair_verified).upper()=='TRUE','数据与codebook未匹配')
        raw=Path(row.dataset_path);pin(records,raw,spine_fr['cohorts'][row.cohort]['source_dataset_sha256'])
        rawpaths[row.cohort]=raw
        b=books.loc[(books.cohort==row.cohort)&(books.usage_status=='authoritative')].iloc[0]
        book=ROOT/b.local_path;pin(records,book,b.sha256);bookpaths[row.cohort]=book
    for directory in ['02_code/01_panel_build','00_admin/variable_maps']:
        for p in (ROOT/directory).rglob('*'):
            if p.is_file() and p.suffix in ['.py','.R','.md','.csv']:pin(records,p)
    for name in ['00_admin/wave_maps.csv','00_admin/panel_governance.md','00_admin/codebooks/codebook_manifest.csv',
        '07_manuscript/Round1/Editor.md','07_manuscript/Round1/Reviewers.md','07_manuscript/Manuscript.docx',
        '07_manuscript/Round1/healthcare-4603924.docx','07_manuscript/Submit/healthcare-4603924-precheck/healthcare-4603924-precheck_revised.docx',
        '02_code/03_results/section_3_3/run_weighted_sensitivity.R','02_code/03_results/section_3_3/run_mi_imputation.R']:
        pin(records,ROOT/name)
    return {'records':records,'paths':paths,'rawpaths':rawpaths,'bookpaths':bookpaths,
            'prior_source_check':previous,'round1c_validation':round1c,'registry':registry}


def read_panel(sources,kind,cohort):
    p=sources['paths'][kind][cohort];frame=pd.read_parquet(p)
    frame['person_id']=frame.person_id.astype(str)
    keys=['cohort','person_id']+(['candidate_shock_wave'] if kind in ['candidate','eligibility','exposure'] else
                               [] if kind in ['ledger','final'] else ['wave'])
    require(not frame[keys].duplicated().any(),'上游键重复：'+kind)
    return frame


def codebook_text(path):
    result=subprocess.run(['pdftotext','-layout','-enc','UTF-8',str(path),'-'],capture_output=True,check=True)
    return result.stdout.decode('utf-8')


def raw_fields(sources,cohort):
    registry=sources['registry'];row=registry.loc[registry.cohort==cohort].iloc[0]
    path=sources['rawpaths'][cohort];idvar=row.person_id_source
    with pd.io.stata.StataReader(path,convert_dates=False,convert_categoricals=False) as reader:
        labels=reader.variable_labels()
    wanted=[idvar]+([v for v in ['radyear','radmonth','raddate','radsrc'] if v in labels])
    wanted+=[v for v in labels if v.startswith('r') and v.endswith('proxy') and v[1:-5].isdigit()]
    pieces=[]
    # 分块读取以避免宽HRS原始表的整表内存占用；特殊缺失码保留为对象。
    with pd.io.stata.StataReader(path,columns=wanted,convert_dates=False,convert_categoricals=False,convert_missing=True,chunksize=1024) as reader:
        for chunk in reader:
            pieces.append(chunk[wanted].copy())
    raw=pd.concat(pieces,ignore_index=True)
    raw['person_id']=raw[idvar].map(lambda x:str(int(x)))
    require(not raw.person_id.duplicated().any(),'原始个人ID重复')
    text=codebook_text(sources['bookpaths'][cohort]);lower=text.lower()
    if cohort=='HRS':require('last day of the month' in lower and 'radyear' in lower,'HRS死亡日构造证据缺失')
    if cohort=='ELSA':require('waves 6 to 10' in lower and 'mortality status is not currently provided' in lower,'ELSA覆盖证据缺失')
    if cohort=='KLoSA':require('exit interview' in lower and 'radmonth' in lower,'KLoSA死亡资料证据缺失')
    pages={}
    for variable in wanted[1:]:
        pages[variable]=[i+1 for i,page in enumerate(text.split('\f')) if variable.lower() in page.lower()][:8]
    inventory=[]
    for v in wanted[1:]:
        a=raw[v];tags=a.map(lambda x:getattr(x,'string',None));num=pd.to_numeric(a,errors='coerce')
        inventory.append({'cohort':cohort,'source_file':str(path),'source_sha256':sources['records'][str(path)],
          'variable':v,'description':labels[v],'valid_values_observed':json.dumps(sorted(num.dropna().unique().tolist()) if num.nunique()<=50 else
              {'min':float(num.min()),'max':float(num.max()),'unique_values':int(num.nunique())},ensure_ascii=False),
          'special_missing_tags':json.dumps(tags.dropna().value_counts().to_dict()),'n_source_participants':len(a),
          'missing_n':int(num.isna().sum()),'time_resolution':{'radyear':'year','radmonth':'month_component','raddate':'constructed_day_not_exact','radsrc':'not_a_time_field'}.get(v,'wave'),
          'record_confirmation_time':'NOT_AVAILABLE','coverage':'current dataset snapshot; no assumption of prospective availability',
          'reliability':'LIMITED_USE' if v=='raddate' else 'VERIFIED_FIELD_SEMANTICS_COVERAGE_LIMITED',
          'codebook_pdf_pages':json.dumps(pages[v]),'codebook_sha256':sources['records'][rel(sources['bookpaths'][cohort])],
          'reason':'HRS RADDATE uses constructed day; retain native year/month precision' if v=='raddate' else 'Use with wave status; no implicit live status from missing death date'})
    death=raw[['person_id']].copy()
    for v in ['radyear','radmonth']:
        death[v]=pd.to_numeric(raw[v],errors='coerce') if v in raw else np.nan
    death['death_month_index']=np.where(death.radyear.between(1900,2100)&death.radmonth.between(1,12),death.radyear*12+death.radmonth,np.nan)
    death['death_year_only']=death.radyear.between(1900,2100)&~death.radmonth.between(1,12)
    proxy=[]
    for v in wanted:
        if v.endswith('proxy'):
            wave=int(v[1:-5]);num=pd.to_numeric(raw[v],errors='coerce')
            proxy.append(pd.DataFrame({'person_id':raw.person_id,'wave':wave,'proxy_raw':num}))
    return death,pd.concat(proxy,ignore_index=True) if proxy else None,inventory
