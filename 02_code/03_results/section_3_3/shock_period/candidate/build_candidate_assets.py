"""Stage5只读指定Round1E freeze，生成英文候选表和时期支持摘要。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'release'))
from validate_release import validate,RELEASE_ID,require
from sp_contract import ROOT,read_json,write_json,sha,rel
from common import provenance
import csv
import json
import shutil
import subprocess
import pandas as pd

NOTES=[
    'FI, frailty index; SE, standard error; CI, confidence interval. The estimate is the substantial-recovery minus persistent-or-limited-recovery difference in annual FI change (FI units/year). Primary is the frozen original reference and was not refitted.',
    'SP01 retains the same participants and FI rows, the original 42-column adjustment basis and frozen standardization. It adds categorical shock-wave main effects and shock-wave by time interactions. Reference shock waves are HRS W3, ELSA W2 and KLoSA W2. Models are cohort-specific, unweighted maximum-likelihood random-intercept LMMs; time is actual years since the original landmark.',
    'SP01 assumes a common recovery-associated annual slope difference across shock waves. It does not estimate period-specific recovery associations or causal effects of the financial crisis. Differences from Primary are descriptive; no independent coefficient-difference test was performed.',
    'NE: HRS/SP01 was accepted with an author-approved numerical exception after independent likelihood, score, full observed-information and covariance validation. BFGS reported precision loss (status 2); L-BFGS converged by relative objective reduction. The original Stage2 HOLD and all history remain unchanged. Acceptance is not a double-convergence PASS or proof of global optimality.',
    'All SP01 results were accepted with caveats. CIs use the full LMM covariance, including variance-parameter uncertainty, and normal Wald inference. Conditional-residual skewness and heavy tails remain, particularly in KLoSA. Each latest shock wave has landmark plus only one later FI observation per participant; late-wave sensitivity is not fully excluded by fixed-covariance influence screening.',
    'HRS uncertainty under the frozen Round1B random-slope and nonlinear-time specifications remains relevant; SP01 does not resolve it. No combined models or meta-analysis were run. ELSA W10 actual FI observations from 2021-2023 are retained. This is a candidate for author review; the manuscript is unchanged.'
]


def require_release_path(path):
    path=Path(path).resolve()
    require(path==ROOT/'03_output/3.3 Robustness of the Main Findings/releases'/RELEASE_ID,'必须只读指定正式release，不允许工作目录回退')
    return path


def verify_six(rows):
    expected={(c,s) for c in ('HRS','ELSA','KLoSA') for s in ('Primary','SP01')}
    require(len(rows)==6 and {(r['cohort'],r['specification']) for r in rows}==expected,'必须六行唯一Primary/SP01')


def read_rows(release,freeze_sha):
    release=require_release_path(release)
    require(sha(release/'freeze.json')==freeze_sha,'指定freeze SHA不匹配')
    validate(release,'release-only')
    require(read_json(release/'Evidence/Approvals/author_result_and_freeze_approval.json')['stage5_candidate_table_authorized'],'候选资产未授权')
    with (release/'Comparisons/cohort_estimates.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    verify_six(rows)
    for r in rows:
        for k in ('estimate','se','ci_lower','ci_upper','p_value'):r[k]=float(r[k])
        for k in ('participants','FI_observations'):r[k]=int(r[k])
        r['QA_qualification']='Frozen reference' if r['specification']=='Primary' else ('With caveats (NE)' if r['cohort']=='HRS' else 'With caveats')
    return rows


def display(r):
    return [r['cohort'],r['specification'],f"{r['participants']:,}",f"{r['FI_observations']:,}",
        f"{r['estimate']:+.6f}",f"{r['se']:.6f}",f"[{r['ci_lower']:+.6f}, {r['ci_upper']:+.6f}]",r['QA_qualification']]


def text_new(path,text):
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(text)


def csv_new(path,rows):
    with path.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def support_rows(release):
    source=pd.read_csv(release/'Evidence/Stage0/Audit/time_support.csv',encoding='utf-8',float_precision='round_trip')
    primary=source.loc[source.population.eq('B_PRIMARY')&source.record_type.isna()]
    upstream=source.loc[source.population.eq('A_INDEX_SHOCK')&source.group.eq('ALL')]
    checked=pd.read_csv(release/'QA/wave_group_time_support.csv',encoding='utf-8',float_precision='round_trip')
    rows=[]
    for c in ('HRS','ELSA','KLoSA'):
        for w in sorted(primary.loc[primary.cohort.eq(c),'shock_wave'].unique()):
            allrow=primary.loc[primary.cohort.eq(c)&primary.shock_wave.eq(w)&primary.group.eq('ALL')].iloc[0]
            a=upstream.loc[upstream.cohort.eq(c)&upstream.shock_wave.eq(w)].iloc[0]
            groups={}
            for value,label in [(1,'substantial_recovery'),(0,'persistent_or_limited_recovery')]:
                r=primary.loc[primary.cohort.eq(c)&primary.shock_wave.eq(w)&primary.group.eq(label)].iloc[0]
                independent=checked.loc[checked.cohort.eq(c)&checked.wave.eq(w)&checked.recovery.eq(value)].iloc[0]
                require(int(r.participants)==int(independent.participants) and int(r.FI_rows)==int(independent.FI_rows)
                    and int(r.two_or_more_later_FI_n)==int(independent.at_least_two_later_FI),'Stage0与Stage3支持不一致')
                groups[value]=r
            require(int(groups[1].participants+groups[0].participants)==int(allrow.participants),'波次两组人数不守恒')
            rows.append({'cohort':c,'shock_wave':int(w),'actual_shock_interview_year_min':int(allrow.shock_interview_year_min),
                'actual_shock_interview_year_max':int(allrow.shock_interview_year_max),'first_shock_pool_participants':int(a.participants),
                'Primary_recovery_participants':int(groups[1].participants),'Primary_reference_participants':int(groups[0].participants),
                'Primary_recovery_FI_rows':int(groups[1].FI_rows),'Primary_reference_FI_rows':int(groups[0].FI_rows),
                'recovery_at_least_two_later_FI':int(groups[1].two_or_more_later_FI_n),
                'reference_at_least_two_later_FI':int(groups[0].two_or_more_later_FI_n),
                'recovery_median_last_FI_years':float(groups[1].last_FI_time_median),
                'reference_median_last_FI_years':float(groups[0].last_FI_time_median),
                'internal_review_only':True,'population':'Frozen Primary; first-shock pool shown as distinct upstream denominator'})
    require(len(rows)==25,'三队列冲击波次应为12+7+6')
    return rows


def support_display(r):
    return [r['cohort'],f"W{r['shock_wave']}",f"{r['actual_shock_interview_year_min']}-{r['actual_shock_interview_year_max']}",
        f"{r['first_shock_pool_participants']:,}",f"{r['Primary_recovery_participants']:,}/{r['Primary_reference_participants']:,}",
        f"{r['Primary_recovery_FI_rows']:,}/{r['Primary_reference_FI_rows']:,}",
        f"{r['recovery_at_least_two_later_FI']:,}/{r['reference_at_least_two_later_FI']:,}",
        f"{r['recovery_median_last_FI_years']:.2f}/{r['reference_median_last_FI_years']:.2f}"]


def make_pdfs(out,rows,support,release):
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib.pagesizes import A4,landscape
    from reportlab.lib import colors
    styles=getSampleStyleSheet()
    title=ParagraphStyle('TitleClean',parent=styles['Normal'],fontName='Helvetica-Bold',fontSize=12,leading=15,spaceAfter=8)
    normal=ParagraphStyle('NormalClean',parent=styles['Normal'],fontName='Helvetica',fontSize=8.6,leading=11,spaceAfter=7)
    head=ParagraphStyle('HeaderClean',parent=normal,fontName='Helvetica-Bold',fontSize=8.3,leading=10)
    def table(headers,data,widths,small=False):
        t=Table([[Paragraph(h,head) for h in headers]]+data,colWidths=widths,repeatRows=1)
        t.setStyle(TableStyle([('FONTNAME',(0,1),(-1,-1),'Helvetica'),('FONTSIZE',(0,1),(-1,-1),8.2 if small else 8.5),
            ('LEADING',(0,1),(-1,-1),9.5 if small else 10.5),
            ('TOPPADDING',(0,0),(-1,-1),3 if small else 6),('BOTTOMPADDING',(0,0),(-1,-1),3 if small else 6),
            ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('ALIGN',(2,1),(-1,-1),'RIGHT'),
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAF0F4')),
            ('LINEABOVE',(0,0),(-1,0),.8,colors.black),('LINEBELOW',(0,0),(-1,0),.5,colors.black),
            ('LINEBELOW',(0,-1),(-1,-1),.8,colors.black)]))
        return t
    def doc(path,story):
        SimpleDocTemplate(str(path),pagesize=landscape(A4),leftMargin=30,rightMargin=30,topMargin=26,bottomMargin=26,
            title='Round 1E candidate assets',author='Author review').build(story)
    story=[Paragraph('Candidate supplementary table. Shock-wave adjustment sensitivity',title),
        Paragraph('Source: shock_period_round1e_v1 | Author review candidate | FI units/year',normal)]
    story.append(table(['Cohort','Specification','Participants','FI<br/>observations','Recovery x time<br/>estimate','SE','95% CI','QA qualification'],
        [display(r) for r in rows],[43,66,66,73,87,64,147,108]))
    story.extend([Spacer(1,10),Paragraph('Notes',head),Spacer(1,4)])
    for n in NOTES:story.append(Paragraph(n,normal))
    doc(out/'table_round1e_shock_period_candidate_v1.pdf',story)
    metrics=read_json(release/'QA/review_metrics.json');late=read_json(release/'QA/late_wave_review_summary.json')['cohorts']
    d=[];waveitems=[]
    for c in ('HRS','ELSA','KLoSA'):
        current=[r for r in support if r['cohort']==c];last=current[-1];l=next(r for r in late if r['cohort']==c)
        d.append([c,f"W{current[0]['shock_wave']}-W{last['shock_wave']}",f"W{current[0]['shock_wave']}",
            f"{sum(r['first_shock_pool_participants'] for r in current):,}",
            f"{sum(r['Primary_recovery_participants']+r['Primary_reference_participants'] for r in current):,}",
            f"{last['Primary_recovery_participants']}/{last['Primary_reference_participants']}",
            f"{l['late_influence_share']:.2%}"])
        waveitems.append([c,f"W{l['wave']}",f"{l['late_slope_estimate']:+.6f}",f"{l['late_slope_se']:.6f}",
            f"[{l['late_slope_ci'][0]:+.6f}, {l['late_slope_ci'][1]:+.6f}]"])
    story=[Paragraph('Shock calendar period and observed follow-up support',title),
        Paragraph('Source: shock_period_round1e_v1 | Author review candidate | Aggregate support only',normal),
        table(['Cohort','Shock waves','Reference','First-shock<br/>pool N','Primary N','Latest wave<br/>recovery/reference N','Latest absolute<br/>score-influence share'],
            d,[52,78,61,94,80,145,160]),Spacer(1,12),
        Paragraph('Latest-wave average slope adjustments relative to the reference shock wave',head),Spacer(1,6),
        table(['Cohort','Latest wave','Wave x time adjustment','SE','95% CI'],waveitems,[75,90,160,100,245]),Spacer(1,12)]
    for n in [
        'Every participant in the latest shock wave (HRS W14, ELSA W8, KLoSA W7) has a landmark FI and one later FI only; no participant has two later FI observations. Latest-wave slope SEs are the largest among wave-by-time adjustments within each cohort. These coefficients describe mean FI progression adjustments, not period-specific wealth-recovery associations.',
        'Late-wave absolute individual score-contribution shares are 1.03%, 2.92% and 2.08% in HRS, ELSA and KLoSA, respectively. They do not dominate this fixed-covariance first-order screen. The screen does not reestimate variances, delete participants or waves, or establish that late-wave sensitivity has been excluded.',
        'The first-shock pool is an upstream denominator without Primary recovery/FI/covariate completeness requirements. It is distinct from the frozen Primary model population. Counts are observed support, not confirmed survival, individual administrative opportunity, or censoring risk sets. Last observed FI is not a withdrawal, death or administrative censoring date.',
        'ELSA W10 is nominally labeled 2020, but Primary includes actual 2021, 2022 and 2023 FI observations: 23, 229 and 20 rows, totaling 272 people/rows (101 recovery; 171 reference). Primary FI observation years span 2006-2023. These rows are retained. Verified local Table S1 and saved supplements describe 2002-2023; the journal-uploaded version remains externally unverified.',
        'Conditional-residual skewness and heavy tails persist, especially in KLoSA. HRS/SP01 retains its author-approved numerical exception and original Stage2 HOLD history. SP01 assumes a common recovery-associated slope across waves, does not identify crisis effects, and does not resolve the Round1B model-structure/time-function sensitivity.'
    ]:story.append(Paragraph(n,normal))
    story.extend([PageBreak(),Paragraph('Detailed observed support by shock wave',title),
        Paragraph('R = substantial recovery; Ref = persistent or limited recovery. All R/Ref pairs use this order.',normal),
        table(['Cohort','Shock<br/>wave','Actual shock<br/>interview years','First-shock<br/>pool N','Primary N<br/>R/Ref','FI rows<br/>R/Ref','N with at least<br/>2 later FI: R/Ref','Median last FI time<br/>since landmark: R/Ref (years)'],
            [support_display(r) for r in support],[44,39,91,89,91,93,111,151],small=True),Spacer(1,8),
        Paragraph('Actual shock interview years differ from nominal wave labels. Median last FI times summarize observed follow-up only. FI rows include landmark and later observations. Small cells are retained for internal author review; this detailed support sheet is not a finalized submission table.',normal)])
    doc(out/'period_support_summary_round1e_candidate_v1.pdf',story)


def main(release,freeze_sha,bundle):
    release=require_release_path(release);rows=read_rows(release,freeze_sha)
    out=ROOT/'03_output/3.3 Robustness of the Main Findings/Shock Calendar Period/Candidate Assets/round1e_candidate_v1'
    process=ROOT/'06_project_progress/round1e_shock_period_v1/Stage5/round1e_candidate_v1'
    require(not out.exists() and not process.exists(),'拒绝覆盖候选资产')
    before={rel(p):sha(p) for p in release.rglob('*') if p.is_file()}
    support=support_rows(release)
    out.mkdir(parents=True,exist_ok=False);process.mkdir(parents=True,exist_ok=False)
    csv_new(out/'table_round1e_shock_period_candidate_v1.csv',rows)
    csv_new(out/'period_support_summary_round1e_candidate_v1.csv',support)
    headers=['Cohort','Specification','Participants','FI observations','Estimate','SE','95% CI','QA qualification']
    md=['# Candidate supplementary table. Shock-wave adjustment sensitivity','',f'Source: {RELEASE_ID}. Author review candidate.','',
        '| '+' | '.join(headers)+' |','|---|---|---:|---:|---:|---:|---|---|']
    md+=['| '+' | '.join(display(r))+' |' for r in rows];md+=['','## Notes','']+NOTES+['']
    text_new(out/'table_round1e_shock_period_candidate_v1.md','\n'.join(md))
    escape=lambda s:s.replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
    tex=[r'\documentclass[10pt]{article}',r'\usepackage[a4paper,landscape,margin=18mm]{geometry}',r'\usepackage{booktabs}',
        r'\begin{document}',r'\pagestyle{empty}',r'\noindent\textbf{Candidate supplementary table. Shock-wave adjustment sensitivity}',
        r'\par\smallskip Source: shock\_period\_round1e\_v1. Author review candidate.',r'\begin{center}\small\begin{tabular}{llrrrrll}',
        r'\toprule',' & '.join(escape(h) for h in headers)+r' \\',r'\midrule']
    tex+=[' & '.join(escape(v) for v in display(r))+r' \\' for r in rows]
    tex += [r'\bottomrule\end{tabular}\end{center}',r'\footnotesize']+[r'\noindent '+escape(n)+r'\par\smallskip' for n in NOTES]+[r'\end{document}','']
    text_new(out/'table_round1e_shock_period_candidate_v1.tex','\n'.join(tex))
    smd=['# Shock calendar period and observed follow-up support','',f'Source: {RELEASE_ID}. Author review candidate; internal aggregate detail.','',
        'SP01 assumes a common recovery-associated annual slope across shock waves; wave-by-time adjustments are not period-specific recovery associations.',
        'All latest-wave participants have landmark plus one later FI only. Late-wave sensitivity is not fully excluded.','',
        '| Cohort | Shock wave | Actual shock interview years | First-shock pool N | Primary N R/Ref | FI rows R/Ref | At least 2 later FI R/Ref | Median last FI years R/Ref |',
        '|---|---|---|---:|---|---|---|---|']
    smd+=['| '+' | '.join(support_display(r))+' |' for r in support]
    smd+=['','R = substantial recovery; Ref = persistent or limited recovery. The first-shock pool and Primary are distinct denominators. Support counts are not survival or censoring risk sets. Last FI is not a withdrawal/death date.','',
        'ELSA nominal W10=2020 includes actual 2021-2023 FI (23/229/20 rows; 272 people/rows). Current local Table S1 is consistent with 2002-2023; journal upload identity remains externally unverified.','']
    text_new(out/'period_support_summary_round1e_candidate_v1.md','\n'.join(smd))
    sys.path.append(str(bundle));make_pdfs(out,rows,support,release)
    from pypdf import PdfReader
    tablepdf=PdfReader(out/'table_round1e_shock_period_candidate_v1.pdf');supportpdf=PdfReader(out/'period_support_summary_round1e_candidate_v1.pdf')
    require(len(tablepdf.pages)==1 and len(supportpdf.pages)==2,'候选表应1页，支持摘要应2页')
    tabletext=' '.join(' '.join(p.extract_text() for p in tablepdf.pages).split())
    for r in rows:
        require(f"{r['estimate']:+.6f}" in tabletext and f"{r['se']:.6f}" in tabletext,'PDF数值不匹配')
    require('precision loss (status 2)' in tabletext and 'original Stage2 HOLD' in tabletext and
        'common recovery-associated annual slope' in tabletext,'关键科学或例外表注缺失')
    supporttext=' '.join(' '.join(p.extract_text() for p in supportpdf.pages).split())
    require('272' in supporttext and 'not a withdrawal' in supporttext,'支持摘要时间或分母限定缺失')
    preview=out/'Preview';preview.mkdir()
    renderer=shutil.which('pdftoppm');require(renderer is not None,'缺少PDF渲染器')
    for stem in ['table_round1e_shock_period_candidate_v1','period_support_summary_round1e_candidate_v1']:
        subprocess.run([renderer,'-r','130','-png',str(out/(stem+'.pdf')),str(preview/stem)],check=True)
    for name,digest in before.items():require(sha(ROOT/name)==digest,'制表改变了冻结包')
    csvrows=list(csv.DictReader((out/'table_round1e_shock_period_candidate_v1.csv').open(encoding='utf-8',newline='')))
    for actual,expected in zip(csvrows,rows):
        for key in ('estimate','se','ci_lower','ci_upper','p_value'):require(float(actual[key])==expected[key],'CSV未保留全精度')
    validation={'status':'NUMERIC_SOURCE_AND_RENDER_CHECK_PASS_PENDING_VISUAL_REVIEW','release_id':RELEASE_ID,
        'release_freeze_sha256':freeze_sha,'rows':6,'period_wave_rows':25,'pdf_pages':{'table':1,'support':2},
        'CSV_full_precision':True,'display_decimals':6,'source_mode':'RELEASE_ONLY_NO_WORKING_FALLBACK',
        'release_unchanged':True,'HRS_exception_and_HOLD_retained':True,'visual_review':'PENDING_IMAGE_INSPECTION',
        'author_candidate_review':'PENDING','submission_ready':False,'manuscript_changed':False,'new_model_fits':0,
        'output_hashes':{rel(p):sha(p) for p in out.rglob('*') if p.is_file()},'builder_sha256':sha(__file__),
        'validator_sha256':sha(Path(__file__).resolve().parent.parent/'release/validate_release.py'),
        'PDF_dependency_paths':{'reportlab':str(__import__('reportlab').__file__),'pypdf':str(__import__('pypdf').__file__)},**provenance()}
    write_json(process/'candidate_assets_manifest.json',validation)
    print(json.dumps({'status':validation['status'],'table_pdf':str(out/'table_round1e_shock_period_candidate_v1.pdf'),
        'support_pdf':str(out/'period_support_summary_round1e_candidate_v1.pdf'),'preview':str(preview)},ensure_ascii=False,indent=2))


if __name__=='__main__':
    import argparse
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--freeze-sha256',required=True)
    p.add_argument('--pdf-dependencies',type=Path,required=True);a=p.parse_args();main(a.release,a.freeze_sha256,a.pdf_dependencies)
