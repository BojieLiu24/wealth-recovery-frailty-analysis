"""Stage5仅从指定Round1C冻结包生成一张英文候选表，不读工作结果。"""
import argparse
import csv
from pathlib import Path
import shutil
import subprocess
import sys
from lf_common import ROOT,CODE,PROGRESS,read_json,write_json,sha,rel,require,provenance
from run_landmark_fi import safe_id
from validate_round1c_release import validate,RELEASE_ID

NOTES=[
 'Abbreviations: FI, frailty index; SE, standard error; CI, confidence interval; HRS, Health and Retirement Study; ELSA, English Longitudinal Study of Ageing; KLoSA, Korean Longitudinal Study of Aging.',
 'The coefficient is the substantial-recovery minus persistent-or-limited-recovery difference in annual FI change (FI units/year). Estimates are cohort-specific, unweighted, maximum-likelihood random-intercept LMMs. Primary is the frozen original reference; it was not refitted.',
 'PO01 uses only post-landmark FI observations (T > 0), retaining all Primary participants and the original adjustment, reference categories and frozen standardization. LM01 uses exactly the same post-landmark rows and additionally includes standardized landmark FI and its interaction with time; shock FI and its time interaction remain included.',
 'LM01 is a post-landmark conditional association analysis, not an alternative estimate of a causally identified effect of wealth recovery. Conditioning may introduce overadjustment or collider/selection bias. Differences between coefficients do not identify mediation or reverse-causality contributions.',
 'Time remains years since the original landmark. Post-only intercepts and recovery main effects extrapolate to time zero; no observed landmark FI level differences are inferred. CIs use the full observed-information covariance and normal Wald inference.',
 'KLoSA uses an equivalent 41-column PO01 / 43-column LM01 basis: Q1 x T = (23/12) Q1. All participants are retained; the combined Q1 effect is estimated without separate, nonidentifiable Q1 main and time-interaction coefficients.',
 'NE: HRS/PO01 was accepted by the author with a numerical exception after independent validation. Its original Stage2 HOLD is retained: BFGS converged, L-BFGS did not report convergence, and the exact original SciPy termination reason cannot be recovered. This is not an ordinary numerical PASS or proof of global optimality.',
 'All new results were accepted with caveats: conditional-residual skew/heavy tails, sparse distant follow-up in ELSA, and long-follow-up influence risk, particularly in KLoSA. Fixed-covariance influence screening is a first-order diagnostic, not deletion refitting or a replacement for LMM SEs; distant-follow-up sensitivity is not fully excluded.',
 'Round1C does not resolve the HRS uncertainty increase under Round1B random-slope or nonlinear-time specifications. No random-slope/landmark-FI or nonlinear-time/landmark-FI joint models were fitted. This is a candidate table for author review; the manuscript and existing tables/figures are unchanged.'
]


def read_release_rows(release):
    # 仅验证包内资产；不存在工作目录回退，也不读取旧的latest指针。
    validate(release,'release-only')
    approval=read_json(release/'Evidence/Approvals/author_approval.json')
    require(approval['stage5_candidate_table_authorized'] is True,'候选表未授权')
    with (release/'Comparisons/cohort_estimates.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    require(len(rows)==9,'候选表须九行')
    for row in rows:
        for key in ['estimate','se','ci_lower','ci_upper','p_value']:row[key]=float(row[key])
        for key in ['participants','FI_observations']:row[key]=int(row[key])
        row['QA_qualification']='Frozen reference' if row['specification']=='Primary' else (
            'With caveats (NE)' if (row['cohort'],row['specification'])==('HRS','PO01') else 'With caveats')
    return rows


def display(row):
    return [row['cohort'],row['specification'],f"{row['participants']:,}",f"{row['FI_observations']:,}",
            f"{row['estimate']:+.6f}",f"{row['se']:.6f}",f"[{row['ci_lower']:+.6f}, {row['ci_upper']:+.6f}]",row['QA_qualification']]


def normalized_pdf_text(text):
    # PDF换行不能被误判为表注缺失；只归一化空白，不改正文或数字。
    return ' '.join(text.split())


def tex_escape(text):
    return text.replace('\\',r'\textbackslash{}').replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('>',r'$>$')


def write_tex(path,rows):
    lines=[r'\documentclass[10pt]{article}',r'\usepackage[a4paper,landscape,margin=18mm]{geometry}',
           r'\usepackage{booktabs,tabularx}',r'\begin{document}',r'\pagestyle{empty}',
           r'\noindent\textbf{Candidate supplementary table. Landmark FI conditional sensitivity}',
           r'\par\smallskip\noindent Source: landmark\_fi\_round1c\_v1. Author review draft.',
           r'\begin{center}\small',r'\begin{tabular}{llrrrrll}',r'\toprule',
           r'Cohort & Specification & Participants & FI observations & Estimate & SE & 95\% CI & QA qualification \\',r'\midrule']
    for row in rows:lines.append(' & '.join(tex_escape(x) for x in display(row))+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{center}',r'\footnotesize'])
    lines.extend(r'\noindent '+tex_escape(note)+r'\par\smallskip' for note in NOTES)
    lines.extend([r'\end{document}',''])
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines))


def pdf(path,rows,bundle):
    # PDF使用已配置的只读依赖；不修改仓库环境或uv.lock。
    if bundle:sys.path.append(str(bundle))
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4,landscape
    styles=getSampleStyleSheet()
    title=ParagraphStyle('TableTitle',parent=styles['Normal'],fontName='Helvetica-Bold',fontSize=12,leading=15)
    note=ParagraphStyle('TableNote',parent=styles['Normal'],fontSize=8.3,leading=10.2,spaceAfter=5)
    header=ParagraphStyle('TableHeader',parent=styles['Normal'],fontName='Helvetica-Bold',fontSize=8.3,leading=10)
    story=[Paragraph('Candidate supplementary table. Landmark FI conditional sensitivity',title),Spacer(1,6),
           Paragraph('Source: landmark_fi_round1c_v1 | Author review draft | Estimate unit: FI/year',note),Spacer(1,6)]
    headers=['Cohort','Specification','Participants','FI<br/>observations','Recovery x time<br/>estimate','SE','95% CI','QA qualification']
    data=[[Paragraph(s,header) for s in headers]]+[display(r) for r in rows]
    table=Table(data,colWidths=[43,66,66,73,87,64,147,108],repeatRows=1)
    table.setStyle(TableStyle([('FONTNAME',(0,1),(-1,-1),'Helvetica'),('FONTSIZE',(0,1),(-1,-1),8.4),
      ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
      ('ALIGN',(2,1),(6,-1),'RIGHT'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAF0F4')),
      ('LINEABOVE',(0,0),(-1,0),.8,colors.black),('LINEBELOW',(0,0),(-1,0),.5,colors.black),
      ('LINEBELOW',(0,-1),(-1,-1),.8,colors.black),('LINEABOVE',(0,4),(-1,4),.4,colors.HexColor('#AAAAAA')),
      ('LINEABOVE',(0,7),(-1,7),.4,colors.HexColor('#AAAAAA'))]))
    story.extend([table,Spacer(1,12),Paragraph('Notes',header),Spacer(1,4)])
    for note_text in NOTES:story.append(Paragraph(note_text.replace('>','&gt;'),note))
    doc=SimpleDocTemplate(str(path),pagesize=landscape(A4),leftMargin=30,rightMargin=30,topMargin=26,bottomMargin=26,
                          title='Round1C landmark FI conditional sensitivity - candidate table',author='Author review draft')
    doc.build(story)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--candidate-id',required=True);parser.add_argument('--pdf-dependencies',type=Path)
    parser.add_argument('--validate-existing',action='store_true')
    args=parser.parse_args();candidate_id=safe_id(args.candidate_id);release=args.release.resolve()
    require(release==ROOT/'03_output/3.3 Robustness of the Main Findings/releases'/RELEASE_ID,'必须明确读取指定正式release')
    rows=read_release_rows(release);before={rel(p):sha(p) for p in release.rglob('*') if p.is_file()}
    out=ROOT/'03_output/3.3 Robustness of the Main Findings/Landmark FI Sensitivity/Candidate Assets'/candidate_id
    process=PROGRESS/'Stage5'/candidate_id
    stem='table_round1c_landmark_fi_candidate_v1'
    if args.validate_existing:
        require(out.is_dir() and process.is_dir() and not (process/'candidate_table_manifest.json').exists(),
                '只能补齐未完成的本次候选表验证，不覆盖完成记录')
        with (out/(stem+'.csv')).open(encoding='utf-8',newline='') as f:existing=list(csv.DictReader(f))
        require(len(existing)==9,'现有候选表行数错误')
        for actual,expected in zip(existing,rows):
            for key,value in expected.items():
                require((float(actual[key])==value) if isinstance(value,(float,int)) else actual[key]==value,'现有CSV与冻结来源不符')
    else:
        require(not out.exists() and not process.exists(),'拒绝覆盖候选表')
        out.mkdir(parents=True,exist_ok=False);process.mkdir(parents=True,exist_ok=False)
        with (out/(stem+'.csv')).open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        write_tex(out/(stem+'.tex'),rows);pdf(out/(stem+'.pdf'),rows,args.pdf_dependencies)
    if args.pdf_dependencies:sys.path.append(str(args.pdf_dependencies))
    from pypdf import PdfReader
    reader=PdfReader(str(out/(stem+'.pdf')));text=normalized_pdf_text('\n'.join(p.extract_text() for p in reader.pages))
    require(len(reader.pages)==1,'可读候选表应为单页')
    for row in rows:
        require(f"{row['estimate']:+.6f}" in text and f"{row['se']:.6f}" in text,'PDF数值缺失')
    require('not an alternative estimate of a causally identified effect' in text and
            'exact original SciPy termination reason' in text,'关键表注缺失')
    renderer=shutil.which('pdftoppm');require(renderer is not None,'缺少PDF渲染器')
    preview=out/'Preview';preview.mkdir()
    subprocess.run([renderer,'-scale-to','1800','-png','-singlefile',str(out/(stem+'.pdf')),str(preview/'table_preview')],check=True)
    for path,digest in before.items():require(sha(ROOT/path)==digest,'制表修改了冻结包')
    validation={'status':'PASS_NUMERIC_TEXT_AND_SOURCE_CHECK','rows':9,'pdf_pages':len(reader.pages),
       'release_id':RELEASE_ID,'release_freeze_sha256':sha(release/'freeze.json'),
       'display_estimate_decimals':6,'display_SE_decimals':6,'raw_CSV_full_precision':True,
       'source_mode':'RELEASE_ONLY_NO_WORKING_DIRECTORY_FALLBACK','release_unchanged':True,
       'manuscript_changed':False,'submission_ready':False,'author_table_review':'PENDING',
       'visual_review':'PENDING_IMAGE_INSPECTION','output_hashes':{rel(p):sha(p) for p in out.rglob('*') if p.is_file()},
       'builder_sha256':sha(__file__),'validator_sha256':sha(CODE/'validate_round1c_release.py'),**provenance()}
    write_json(process/'candidate_table_manifest.json',validation)
    print({'status':validation['status'],'candidate_table':str(out/(stem+'.pdf')),'preview':str(preview/'table_preview.png')})


if __name__=='__main__':main()
