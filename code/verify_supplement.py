"""逐行逐列核对最终补表的显示值与公开聚合，不将格式差异当成科学差异。"""
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT/'outputs/published_aggregates'
TABLES = ROOT/'docs/reference_tables'
COHORTS = ['HRS','ELSA','KLoSA']


def numbers(cell):
    cell = cell.replace('$','').replace('−','-')
    return [float(x) for x in re.findall(r'-?\d+\.\d+',cell)]


def compare(values, expected, places):
    if len(values) != len(expected):
        raise ValueError(f'Table schema mismatch: {values} versus {expected}')
    tolerance = .500001 * 10**(-places)
    if any(abs(a-b)>tolerance for a,b in zip(values,expected)):
        raise ValueError(f'Supplement mismatch: {values} versus {expected}')
    return len(values)


def verify():
    checks = {}
    data = pd.read_csv(AGG/'sensitivity_S7.csv', float_precision='round_trip').sort_values('display_order')
    lines = [l for l in (TABLES/'Table S7.tex').read_text(encoding='utf-8').splitlines() if re.search(r'& (?:HRS|ELSA|KLoSA|\\textbf\{Overall)',l)]
    if len(lines) != len(data):
        raise ValueError('Table S7 row count differs from the public aggregate.')
    n=0
    for line,(_,r) in zip(lines,data.iterrows()):
        cells=line.split('&')
        cohort = re.sub(r'\\textbf\{([^}]+)\}',r'\1',cells[1]).strip()
        if cohort != r['cohort']:
            raise ValueError('Table S7 cohort order mismatch.')
        n+=compare(numbers(cells[4])+numbers(cells[5]),[r.estimate,r.ci_lower,r.ci_upper],4)
    checks['Table S7']=n
    data=pd.read_csv(AGG/'wealth_S9.csv',float_precision='round_trip')
    lines=[l for l in (TABLES/'Table S9.tex').read_text(encoding='utf-8').splitlines() if '\\shortstack[c]' in l]
    if len(lines)!=6:raise ValueError('Table S9 row schema mismatch.')
    n=0
    for spec,line in zip(['PRIMARY','WM01','WM02','WM03','WM04','WM05'],lines):
        cells=line.split('&')
        for i,cohort in enumerate(COHORTS):
            r=data[(data.analysis_id==spec)&(data.cohort==cohort)].iloc[0]
            if int(cells[1+2*i].replace(',','').strip())!=int(r.participants):raise ValueError('Table S9 population mismatch.')
            n+=compare(numbers(cells[2+2*i]),[r.estimate,r.ci_lower,r.ci_upper],6)
    checks['Table S9']=n
    data=pd.read_csv(AGG/'longitudinal_S10.csv',float_precision='round_trip')
    lines=[l for l in (TABLES/'Table S10.tex').read_text(encoding='utf-8').splitlines() if '\\shortstack[c]' in l]
    if len(lines)!=4:raise ValueError('Table S10 row schema mismatch.')
    n=0
    for spec,scale,line in zip(['Primary','RS01','NL01','NL01'],[1,1,1,8],lines):
        cells=line.split('&')
        for i,cohort in enumerate(COHORTS):
            r=data[(data.spec_id==spec)&(data.cohort==cohort)].iloc[0]
            n+=compare(numbers(cells[1+2*i])+numbers(cells[2+2*i]),[scale*r.estimate,scale*r.ci_lower,scale*r.ci_upper,scale*r.se],6)
    checks['Table S10']=n
    for table,name,specs in [(11,'landmark_S11.csv',['Primary','PO01','LM01']),(12,'shock_wave_S12.csv',['Primary','SP01'])]:
        data=pd.read_csv(AGG/name,float_precision='round_trip')
        lines=[l for l in (TABLES/f'Table S{table}.tex').read_text(encoding='utf-8').splitlines() if re.search(r'& (HRS|ELSA|KLoSA)(?: |\\)',l)]
        if len(lines)!=len(specs)*3:raise ValueError(f'Table S{table} row schema mismatch.')
        n=0
        for j,line in enumerate(lines):
            cells=line.split('&');cohort=re.match(r'(HRS|ELSA|KLoSA)',cells[1].strip()).group(1);spec=specs[j//3]
            r=data[(data.specification==spec)&(data.cohort==cohort)].iloc[0]
            if int(cells[2].replace(',','').strip())!=int(r.participants):raise ValueError('Population mismatch.')
            n+=compare(numbers(cells[4])+numbers(cells[5])+numbers(cells[6]),[r.estimate,r.se,r.ci_lower,r.ci_upper],6)
        checks[f'Table S{table}']=n
    return {'status':'PASS','mode':'AGGREGATE_BASED_VERIFICATION','numeric_cells_checked':sum(checks.values()),'tables':checks}


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2))
