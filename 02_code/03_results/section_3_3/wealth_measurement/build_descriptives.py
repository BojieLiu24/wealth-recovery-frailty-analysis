"""WM00：按原主要参与者每人一次输出财富描述统计。"""
import numpy as np
import pandas as pd
from prepare_inputs import require

def summarize(contract,cohort,person):
 require(not person.person_id.duplicated().any(),'描述统计重复参与者')
 data=person.copy()
 data['absolute_loss']=data.wealth_tminus1-data.wealth_t
 data['absolute_recovery']=data.wealth_tplus1-data.wealth_t
 fields=['wealth_tminus1','wealth_t','wealth_tplus1','absolute_loss','absolute_recovery','recovery_fraction_raw']
 rows=[]
 for group,df in data.groupby('recovery_status',observed=True):
  for field in fields:
   v=df[field].to_numpy(float);require(np.isfinite(v).all(),'描述统计非有限金额')
   rows.append({'analysis_id':'WM00','cohort':cohort,'recovery_status':group,'variable':field,'unit':'ratio' if field=='recovery_fraction_raw' else contract['analyses'][0]['amount_units'][cohort], 'n':len(v),'median':float(np.median(v)),'q1':float(np.quantile(v,.25)),'q3':float(np.quantile(v,.75)),'negative_n':int((v<0).sum()),'negative_percent':float((v<0).mean()*100),'zero_n':int((v==0).sum()),'zero_percent':float((v==0).mean()*100)})
 return rows
