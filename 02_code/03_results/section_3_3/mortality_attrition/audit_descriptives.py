"""聚合选择、观察模式与SMD；不输出参与者记录或推断检验。"""
import json
import numpy as np
import pandas as pd
from audit_sources import require
from audit_states import pools,failed_conditions,FINAL_GATES

NUMERIC=['fi_t','fi_landmark','age','household_size','household_income_rank','wealth_loss_fraction','lm_year']
CATEGORICAL=['sex','education','marital_status','labour_force_status','current_smoking','wealth_quintile']


def available_number(value):
    return np.nan if pd.isna(value) else float(value)


def smd_continuous(a,b):
    a=pd.to_numeric(a,errors='coerce').dropna();b=pd.to_numeric(b,errors='coerce').dropna()
    if len(a)<5 or len(b)<5:return None,'INSUFFICIENT_GROUP_SUPPORT'
    scale=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
    if scale==0:return (0.,'ZERO_DIFFERENCE_ZERO_VARIANCE') if a.mean()==b.mean() else (None,'DEGENERATE_DENOMINATOR')
    return float((a.mean()-b.mean())/scale),'CALCULABLE_DESCRIPTIVE_ONLY'


def count_row(cohort,population,node,mask,denominator,unit,source,reason=''):
    count=int(np.asarray(mask,bool).sum());den=int(denominator)
    return {'cohort':cohort,'population_id':population,'node_id':node,'unit':unit,'denominator_n':den,
            'satisfied_n':count,'excluded_n':den-count,'reason':reason,'source_evidence':source,
            'computability_status':'VERIFIED','small_positive_count':0<count<5,'public_disclosure_approved':False}


def flow_counts(cohort,person,exposure,candidate,eligibility):
    source='Frozen Exposure + Full Spine + Ledger + Final Eligibility; actual code gate order'
    rows=[];base=eligibility.merge(exposure,on=['cohort','person_id','candidate_shock_wave'],validate='one_to_one',suffixes=('','_exposure'))
    rows.append(count_row(cohort,'upstream','global_candidate_intervals',np.ones(len(candidate),bool),len(candidate),'person_interval',source))
    m=base.structural_eligibility_status.eq('pass')
    rows.append(count_row(cohort,'upstream','structural_eligible_intervals',m,len(base),'person_interval',source))
    evaluable=m&base.wealth_tminus1.gt(0)&base.wealth_t.notna()&base.wealth_tminus1.notna()
    rows.append(count_row(cohort,'upstream','wealth_evaluable_intervals',evaluable,int(m.sum()),'person_interval',source))
    # 可评估财富节点使用结构通过子集，避免把总区间分母误称独立人数。
    rows[-1]['satisfied_n']=int(evaluable.sum());rows[-1]['excluded_n']=int(m.sum()-evaluable.sum())
    window={'HRS':(3,14),'ELSA':(2,8),'KLoSA':(2,7)}[cohort]
    qualifying=base.major_shock.eq(True).fillna(False)&base.candidate_shock_wave.between(*window)
    first=base.loc[qualifying].sort_values('candidate_shock_wave').drop_duplicates('person_id')
    index=base.loc[base.is_index_shock]
    require(set(first.person_id)==set(index.person_id)==set(person.person_id),'首次冲击集合不一致')
    require(first.set_index('person_id').candidate_shock_wave.sort_index().equals(index.set_index('person_id').candidate_shock_wave.sort_index()),'首次冲击选择漂移')
    rows.append(count_row(cohort,'shock','first_qualifying_index_shock',np.ones(len(person),bool),base.person_id.nunique(),'participant',source,
                          'Observed qualifying shock within locked analytic window; not lifetime first shock'))
    active=np.ones(len(person),bool)
    for gate,failed in zip(FINAL_GATES,failed_conditions(person)):
        den=int(active.sum());active=active&~failed.to_numpy(bool)
        row=count_row(cohort,'actual_primary_flow',gate,active,den,'participant',source)
        row['satisfied_n']=int(active.sum());row['excluded_n']=den-int(active.sum());rows.append(row)
    require(np.array_equal(active,person.primary_complete_case_eligible.to_numpy(bool)),'重建Primary资格不一致')
    pp=pools(person)
    for name,parent in [('L0',np.ones(len(person),bool)),('L1',pp['L0']),('L2',pp['L1']),('Primary',pp['L2'])]:
        require(not (pp[name]&~parent).any(),'诊断池不是嵌套集合')
        row=count_row(cohort,'diagnostic_pools',name,pp[name],int(parent.sum()),'participant',source)
        row['excluded_n']=int(parent.sum()-pp[name].sum());rows.append(row)
    for pool in ['L0','L1','L2']:
        mask=pp[pool];subset=person.loc[mask]
        for col in ['landmark_fi_available','lm_timing_valid','fi_t_available','nine_covariates_complete','has_valid_post_FI','has_timed_post_FI']:
            rows.append(count_row(cohort,pool,col,subset[col].fillna(False),len(subset),'participant',source,'Parallel measurement support, not reordered original gates'))
    for i,gate in enumerate(FINAL_GATES):
        f=failed_conditions(person)[i]
        rows.append(count_row(cohort,'overlapping_failure_flags',gate,f,len(person),'participant',source,'Overlap allowed; not exclusive exclusions'))
    for i,a in enumerate(FINAL_GATES):
        for j in range(i+1,len(FINAL_GATES)):
            mask=failed_conditions(person)[i]&failed_conditions(person)[j]
            rows.append(count_row(cohort,'pairwise_failure_overlap',a+' AND '+FINAL_GATES[j],mask,len(person),'participant',source,'Overlap, not exclusive exclusions'))
    return rows


def observability(cohort,person,post):
    rows=[];pp=pools(person)
    def add(pop,group,metric,num,den,unit='participant',status='CALCULABLE',stratum='all',meaning=''):
        rows.append({'cohort':cohort,'population_id':pop,'recovery_status':group,'stratum':stratum,'metric':metric,
             'numerator':num,'denominator':den,'value':num/den if num is not None and den else None,'unit':unit,
             'computability_status':status,'meaning':meaning,'small_positive_count':bool(num is not None and 0<num<5),
             'public_disclosure_approved':False})
    flags=['has_valid_post_FI','has_timed_post_FI','post_FI_gap_then_return','post_interview_gap_then_return',
          'post_landmark_gap_then_FI','any_post_interview_invalid_FI','no_post_FI_but_any_post_interview',
          'known_death_code_after_landmark','known_death_date_after_landmark_snapshot','any_presumed_alive_nonresponse',
          'any_unknown_vital_status','any_conflicting_source','any_permanent_exit','any_confirmed_live_post_interview']
    for population,mask in pp.items():
        block=person.loc[mask]
        for group,g in block.groupby('recovery_status',dropna=False):
            label=str(group);n=len(g);r=post.loc[post.person_id.isin(g.person_id)]
            add(population,label,'population_n',n,n)
            for flag in flags:
                if cohort=='ELSA' and flag=='known_death_date_after_landmark_snapshot':
                    add(population,label,flag,None,n,status='NOT_AVAILABLE_NATIVE_DEATH_DATE')
                else:add(population,label,flag,int(g[flag].fillna(False).sum()),n,meaning='Current snapshot confirmed events; no incidence/risk-set claim' if 'death' in flag else '')
            for k,num in g.valid_post_FI.value_counts().sort_index().items():add(population,label,'valid_FI_count_'+str(int(k)),int(num),n)
            for state,num in g.administrative_opportunity_class.value_counts().items():add(population,label,'person_opportunity_'+state,int(num),n)
            nofi=g.loc[~g.has_valid_post_FI.astype(bool)]
            for flag in ['known_death_code_after_landmark','known_death_date_after_landmark_snapshot','any_presumed_alive_nonresponse',
                         'any_post_interview_invalid_FI','any_unknown_vital_status','any_permanent_exit']:
                if cohort=='ELSA' and flag=='known_death_date_after_landmark_snapshot':
                    add(population,label,'no_first_FI_'+flag,None,len(nofi),status='NOT_AVAILABLE_NATIVE_DEATH_DATE')
                else:add(population,label,'no_first_FI_'+flag,int(nofi[flag].fillna(False).sum()),len(nofi),meaning='Overlapping evidence, not mutually exclusive reasons')
            for time in ['first_FI_time','last_FI_time']:
                a=g[time].dropna()
                for q in [.25,.5,.75]:
                    add(population,label,time+'_P'+str(int(q*100)),None,len(a),unit='years',status='NO_VALID_TIME' if not len(a) else 'CALCULABLE')
                    rows[-1]['value']=float(a.quantile(q)) if len(a) else None
            for wave,w in r.groupby('wave'):
                for field in ['administrative_opportunity_frame','vital_evidence','FI_observation_state']:
                    for value,count in w[field].value_counts().items():
                        add(population,label,field+':'+value,int(count),len(w),unit='person_wave',stratum='wave_'+str(int(wave)),
                            meaning='Schedule-frame presence never implies confirmed survival' if field=='administrative_opportunity_frame' else '')
                add(population,label,'mortality_coverage_gap_rows',int(w.mortality_coverage_gap.sum()),len(w),unit='person_wave',stratum='wave_'+str(int(wave)))
                if 'proxy_raw' in w:
                    interviewed=w.loc[w.observed_in_wave.eq(True)]
                    for code,g3 in interviewed.groupby('proxy_raw'):
                        add(population,label,'FI_available_by_native_proxy_code_'+str(code),int(g3.fi_primary.notna().sum()),len(g3),
                            unit='person_wave',stratum='wave_'+str(int(wave)),meaning='Native proxy code retained; no automatic proxy exclusion')
                for evidence,g3 in w.groupby('vital_evidence'):
                    if evidence in ['LIVE_AT_ACTUAL_INTERVIEW','SOURCE_PRESUMED_ALIVE_NONRESPONSE']:
                        add(population,label,'FI_available_given_'+evidence,int(g3.fi_primary.notna().sum()),len(g3),
                            unit='person_wave',stratum='wave_'+str(int(wave)),meaning='Separate live-confirmed and source-presumed support; ELSA mortality gap still applies')
            for wave,g2 in g.groupby('landmark_wave'):
                add(population,label,'first_post_FI_by_landmark_wave',int(g2.has_valid_post_FI.sum()),len(g2),stratum='landmark_wave_'+str(int(wave)))
            valid=r.loc[r.fi_primary.notna()&r.time_years.gt(0)].sort_values(['person_id','wave'])
            gaps=valid.groupby('person_id').time_years.diff().dropna()
            add(population,label,'adjacent_valid_FI_interval_median',None,len(gaps),unit='years',status='CALCULABLE' if len(gaps) else 'NO_PAIR_SUPPORT')
            rows[-1]['value']=float(gaps.median()) if len(gaps) else None
            add(population,label,'exact_administrative_censor_date',None,n,unit='date',status='NOT_AVAILABLE',meaning='Last global wave is not a person-specific exact date')
            if cohort=='ELSA':add(population,label,'complete_death_count_W6_W10',None,n,status='NOT_IDENTIFIABLE_MORTALITY_COVERAGE_GAP')
    # 未达到landmark者只作冲击前/冲击时描述，不分恢复组。
    unclassified=person.loc[~person.recovery_classifiable|~person.lm_observed.eq(True).fillna(False)]
    for state,count in unclassified.lm_vital_evidence.value_counts().items():
        add('pre_landmark_not_L0','NOT_DEFINED',state,int(count),len(unclassified),meaning='No recovery-group assignment')
    return rows


def health_descriptives(cohort,person,post):
    rows=[]
    def compare(block,flag,population,comparison):
        a=block.loc[flag];b=block.loc[~flag]
        for variable in NUMERIC:
            if variable not in block:continue
            if population=='shock' and variable in ['fi_landmark','lm_year']:continue
            smd,status=smd_continuous(a[variable],b[variable]);x=pd.to_numeric(a[variable],errors='coerce');y=pd.to_numeric(b[variable],errors='coerce')
            rows.append({'cohort':cohort,'population_id':population,'comparison':comparison,'variable':variable,
                'level':'continuous','group1_n':len(a),'group0_n':len(b),'group1_available_n':int(x.notna().sum()),
                'group0_available_n':int(y.notna().sum()),'group1_mean':available_number(x.mean()),'group0_mean':available_number(y.mean()),
                'group1_sd':available_number(x.std(ddof=1)),'group0_sd':available_number(y.std(ddof=1)),'SMD':smd,'computability_status':status})
        for variable in CATEGORICAL:
            for level in block[variable].astype('string').fillna('MISSING').unique():
                x=a[variable].astype('string').fillna('MISSING').eq(level);y=b[variable].astype('string').fillna('MISSING').eq(level)
                p=float(x.mean()) if len(a) else np.nan;q=float(y.mean()) if len(b) else np.nan
                scale=np.sqrt((p*(1-p)+q*(1-q))/2)
                value=(p-q)/scale if len(a)>=5 and len(b)>=5 and scale>0 else 0. if p==q and len(a)>=5 and len(b)>=5 else None
                rows.append({'cohort':cohort,'population_id':population,'comparison':comparison,'variable':variable,'level':str(level),
                  'group1_n':len(a),'group0_n':len(b),'group1_proportion':p,'group0_proportion':q,'SMD':value,
                  'computability_status':'CALCULABLE_DESCRIPTIVE_ONLY' if value is not None else 'INSUFFICIENT_OR_DEGENERATE_SUPPORT'})
    pp=pools(person)
    compare(person,pp['L0'],'shock','entered_L0_vs_not_L0')
    for pop in ['L0','L1','L2','Primary']:
        block=person.loc[pp[pop]]
        compare(block,block.has_valid_post_FI.astype(bool),pop,'first_post_FI_vs_no_first_post_FI')
        for group,g in block.groupby('recovery_status'):
            compare(g,g.has_valid_post_FI.astype(bool),pop+':'+str(group),'first_post_FI_vs_none')
        for variable in ['fi_t','fi_landmark']:
            value=pd.to_numeric(block[variable],errors='coerce')
            try:strata=pd.qcut(value,4,duplicates='drop')
            except ValueError:continue
            for label,g in block.groupby(strata,observed=True):
                rows.append({'cohort':cohort,'population_id':pop,'comparison':'health_quartile_first_FI_support','variable':variable,
                  'level':str(label),'group1_n':int(g.has_valid_post_FI.sum()),'group0_n':int((~g.has_valid_post_FI.astype(bool)).sum()),
                  'group1_proportion':float(g.has_valid_post_FI.mean()),'computability_status':'DESCRIPTIVE_STRATUM_NO_CAUSAL_INFERENCE'})
        for label,g in block.groupby(pd.cut(block.age.astype(float),[59,69,79,np.inf],labels=['60-69','70-79','80+']),observed=True):
            rows.append({'cohort':cohort,'population_id':pop,'comparison':'age_stratum_first_FI_support','variable':'age',
               'level':str(label),'group1_n':int(g.has_valid_post_FI.sum()),'group0_n':int((~g.has_valid_post_FI.astype(bool)).sum()),
               'group1_proportion':float(g.has_valid_post_FI.mean()),'computability_status':'DESCRIPTIVE_STRATUM_NO_CAUSAL_INFERENCE'})
    # 每个目标波次之前的最近FI；当前波次及未来回访不进入既往健康值。
    chronological=post.sort_values(['person_id','wave']).copy()
    for pop in ['L1','L2','Primary']:
        subset=chronological.loc[chronological.person_id.isin(person.loc[pp[pop],'person_id'])]
        for wave,g in subset.groupby('wave'):
            eligible=g.vital_evidence.isin(['LIVE_AT_ACTUAL_INTERVIEW','SOURCE_PRESUMED_ALIVE_NONRESPONSE'])
            g=g.loc[eligible&g.prior_FI.notna()]
            a=g.loc[g.fi_primary.notna(),'prior_FI'];b=g.loc[g.fi_primary.isna(),'prior_FI'];s,status=smd_continuous(a,b)
            rows.append({'cohort':cohort,'population_id':pop,'comparison':'prior_FI_observed_vs_not_observed_at_wave_'+str(int(wave)),
               'variable':'strictly_prior_FI','level':'prior_only','group1_n':len(a),'group0_n':len(b),
               'group1_mean':available_number(a.mean()),'group0_mean':available_number(b.mean()),'SMD':s,'computability_status':status,
               'death_handling':'confirmed deaths and unknown vital states excluded from this limited descriptive comparison'})
    return rows
