"""独立选择节点与多维观察状态；不从未来FI定义上游资格。"""
import numpy as np
import pandas as pd
from audit_sources import require

FINAL_GATES=['recovery_unavailable','landmark_fi_missing','landmark_timing_unavailable','no_later_fi',
             'no_timed_later_fi','fi_t_missing','prespecified_covariate_missing']


def pools(person):
    l0=person.recovery_classifiable.fillna(False)&person.lm_observed.eq(True).fillna(False)
    l1=l0&person.landmark_fi_available&person.lm_timing_valid
    l2=l1&person.fi_t_available&person.nine_covariates_complete
    return {'L0':l0,'L1':l1,'L2':l2,'Primary':person.primary_complete_case_eligible}


def failed_conditions(person):
    return [~person.recovery_classifiable,~person.landmark_fi_available,~person.lm_timing_valid,
            person.later_fi_count.lt(1),person.primary_later_fi_model_row_count.lt(1),
            ~person.fi_t_available,~person.nine_covariates_complete]


def first_failed(person):
    return np.select([c.to_numpy(bool) for c in failed_conditions(person)],FINAL_GATES,default='none')


def evidence_states(frame,cohort):
    """死亡编码按当前快照解读，不伪造当时已知或通知日期。"""
    status=pd.to_numeric(frame.scope_raw_value,errors='coerce')
    observed=frame.observed_in_wave.eq(True).fillna(False)
    ym=frame.interview_year.astype(float)*12+frame.interview_month.astype(float)
    dated_before=(frame.death_month_index.notna()&ym.notna()&(frame.death_month_index<ym))|(
        frame.death_year_only&frame.interview_year.notna()&(frame.radyear<frame.interview_year))
    death_code=status.isin([5,6])
    conflict=observed&(death_code|dated_before|frame.person_wave_scope_status.eq('post_death'))
    result=np.select([conflict,death_code,observed,status.eq(4)],
          ['SOURCE_CONFLICT','CONFIRMED_DEATH','LIVE_AT_ACTUAL_INTERVIEW','SOURCE_PRESUMED_ALIVE_NONRESPONSE'],
          default='DEATH_STATUS_UNKNOWN')
    frame=frame.copy();frame['vital_evidence']=result
    scope=frame.person_wave_scope_status
    frame['administrative_opportunity_frame']=np.select(
        [scope.isin(['administratively_unavailable','pre_entry']),scope.isin(['in_scope','post_death','post_permanent_exit'])],
        ['NO_ADMINISTRATIVE_OPPORTUNITY','PERSONAL_SCHEDULED_FRAME_PRESENT'],default='OPPORTUNITY_UNKNOWN')
    # 已知死亡/永久退出具有独立不具备FI观察资格标记，不被归成行政结束或普通缺失。
    frame['FI_observation_state']=np.select([
        frame.vital_evidence.eq('SOURCE_CONFLICT'),frame.vital_evidence.eq('CONFIRMED_DEATH'),
        scope.eq('post_permanent_exit'),observed&frame.fi_primary.notna(),observed&frame.fi_primary.isna(),
        status.eq(4)],['SOURCE_CONFLICT','DEATH_PRECLUDES_FI','PERMANENT_EXIT','INTERVIEW_VALID_FI',
          'INTERVIEW_INVALID_FI','PRESUMED_ALIVE_NONRESPONSE'],default='UNKNOWN')
    frame['mortality_coverage_gap']=(cohort=='ELSA')&frame.wave.ge(6)
    frame['confirmation_record_time_known']=False
    frame['actual_month_index']=ym
    return frame


def person_opportunity(group):
    if len(group)==0:return 'NO_ADMINISTRATIVE_POST_WAVE'
    if group.administrative_opportunity_frame.eq('NO_ADMINISTRATIVE_OPPORTUNITY').all():return 'NO_ADMINISTRATIVE_OPPORTUNITY'
    if group.administrative_opportunity_frame.eq('PERSONAL_SCHEDULED_FRAME_PRESENT').any():return 'PERSONAL_FRAME_PRESENT_NOT_PROOF_OF_SURVIVAL'
    return 'OPPORTUNITY_OR_PERSON_STATUS_UNKNOWN'


def summary_persons(post,person):
    rows=[]
    for pid,g in post.groupby('person_id',sort=False):
        g=g.sort_values('wave');valid=g.fi_primary.notna()&g.observed_in_wave.eq(True).fillna(False)
        timed=valid&g.time_years.gt(0).fillna(False)
        observed=g.observed_in_wave.eq(True).fillna(False)
        waves=g.wave.to_numpy();v=valid.to_numpy();o=observed.to_numpy()
        gap_return=False;interview_return=False
        if v.any():
            first,last=np.flatnonzero(v)[[0,-1]];gap_return=bool((~v[first:last+1]).any())
        if o.any():
            first,last=np.flatnonzero(o)[[0,-1]];interview_return=bool((~o[first:last+1]).any())
        times=g.loc[timed,'time_years'].to_numpy(float)
        rows.append({'person_id':pid,'post_wave_rows':len(g),'valid_post_FI':int(valid.sum()),'timed_post_FI':int(timed.sum()),
          'has_valid_post_FI':bool(valid.any()),'has_timed_post_FI':bool(timed.any()),
          'first_FI_time':float(times.min()) if len(times) else np.nan,
          'last_FI_time':float(times.max()) if len(times) else np.nan,
          'post_FI_gap_then_return':gap_return,'post_interview_gap_then_return':interview_return,
          'post_landmark_gap_then_FI':bool(v.any() and ((~v[:np.flatnonzero(v)[-1]+1]).any())),
          'any_post_interview_invalid_FI':bool(g.FI_observation_state.eq('INTERVIEW_INVALID_FI').any()),
          'no_post_FI_but_any_post_interview':bool(not valid.any() and observed.any()),
          'known_death_code_after_landmark':bool(g.vital_evidence.eq('CONFIRMED_DEATH').any()),
          'any_presumed_alive_nonresponse':bool(g.vital_evidence.eq('SOURCE_PRESUMED_ALIVE_NONRESPONSE').any()),
          'any_unknown_vital_status':bool(g.vital_evidence.eq('DEATH_STATUS_UNKNOWN').any()),
          'any_conflicting_source':bool(g.vital_evidence.eq('SOURCE_CONFLICT').any()),
          'any_permanent_exit':bool(g.person_wave_scope_status.eq('post_permanent_exit').any()),
          'any_confirmed_live_post_interview':bool(g.vital_evidence.eq('LIVE_AT_ACTUAL_INTERVIEW').any()),
          'administrative_opportunity_class':person_opportunity(g)})
    result=pd.DataFrame(rows)
    require(not result.person_id.duplicated().any(),'个人后续汇总重复')
    merged=person.merge(result,on='person_id',how='left',validate='one_to_one')
    zero=merged.post_wave_rows.isna()
    for name in ['post_wave_rows','valid_post_FI','timed_post_FI']:merged.loc[zero,name]=0
    for name in result.columns:
        if name not in ['person_id','post_wave_rows','valid_post_FI','timed_post_FI','first_FI_time','last_FI_time','administrative_opportunity_class']:
            merged.loc[zero,name]=False
    merged.loc[zero,'administrative_opportunity_class']='NO_ADMINISTRATIVE_POST_WAVE'
    return merged
