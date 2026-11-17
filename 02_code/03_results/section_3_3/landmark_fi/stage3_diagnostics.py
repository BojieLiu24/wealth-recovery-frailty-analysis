"""条件残差、真实观察支持与固定协方差影响筛查；仅输出聚合。"""
import numpy as np
from scipy.stats import skew, kurtosis, norm
from lf_common import require


def summary(values):
    a=np.asarray(values,float)
    if len(a)==0:return {'n':0}
    return {'n':len(a),'mean':float(a.mean()),'sd':float(a.std(ddof=1)) if len(a)>1 else None,
            'min':float(a.min()),'max':float(a.max()),'skewness':float(skew(a,bias=False)) if len(a)>2 else None,
            'excess_kurtosis':float(kurtosis(a,bias=False)) if len(a)>3 else None,
            'quantile_probabilities':[.01,.05,.25,.5,.75,.95,.99],
            'quantiles':np.quantile(a,[.01,.05,.25,.5,.75,.95,.99]).tolist()}


def correlation(a,b):
    return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None


def support_and_residuals(frame,block,sigma,tau):
    t=frame.time_years.to_numpy(float);r=block['conditional_residual'];f=block['conditional_fitted']
    y=frame.fi_primary.to_numpy(float);group=frame.recovery.to_numpy(int)
    require((t>0).all(),'post-only诊断含非正时间')
    bins=[];counts=[];groups=[];deciles=[]
    for value,label in [(0,'persistent_or_limited_recovery'),(1,'substantial_recovery')]:
        mask=group==value;part=frame.loc[mask]
        n=part.groupby('person_id').size()
        counts.extend({'recovery_status':label,'observations_per_person':int(k),'participants':int(v)}
                      for k,v in n.value_counts().sort_index().items())
        tail=mask&(t>8)
        groups.append({'recovery_status':label,'participants':len(n),'observations':int(mask.sum()),
             'single_observation_participants':int((n==1).sum()),'time':summary(t[mask]),
             'participant_max_followup':summary(part.groupby('person_id').time_years.max()),
             'over_8_year_participants':int(frame.loc[tail,'person_id'].nunique()),'over_8_year_rows':int(tail.sum()),
             'FI_distribution':summary(y[mask]),'conditional_residual_distribution':summary(r[mask]),
             'conditional_fitted_range':summary(f[mask]),'conditional_fitted_outside_0_1_rows':int(((f<0)|(f>1))[mask].sum()),
             'fixed_fitted_outside_0_1_rows':int(((block['fixed_fitted']<0)|(block['fixed_fitted']>1))[mask].sum()),
             'residual_time_correlation':correlation(r[mask],t[mask]),
             'residual_fitted_correlation':correlation(r[mask],f[mask]),
             'absolute_residual_fitted_correlation':correlation(abs(r[mask]),f[mask])})
        for lo in np.arange(0,np.ceil(float(t.max())/2)*2,2):
            m=mask&(t>lo)&(t<=lo+2);people=int(frame.loc[m,'person_id'].nunique())
            bins.append({'recovery_status':label,'lower_exclusive':float(lo),'upper_inclusive':float(lo+2),
               'participants':people,'observations':int(m.sum()),'sparse_display_only':people<30,
               'residual':summary(r[m]),'FI':summary(y[m]),'conditional_fitted':summary(f[m])})
        # 拟合值十分位分箱仅用聚合残差描述，不改变任何模型。
        edges=np.quantile(f[mask],np.linspace(0,1,11))
        for j in range(10):
            m=mask&(f>=edges[j])&((f<=edges[j+1]) if j==9 else (f<edges[j+1]))
            deciles.append({'recovery_status':label,'decile':j+1,'lower':float(edges[j]),'upper':float(edges[j+1]),
                           'participants':int(frame.loc[m,'person_id'].nunique()),'residual':summary(r[m])})
    extreme=abs(r)>3*np.sqrt(sigma)
    tail_bins=[]
    for b in bins:
        value=1 if b['recovery_status']=='substantial_recovery' else 0
        m=extreme&(group==value)&(t>b['lower_exclusive'])&(t<=b['upper_inclusive'])
        tail_bins.append({'recovery_status':b['recovery_status'],'lower_exclusive':b['lower_exclusive'],
                         'upper_inclusive':b['upper_inclusive'],'extreme_rows':int(m.sum()),
                         'extreme_participants':int(frame.loc[m,'person_id'].nunique())})
    probs=[.001,.01,.025,.05,.25,.5,.75,.95,.975,.99,.999]
    standard=r/r.std(ddof=1)
    out={'conditional_residual_distribution':summary(r),'FI_distribution':summary(y),
         'QQ_quantiles':{'probabilities':probs,'Gaussian':norm.ppf(probs).tolist(),
                         'empirical_residual_standardized_by_empirical_SD':np.quantile(standard,probs).tolist()},
         'groups':groups,'two_year_bins':bins,'observation_count_distribution':counts,'fitted_value_deciles':deciles,
         'extreme_threshold':'abs(conditional residual)>3*sqrt(saved residual variance); descriptive only',
         'extreme_residual_time_bins':tail_bins,'extreme_rows':int(extreme.sum()),
         'residual_time_correlation':correlation(r,t),'residual_fitted_correlation':correlation(r,f),
         'absolute_residual_fitted_correlation':correlation(abs(r),f),
         'random_intercept_variance':tau,'residual_variance':sigma,
         'ICC':tau/(tau+sigma),'counts_are_support_not_survival_or_risk_sets':True,
         'diagnostic_8_year_boundary_changes_model':False,'participant_or_row_data_exported':False,
         'Gaussian_comment':'FI偏态不等于条件残差偏态；小相关不证明均值线性或方差恒定。'}
    return out


def influence_screen(y,x,ids,time,group,beta,tau,sigma,core,se):
    """按合同使用固定V下的个人固定效应得分；不重估任何参数。"""
    y,x,ids,time,group=map(np.asarray,(y,x,ids,time,group));p=x.shape[1]
    residual=y-x@beta
    _,code=np.unique(ids,return_inverse=True);order=np.argsort(code,kind='stable')
    starts=np.r_[0,np.flatnonzero(np.diff(code[order]))+1,len(y)]
    information=np.zeros((p,p));row_scores=np.zeros_like(x)
    index_blocks=[]
    for a,b in zip(starts[:-1],starts[1:]):
        index=order[a:b];xi=x[index];ri=residual[index];n=len(index)
        den=sigma+n*tau
        vir=(ri-tau/den*ri.sum())/sigma
        vix=(xi-tau/den*xi.sum(axis=0))/sigma
        information+=xi.T@vix;row_scores[index]=xi*vir[:,None];index_blocks.append(index)
    require(np.linalg.eigvalsh(information).min()>0,'影响筛查条件信息奇异')
    # 此条件GLS逆只用于影响筛查，绝不能替代完整观察信息的LMM SE。
    target=np.zeros(p);target[core]=1
    direction=np.linalg.solve(information,target)
    row_influence=row_scores@direction
    u=np.array([row_influence[i].sum() for i in index_blocks])
    maxima=np.array([time[i].max() for i in index_blocks])
    person_group=np.array([group[i[0]] for i in index_blocks]);absolute=abs(u);total=absolute.sum()
    ranked=np.argsort(-absolute,kind='stable')
    concentration=[]
    for n in sorted(set([1,5,10,int(np.ceil(len(u)*.01)),int(np.ceil(len(u)*.05))])):
        index=ranked[:n]
        concentration.append({'top_participants':n,'absolute_influence_share':float(absolute[index].sum()/total),
              'signed_sum':float(u[index].sum()),'max_followup_summary':summary(maxima[index]),
              'over_8_year_participants':int((maxima[index]>8).sum())})
    strata=[]
    for value in [0,1]:
        for tail in [False,True]:
            m=(person_group==value)&((maxima>8)==tail)
            strata.append({'recovery':value,'has_observation_over_8':tail,'participants':int(m.sum()),
                           'signed_sum':float(u[m].sum()),'absolute_influence_share':float(absolute[m].sum()/total),
                           'max_abs_influence_over_formal_SE':float(absolute[m].max()/se) if m.any() else None})
    return {'method':'e_RT.T @ inv(sum(Xi.T Vi^-1 Xi)) @ Xi.T Vi^-1 ri',
       'fixed_covariance_first_order_only':True,'variance_components_reestimated':False,'leave_one_out_refit':False,
       'is_formal_LMM_SE':False,'person_identifiers_exported':False,'participants':len(u),
       'sum_person_score_influences':float(u.sum()),'max_abs_influence':float(absolute.max()),
       'max_abs_influence_over_formal_SE':float(absolute.max()/se),'absolute_influence_total':float(total),
       'top_concentration':concentration,'followup_group_strata':strata,
       'influence_abs_max_followup_correlation':correlation(absolute,maxima),
       'over_8_row_signed_influence':float(row_influence[time>8].sum()),
       'over_8_row_absolute_influence_share':float(abs(row_influence[time>8]).sum()/abs(row_influence).sum()),
       'limitation':'固定协方差一阶近似不能排除截短随访敏感性，不据此删人或证明尾部无影响。'}
