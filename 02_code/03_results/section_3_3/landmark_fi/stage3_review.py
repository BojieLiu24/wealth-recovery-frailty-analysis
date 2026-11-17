"""逐步比较及跨模块科学审阅；不得把描述差异当作因果分解。"""
from lf_common import ROOT, read_json, require
from lf_spec import LM_STATEMENT


def compare_rows(a,b,allow_fit_metrics=False):
    same=a['ordered_key_sha256']==b['ordered_key_sha256'] and a['FI_observations']==b['FI_observations']
    require(not allow_fit_metrics or same,'不同结局窗口禁止直接比较拟合指标')
    result={'from':a['specification'],'to':b['specification'],'same_outcome_rows':same,
       'estimate_change':b['estimate']-a['estimate'],'SE_change':b['se']-a['se'],
       'SE_ratio':b['se']/a['se'],'CI_width_ratio':(b['ci_upper']-b['ci_lower'])/(a['ci_upper']-a['ci_lower']),
       'from_CI':[a['ci_lower'],a['ci_upper']],'to_CI':[b['ci_lower'],b['ci_upper']],
       'FI_observation_change':b['FI_observations']-a['FI_observations'],
       'interpretation':'descriptive_only_not_independent_estimates_no_difference_test',
       'fit_metric_comparison_allowed':allow_fit_metrics and same}
    if allow_fit_metrics:
        result['LM_minus_PO_loglik']=b['loglik']-a['loglik']
        require(result['LM_minus_PO_loglik']>=-1e-4,'同行集嵌套LM似然低于PO超过容差')
        result['AIC_change_auxiliary_only']=-2*result['LM_minus_PO_loglik']+4
    return result


def round1b_reference():
    base=ROOT/'03_output/3.3 Robustness of the Main Findings/releases/model_robustness_round1b_v1'
    result=[]
    for cohort in ['HRS','ELSA','KLoSA']:
        for spec,folder,term in [('RS01','Random Slope','S'),('NL01','Nonlinear Time','A_0_8')]:
            obj=read_json(base/'Model Results'/folder/spec/cohort/'result_object.json')
            row=next(r for r in obj['contrasts'] if r['estimand_id']==term and (term=='A_0_8' or r['time_years']==0))
            result.append({'cohort':cohort,'specification':spec,'estimand':term,
                           **{k:row[k] for k in ['estimate','se','ci_lower','ci_upper']}})
    return result


def estimand_axis(module):
    return {'Round1B':'random_effect_structure_and_time_function',
            'Round1C':'outcome_window_and_landmark_FI_conditioning'}[module]


def number(v):return f'{v:+.6f}'


def reports(validations,comparisons,reference,overall):
    scientific=['# Round 1C — 科学解释（作者审阅稿）','',
      f'独立审阅建议：`{overall}`。仅为Stage3审阅建议，最终科学批准及Stage4冻结仍待作者决定。','',
      '按冻结Primary→PO01→LM01顺序解释，单位为FI units/year。模型间差异仅作描述，不对相关估计作独立样本差异检验。','']
    for cohort in ['HRS','ELSA','KLoSA']:
        c=comparisons[cohort]; rows=c['models']
        scientific.extend([f'## {cohort}',''])
        for r in rows:
            if 'estimate' in r:
                scientific.append(f"- {r['specification']}：β={number(r['estimate'])}，SE={r['se']:.6f}，95%CI [{number(r['ci_lower'])}, {number(r['ci_upper'])}]；{r['Participants']:,}人、{r['FI_observations']:,}条FI。")
        scientific.append('')
        if not c['complete_chain']:
            scientific.append('核心检查未通过，递进比较存在证据缺口；不将该候选估计视为有效结果。')
            continue
        first,second=c['steps']
        scientific.extend([f"Primary→PO01：点估计改变{number(first['estimate_change'])}，SE为原来的{first['SE_ratio']:.3f}倍，CI宽度为原来的{first['CI_width_ratio']:.3f}倍，减少{-first['FI_observation_change']:,}条landmark结局行，参与者保留。该变化与结局观察窗口改变相联系，不能量化landmark行的因果贡献。",'',
           f"PO01→LM01：同行集下点估计改变{number(second['estimate_change'])}，SE为PO01的{second['SE_ratio']:.3f}倍，CI宽度为PO01的{second['CI_width_ratio']:.3f}倍。此步估计变化绝对值{'小于' if abs(second['estimate_change'])<abs(first['estimate_change']) else '不小于'}前一步；这一描述不依赖P值或任意百分比门槛。",''])
        if cohort=='HRS':
            scientific.extend(['HRS/PO01需附作者数值例外及原L-BFGS未收敛限制；在限定性数值QA支持下，PO01与LM01点估计均为正且CI不跨零。landmark健康条件化对本轮线性随机截距估计的增量影响较小，但不能据此消除Round1B显示的规格敏感性。',''])
        else:
            scientific.extend(['主要可见变化发生在Primary→PO01，PO01→LM01点估计和不确定性较接近。不能据此证明不存在健康相关选择、反向因果或其他偏倚。',''])
    scientific.extend(['## 解释边界','',LM_STATEMENT,'',
       'Landmark FI与财富恢复在同次访谈测量，可能同时包含既往健康与冲击后的健康变化。LM01是条件关联，不称为更充分控制混杂的因果模型；过度调整、碰撞偏倚及其他选择问题仍需限定。','',
       'PO01/LM01所有结局在T>0，截距及恢复主效应为向时间0外推的参数。本轮仅解释后续年度速度差，不解释实测landmark水平差，不重做Figure3。','',
       '不直接比较Primary与PO01的AIC、BIC或loglik；同行集PO01与LM01的拟合改善仅为辅助证据，不决定Primary选用。没有识别中介效应、反向因果贡献或混杂消除比例。','',
       '以下限制须保留：Gaussian条件残差的偏态/重尾、稀疏远期观察，以及固定协方差影响筛查的不完备性。无新增截短或删除个体模型，不能宣称远期观察敏感性已充分排除。'])
    rb=['# Round 1B与Round 1C证据一致性审查','',
        '已核对model_robustness_round1b_v1 Freeze及包内清单，状态FROZEN_WITH_CAVEATS。没有重新拟合Round1B。','',
        'Round1B检验随机效应结构（RS01）和时间函数（NL01）；Round1C检验结局观察窗口（PO01）和landmark FI条件化（LM01）。两轮回答不同的规格问题。','']
    for r in reference:
        rb.append(f"- {r['cohort']}/{r['specification']}（{r['estimand']}）：β={number(r['estimate'])}，SE={r['se']:.6f}，95%CI [{number(r['ci_lower'])}, {number(r['ci_upper'])}]。")
    rb.extend(['','HRS的RS01年度速度差SE约为Primary的1.96倍，NL01的0–8年平均速度差SE也增加，两者CI均跨零。NL01平均速度差与恒定年度交互不是完全相同的估计对象。','',
       'Round1C的HRS线性随机截距条件关联即使为正且CI不跨零，也不代表Round1B的不确定性问题消失。两轮结果可以同时成立：本轮对landmark条件化相对稳定，而对随机斜率和非线性时间仍敏感。','',
       '没有拟合随机斜率×landmark FI或非线性×landmark FI联合模型，不能声称排除了两类规格敏感性的共同影响。不得依据哪轮更显著选择证据。Primary宜保持观察性解释，并同时陈述Round1B的不确定性扩大与Round1C的窗口/条件化证据。'])
    exceptions=['# Round 1C — 例外及作者决策','',
       'HRS/PO01：原Stage2 HOLD_NOT_ESTIMABLE保留；作者已限定批准AUTHOR_ACCEPTED_NUMERICAL_EXCEPTION，仅准入Stage3。Stage3独立数值结论见model_qa_matrix.csv及independent_validation.json。','',
       'BFGS收敛，L-BFGS未报告收敛；原准确SciPy终止原因无法还原。不得声称所有锁定优化条件都已满足，不推断具体线搜索或舍入原因，也不声称全球最优。限定推荐以保存参数处的独立似然、score、Hessian及完整观察信息推断为依据。','',
       '其余单元的statsmodels绝对0.01 boundary warning须与真实方差比区分；若随机方差为合法内点且完整信息可用，warning本身不等于边界。','',
       '所有单元的Gaussian残差、时间支持及影响近似限制均进入科学解释。仅保存聚合统计；没有逐人残差、BLUP或参与者身份输出。','',
       f'当前建议：{overall}。请作者审阅具体结果与限制后，决定是否接受HRS/PO01限定性结果以及整个Round1C科学结果、是否授权Stage4。此报告不替代批准。','',
       '如作者认为需进一步排除远期支持敏感性或联合规格影响，应单独批准新的分析；本轮没有新增模型。']
    return {'scientific_interpretation.md':'\n'.join(scientific)+'\n',
            'round1b_consistency_review.md':'\n'.join(rb)+'\n',
            'exceptions_and_decisions.md':'\n'.join(exceptions)+'\n'}
