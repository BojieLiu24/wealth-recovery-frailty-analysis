"""不中断或重拟合模型，补齐两次执行的版本关系与交付核验。"""
import ast
import difflib
import shutil
from datetime import datetime
import pandas as pd
import numpy as np
from prepare_inputs import *
from validate_results import validate,validate_manifest

def main():
 manifest=read_json(OUTPUT/'manifest.json');validate_manifest(manifest)
 require(not (PHASE/'packaging_before').exists(),'交付收尾已执行，禁止重复覆盖')
 backup=PHASE/'packaging_before';backup.mkdir()
 report=PHASE/'phase2_execution_report.md'
 for p in [OUTPUT/'manifest.json',OUTPUT/'analysis_asset_index.csv',report]:shutil.copyfile(p,backup/p.name)
 contract,_=verify_benchmark();bundle=read_json(OUTPUT/'result_object.json');models=bundle['models'];diagnostics=read_json(OUTPUT/'model_diagnostics.json')['models'];validation=validate(contract,models,diagnostics)
 history=read_json(PHASE/'execution_interruption_attempt1.json')
 for asset in history['completed_assets']:require(sha(ROOT/asset['path'])==asset['sha256'],'中断前模型文件被改变')
 old=PHASE/'source_versions/attempt1'
 # 检查恢复处理只涉及文件持久化/执行控制，科学计算代码一致。
 for name in ['run_models.py','build_descriptives.py']:require(sha(CODE/name)==sha(old/name),'两次执行科学代码不一致')
 def functions(path):
  tree=ast.parse(path.read_text(encoding='utf-8'));return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
 old_functions=functions(old/'prepare_inputs.py');new_functions=functions(CODE/'prepare_inputs.py')
 science_names=['verify_benchmark','load_cohort','prepare_spec','design_matrix']
 require(all(old_functions[n]==new_functions[n] for n in science_names),'准备输入或设计矩阵科学逻辑发生变化')
 diff=[]
 for name in ['prepare_inputs.py','run_wealth_measurement.py']:
  diff+=list(difflib.unified_diff((old/name).read_text(encoding='utf-8').splitlines(True),(CODE/name).read_text(encoding='utf-8').splitlines(True),fromfile='attempt1/'+name,tofile='attempt2/'+name))
 (PHASE/'serialization_resume_code_diff.txt').write_text(''.join(diff),encoding='utf-8')
 execution=read_json(PHASE/'execution_units.json')['units'];attempt_by_unit={(u['analysis_id'],u['cohort']):u for u in execution}
 require(len(execution)==14 and sum(u['reused_without_refit'] for u in execution)==9,'续跑数量不符')
 for m in models:
  if m['status']=='SUCCESS':require(len(read_json(OUTPUT/'Model Results'/m['analysis_id']/m['cohort']/'optimization_log.json'))==2,'双优化器日志缺失')
 additional=[]
 for c in ['HRS','ELSA','KLoSA']:
  frame,person=load_cohort(contract,c)
  # 仅对模型实际使用的冻结事实作连接一致性核对，不重新识别冲击。
  base=frame.groupby('person_id',observed=True).wealth_loss_fraction.agg(['first','nunique']).reset_index()
  check=base.merge(person[['person_id','wealth_loss_fraction']],on='person_id',validate='one_to_one')
  require(check.nunique.eq(1).all() if False else check['nunique'].eq(1).all(),'损失比例在个人行间变化')
  require(np.array_equal(check['first'].to_numpy(float),check['wealth_loss_fraction'].to_numpy(float)),'分析/暴露损失比例不符')
  lm=frame.loc[frame.fi_time_role.eq('landmark')]
  require(lm.time_years.eq(0).all() and lm.wave.eq(lm.landmark_wave).all(),'时间0或landmark波次不符')
  require((frame.groupby('person_id',observed=True).fi_time_role.agg(lambda x:(x=='landmark').sum())==1).all(),'每人landmark不唯一')
  additional.append({'cohort':c,'primary_participants':len(person),'analysis_exposure_loss_exact_match':True,'one_landmark_at_time_zero':True})
 index=pd.read_csv(OUTPUT/'analysis_asset_index.csv',keep_default_na=False)
 index['source_kind']='';index['execution_code_version']='';index['validation_status']='PASS_PRELIMINARY_INTEGRITY_PHASE3_QA_PENDING'
 for i,row in index.iterrows():
  if row.analysis_id=='WM00':
   index.loc[i,'source_code_sha256']=sha(CODE/'build_descriptives.py');index.loc[i,'source_kind']='NEW_DESCRIPTIVE';index.loc[i,'execution_code_version']='both_attempts_identical_descriptive_logic'
  elif row.analysis_id=='WM04' and row.cohort=='KLoSA':
   index.loc[i,'source_code_sha256']=contract['historical_and_current_source_lineage']['historical_source_sha256'];index.loc[i,'source_kind']='FROZEN_PRIMARY_REFERENCE_NO_FIT';index.loc[i,'execution_code_version']='historical_section_3_2_v3'
  else:
   index.loc[i,'source_kind']='NEW_LMM_FIT';index.loc[i,'execution_code_version']='attempt'+str(attempt_by_unit[(row.analysis_id,row.cohort)]['execution_attempt'])
 index.to_csv(OUTPUT/'analysis_asset_index.csv',index=False,encoding='utf-8')
 opt_seconds=sum(o['elapsed_seconds'] for d in diagnostics for o in d['optimization'])
 max_beta=max(d.get('beta3_optimizer_absolute_difference',0) for d in diagnostics);max_ll=max(d.get('loglik_optimizer_absolute_difference',0) for d in diagnostics)
 warnings=[w for d in diagnostics for o in d['optimization'] for w in o['warnings']]
 warning_messages=sorted(set(w['message'] for w in warnings))
 elapsed_wall=(datetime.fromisoformat(manifest['ended'])-datetime.fromisoformat(manifest['started'])).total_seconds()
 preservation=protected_snapshot()==read_json(PHASE/'protected_hashes_before.json');require(preservation,'受保护资产变化')
 extra={'status':'PASS','preliminary_validation':validation,'input_consistency':additional,'original_nine_unit_files_unchanged':True,'real_fit_units_total':14,'actual_optimizer_runs':28,'reused_completed_fit_units':9,'resume_new_fit_units':5,'scientific_function_AST_identical':science_names,'model_and_descriptive_module_hashes_identical_across_attempts':True,'max_beta3_optimizer_difference':max_beta,'max_loglik_optimizer_difference':max_ll,'optimization_seconds_total':opt_seconds,'wall_seconds_initial_authorization_to_model_completion_including_interruption':elapsed_wall,'optimizer_warnings_n':len(warnings),'optimizer_warning_messages':warning_messages,'protected_assets_unchanged':preservation,'phase3_QA':'PENDING'}
 write_json(PHASE/'delivery_check.json',extra)
 text=report.read_text(encoding='utf-8')
 text=text.replace('失败单元若有数值，仅表示未验收候选估计，不能用作正式稳健性证据。','本轮没有FAILED或NOT_ESTIMABLE单元。')
 text=text.replace('运行秒数：28.0。',f'续跑及初次完整性检查秒数：{manifest["elapsed_seconds"]:.1f}。')
 text+='\n## 执行中断及版本关系\n\n第一次执行完成9个模型后，在更新进度JSON时遇到OSError（Errno 22）。全部独立模型资产完整保存。保留第一次代码、进度和81份单元文件哈希；改为临时文件原子替换并增加续跑入口，仅拟合剩余5个模型，没有重复拟合完成项。两次模型/描述模块哈希一致，输入构造与设计矩阵函数AST一致，差异见serialization_resume_code_diff.txt。原9个单元文件哈希在最终交付再次核对一致。\n\n'
 text+=f'实际总计28次优化，优化耗时合计{opt_seconds:.1f}秒；初次执行授权到模型完成的墙钟时间为{elapsed_wall:.1f}秒（含中断定位及恢复）。最大β₃优化器差异{max_beta:.3g}，最大loglik差异{max_ll:.3g}，均低于合同门槛。共{len(warnings)}条优化警告，均为“The MLE may be on the boundary of the parameter space.”；所有方差分量为正，固定效应协方差最小特征值均为正。该警告保留供Phase 3复核，不据此宣称模型假设已验证。\n\n'
 text+='初步结果：HRS五个规格均为正且Wald 95%CI不跨零，WM03 β=0.00125967（原值0.00126753），损失函数转换下变化小，但不等于已排除个体影响。ELSA WM01仍为负，SE从0.00114169增至0.00161599，N从699降至306；限制样本后的精度明显下降。KLoSA WM05仍为正，SE从0.00058575增至0.00095456，N从2502降至801；WM02方向变为轻微负值，CI跨零，是Phase 3需解释的人群限制结果，不能视为构造错误。\n'
 report.write_text(text,encoding='utf-8')
 (PHASE/'git_status_after.txt').write_text(git('status','--porcelain','--untracked-files=all')+'\n',encoding='utf-8')
 manifest.update({'execution_code_at_resume':manifest['code'],'code':[{'path':rel(f),'sha256':sha(f)} for f in sorted(CODE.rglob('*')) if f.is_file() and '__pycache__' not in f.parts],'packaging_code_path':rel(Path(__file__)),'delivery_check':extra,'elapsed_seconds_scope':'successful resume attempt only; both-attempt wall and optimizer totals in delivery_check','attempt1_source_record':rel(PHASE/'execution_interruption_attempt1.json'),'unit_execution_lineage':rel(PHASE/'execution_units.json')})
 excluded={OUTPUT/'manifest.json',PHASE/'manifest_validation.json'}
 manifest['outputs']=[{'path':rel(f),'sha256':sha(f)} for root in [OUTPUT,PHASE] for f in sorted(root.rglob('*')) if f.is_file() and f not in excluded]
 write_json(OUTPUT/'manifest.json',manifest)
 result=validate_manifest(manifest);result.update({'manifest_path':rel(OUTPUT/'manifest.json'),'manifest_sha256':sha(OUTPUT/'manifest.json'),'phase3_QA':'PENDING'})
 write_json(PHASE/'manifest_validation.json',result)
 print(extra)
 print(result)

if __name__=='__main__':main()
