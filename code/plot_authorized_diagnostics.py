"""从本地授权残差表绘制S2所述诊断；不保存或分发个体输入。"""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import probplot
from statsmodels.nonparametric.smoothers_lowess import lowess


def main():
    parser=argparse.ArgumentParser(description='Mode B only: reconstructed primary residual diagnostics, not a pixel-identical Figure S2 layout.')
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    if args.input.resolve().is_relative_to(root):
        raise ValueError('Individual-level diagnostic inputs must remain outside this repository.')
    data=pd.read_csv(args.input)
    required={'cohort','recovery_status','time_years','fitted_fi','conditional_residual'}
    if not required.issubset(data.columns):raise ValueError(f'Required columns: {sorted(required)}')
    fig,axes=plt.subplots(3,3,figsize=(10,9),constrained_layout=True)
    for i,cohort in enumerate(['HRS','ELSA','KLoSA']):
        sub=data[data.cohort==cohort]
        if len(sub)<3:raise ValueError(f'{cohort}: insufficient diagnostic observations.')
        for group,color,marker in [('substantial_recovery','#007C91','o'),('persistent_or_limited_recovery','#D95F02','x')]:
            g=sub[sub.recovery_status==group]
            if len(g)<3:raise ValueError(f'{cohort}/{group}: insufficient observations.')
            for j,col in enumerate(['time_years','fitted_fi']):
                axes[i,j].scatter(g[col],g.conditional_residual,s=4,alpha=.25,color=color,marker=marker)
                smooth=lowess(g.conditional_residual,g[col],return_sorted=True)
                axes[i,j].plot(smooth[:,0],smooth[:,1],color=color)
            (theoretical,observed),(slope,intercept,_)=probplot(g.conditional_residual)
            axes[i,2].scatter(theoretical,observed,s=4,color=color,marker=marker)
            axes[i,2].plot(theoretical,slope*np.asarray(theoretical)+intercept,color=color)
        axes[i,0].set_ylabel(f'{cohort}\nConditional residual')
        for j,label in enumerate(['Years from landmark','Conditional fitted FI','Normal theoretical quantile']):
            axes[i,j].set_xlabel(label)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=600)
    plt.close(fig)


if __name__=='__main__':main()
