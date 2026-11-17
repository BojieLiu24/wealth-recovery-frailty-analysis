"""HRS/PO01固定候选处的独立解析复核；不调用任何优化器。"""
import numpy as np
from scipy.linalg import solve_triangular


def independent_derivatives(y,x,ids,beta,q,sigma):
    """用随机截距Woodbury恒等式推导ML似然、score及两种参数化Hessian。"""
    y,x,beta=map(lambda a:np.asarray(a,dtype=float),(y,x,beta))
    ids=np.asarray(ids);n,p=x.shape
    if not np.isfinite(q) or q<0 or not np.isfinite(sigma) or sigma<=0:
        raise ValueError('方差非法，不裁剪')
    r=y-x@beta
    _,codes=np.unique(ids,return_inverse=True)
    order=np.argsort(codes,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(codes[order]))+1,n]
    Q,U,F,traceJ,E,traceC,traceB2,R_C_B,R_B3,logdet=([0.]*10)
    A=np.zeros((p,p));h=np.zeros(p);D=np.zeros(p)
    for a,b in zip(starts[:-1],starts[1:]):
        index=order[a:b];ni=len(index);ri=r[index];xi=x[index]
        den=1+ni*q;s=float(ri.sum());xs=xi.sum(axis=0);r2=float(ri@ri)
        Q+=r2-q*s*s/den;U+=s*s/den**2;F+=ni*s*s/den**3
        traceJ+=ni/den;E+=ni*ni/den**2;traceC+=ni/den**2
        traceB2+=ni-1+1/den**2;R_C_B+=s*s/den**3
        R_B3+=r2-s*s/ni+s*s/(ni*den**3);logdet+=np.log1p(ni*q)
        A+=xi.T@xi-q*np.outer(xs,xs)/den
        h+=xi.T@ri-q*xs*s/den;D+=xs*s/den**2
    if Q<=0:raise ValueError('残差平方型非法')
    sigma_profile=Q/n
    ll_profile=float(-.5*(n*np.log(2*np.pi)+n*np.log(sigma_profile)+n+logdet))
    score_profile=np.r_[n*h/Q,-.5*traceJ+n*U/(2*Q)]
    H=np.empty((p+1,p+1))
    H[:p,:p]=-n*A/Q+2*n*np.outer(h,h)/Q**2
    H[:p,p]=H[p,:p]=-n*D/Q+n*h*U/Q**2
    H[p,p]=.5*E-n*F/Q+n*U*U/(2*Q**2)
    # 再独立构建未剖面化的(β,τ²,σ²)观察信息，不以条件GLS协方差代替CI。
    ll_full=float(-.5*(n*np.log(2*np.pi)+n*np.log(sigma)+logdet+Q/sigma))
    B2_r=Q-q*U;traceB=n-q*traceJ
    score_full=np.r_[h/sigma,.5*(U/sigma**2-traceJ/sigma),.5*(B2_r/sigma**2-traceB/sigma)]
    full=np.empty((p+2,p+2));full[:p,:p]=-A/sigma
    full[:p,p]=full[p,:p]=-D/sigma**2
    full[:p,p+1]=full[p+1,:p]=-(h-q*D)/sigma**2
    full[p,p]=.5*E/sigma**2-F/sigma**3
    full[p,p+1]=full[p+1,p]=.5*traceC/sigma**2-R_C_B/sigma**3
    full[p+1,p+1]=.5*traceB2/sigma**2-R_B3/sigma**3
    return {'loglik_profile':ll_profile,'loglik_full':ll_full,'sigma_profile':sigma_profile,
        'score_profile':score_profile,'hessian_profile':H,'score_full':score_full,
        'hessian_full':full,'parameterization_profile':'beta and q=tau2/sigma2; sigma2 profiled by Q/n',
        'parameterization_full':'beta, tau2, sigma2; all variance nuisance parameters included'}


def inverse_information(hessian):
    info=-np.asarray(hessian,dtype=float);diagonal=np.diag(info)
    if not np.isfinite(info).all() or (diagonal<=0).any():
        raise ValueError('观察信息对角非法')
    d=np.sqrt(diagonal);balanced=info/d[:,None]/d[None,:]
    factor=np.linalg.cholesky(balanced)
    inverse=solve_triangular(factor.T,solve_triangular(factor,np.eye(len(info)),lower=True),lower=False)
    inverse=inverse/d[:,None]/d[None,:]
    return inverse,{'balanced_eigenvalues':np.linalg.eigvalsh(balanced).tolist(),
        'balanced_condition_number':float(np.linalg.cond(balanced)),
        'inverse_backward_error':float(np.max(abs(info@inverse-np.eye(len(info))))),
        'method':'balanced_Cholesky_no_pseudoinverse_no_clipping'}


def dense_loglik(y,x,ids,beta,q,sigma=None):
    """独立密集Cholesky路径，仅用于小型合成导数测试。"""
    ids=np.asarray(ids);W=np.eye(len(ids))+q*(ids[:,None]==ids[None,:])
    factor=np.linalg.cholesky(W);r=np.linalg.solve(factor,np.asarray(y)-np.asarray(x)@np.asarray(beta))
    Q=r@r;s=Q/len(ids) if sigma is None else sigma
    return float(-.5*(len(ids)*np.log(2*np.pi)+len(ids)*np.log(s)+2*np.log(np.diag(factor)).sum()+Q/s))
