"""HRS/SP01保存候选的独立数值核查；不调用统计软件似然、导数或优化器。"""
from contextlib import contextmanager
import numpy as np
from scipy.linalg import solve_triangular


@contextmanager
def forbid_fitting(counter):
    """阻断公共及本次相关私有拟合接口，退出或异常时恢复。"""
    import scipy.optimize as opt
    import scipy.optimize._optimize as bfgs
    import scipy.optimize._lbfgsb_py as lbfgs
    import statsmodels.api as sm
    import statsmodels.base.optimizer as dispatch
    targets = [(cls, 'fit') for cls in (sm.MixedLM, sm.OLS, sm.GLM, sm.GEE)]
    targets += [(opt, n) for n in ('minimize', 'minimize_scalar', 'fmin', 'fmin_bfgs',
        'fmin_l_bfgs_b', 'fmin_cg', 'fmin_ncg', 'fmin_powell', 'least_squares',
        'leastsq', 'root', 'differential_evolution', 'basinhopping', 'dual_annealing', 'shgo')]
    targets += [(bfgs, '_minimize_bfgs'), (lbfgs, '_minimize_lbfgsb'),
        (dispatch, '_fit_bfgs'), (dispatch, '_fit_lbfgs')]
    original = [(obj, name, getattr(obj, name)) for obj, name in targets]
    def blocked(*args, **kwargs):
        counter['forbidden_attempts'] += 1
        raise RuntimeError('TARGETED_CHECK_FORBIDS_FIT_AND_OPTIMIZATION')
    try:
        for obj, name, _ in original:
            setattr(obj, name, blocked)
        yield
    finally:
        for obj, name, func in original:
            setattr(obj, name, func)


class Blocks:
    """仅在内存中分组；充分统计量和任何个人记录不落地。"""
    def __init__(self, y, x, ids):
        self.y = np.asarray(y, float)
        self.x = np.asarray(x, float)
        _, codes = np.unique(np.asarray(ids), return_inverse=True)
        order = np.argsort(codes, kind='stable')
        starts = np.r_[0, np.flatnonzero(np.diff(codes[order]))+1, len(order)]
        self.indices = [order[a:b] for a, b in zip(starts[:-1], starts[1:])]
        self.n, self.p = self.x.shape
        self.sizes = np.asarray([len(i) for i in self.indices])
        self.xsum = np.asarray([self.x[i].sum(axis=0) for i in self.indices])
        self.order, self.starts = order, starts[:-1]

    def woodbury_profile(self, beta, q):
        """与密集Cholesky不同的秩一恒等式，仅评价指定点。"""
        if not np.isfinite(q) or q < 0:
            raise ValueError('方差比非法')
        r = self.y-self.x@beta
        sums = np.add.reduceat(r[self.order], self.starts)
        den = 1+self.sizes*q
        Q = float(r@r-q*np.sum(sums*sums/den))
        if Q <= 0 or not np.isfinite(Q):
            raise ValueError('残差平方型非法')
        sigma = Q/self.n
        ll = -.5*(self.n*(np.log(2*np.pi)+np.log(sigma)+1)+np.log(den).sum())
        h = self.x.T@r-q*self.xsum.T@(sums/den)
        score = np.r_[h/sigma, -.5*np.sum(self.sizes/den)+np.sum(sums*sums/den**2)/(2*sigma)]
        return float(ll), score, sigma

    def physical(self, beta, tau, sigma, whiten=False):
        """通用块矩阵公式计算(β,τ²,σ²)完整观察导数。"""
        if not np.isfinite([tau, sigma]).all() or tau < 0 or sigma <= 0:
            raise ValueError('物理方差非法，不裁剪')
        p = self.p
        score = np.zeros(p+2)
        H = np.zeros((p+2, p+2))
        logdet, quad = 0., 0.
        wx = np.empty_like(self.x) if whiten else None
        wy = np.empty_like(self.y) if whiten else None
        for index in self.indices:
            xi, yi = self.x[index], self.y[index]
            r = yi-xi@beta
            size = len(index)
            Vk = (np.ones((size, size)), np.eye(size))
            V = tau*Vk[0]+sigma*Vk[1]
            L = np.linalg.cholesky(V)
            W = solve_triangular(L.T, solve_triangular(L, np.eye(size), lower=True), lower=False)
            a = W@r
            logdet += 2*np.log(np.diag(L)).sum()
            quad += r@a
            score[:p] += xi.T@a
            H[:p, :p] -= xi.T@W@xi
            for k in range(2):
                score[p+k] += .5*(a@Vk[k]@a-np.trace(W@Vk[k]))
                H[:p, p+k] -= xi.T@W@Vk[k]@a
                for j in range(2):
                    H[p+k, p+j] += .5*np.trace(W@Vk[k]@W@Vk[j])-.5*(
                        a@Vk[k]@W@Vk[j]@a+a@Vk[j]@W@Vk[k]@a)
            if whiten:
                wx[index] = solve_triangular(L, xi, lower=True)
                wy[index] = solve_triangular(L, yi, lower=True)
        H[p:, :p] = H[:p, p:].T
        return {'loglik': float(-.5*(self.n*np.log(2*np.pi)+logdet+quad)),
            'score': score, 'hessian': H, 'wx': wx, 'wy': wy}

    def profile_from_physical(self, beta, q):
        """先转为(β,q,σ²)，再用Schur补消除剖面残差方差。"""
        ll, _, sigma = self.woodbury_profile(beta, q)
        a = self.physical(beta, q*sigma, sigma)
        J = np.eye(self.p+2)
        J[-2, -2], J[-2, -1] = sigma, q
        score = J.T@a['score']
        H = J.T@a['hessian']@J
        # τ²=qσ²的二阶导数项在非精确驻点处不可省略。
        H[-2, -1] += a['score'][-2]
        H[-1, -2] += a['score'][-2]
        profile = H[:-1, :-1]-np.outer(H[:-1, -1], H[-1, :-1])/H[-1, -1]
        return {'loglik': ll, 'sigma': sigma, 'physical': a,
            'score': score[:-1], 'hessian': profile, 'joint_ratio_hessian': H}


def inverse_information(H):
    """正对角均衡后Cholesky求逆；禁止伪逆和特征值裁剪。"""
    I = -np.asarray(H, float)
    if not np.isfinite(I).all() or not np.allclose(I, I.T, rtol=1e-10, atol=1e-12) or np.any(np.diag(I) <= 0):
        raise ValueError('观察信息不合法')
    d = np.sqrt(np.diag(I))
    balanced = I/d[:, None]/d[None, :]
    L = np.linalg.cholesky(balanced)
    cov = solve_triangular(L.T, solve_triangular(L, np.eye(len(I)), lower=True), lower=False)
    cov = cov/d[:, None]/d[None, :]
    return cov, {'eigenvalues': np.linalg.eigvalsh(I).tolist(),
        'raw_condition': float(np.linalg.cond(I)),
        'balanced_eigenvalues': np.linalg.eigvalsh(balanced).tolist(),
        'balanced_condition': float(np.linalg.cond(balanced)),
        'inverse_backward_error': float(np.max(abs(I@cov-np.eye(len(I)))))}


def finite_difference_profile(blocks, theta, H, step):
    """信息对角尺度下对独立Woodbury score做五点有限差分。"""
    scale = 1/np.sqrt(np.diag(-H))
    answer = np.empty_like(H)
    for j in range(len(theta)):
        shift = np.zeros(len(theta)); shift[j] = scale[j]*step
        values = []
        for k in (-2, -1, 1, 2):
            z = theta+k*shift
            values.append(blocks.woodbury_profile(z[:-1], z[-1])[1]*scale)
        answer[:, j] = (values[0]-8*values[1]+8*values[2]-values[3])/(12*step)
    exact = scale[:, None]*H*scale[None, :]
    return answer, float(np.max(abs(answer-exact)))


def log_variance_coordinates(score, H, tau, sigma):
    """保留非零score的坐标曲率项，检查对数方差坐标不变性。"""
    J = np.eye(len(score)); J[-2, -2] = tau; J[-1, -1] = sigma
    transformed = J.T@H@J
    transformed[-2, -2] += tau*score[-2]
    transformed[-1, -1] += sigma*score[-1]
    return J.T@score, transformed
