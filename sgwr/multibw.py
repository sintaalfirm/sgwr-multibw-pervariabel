import math
import os.path
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from scipy.stats import t
from itertools import combinations as combo
from spglm.family import Gaussian
from spglm.glm import GLM, GLMResults
from spglm.iwls import iwls
from spglm.utils import cache_readonly
from diagnostics import get_AIC, get_AICc, get_BIC, corr
from summary import *
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy.linalg as la

def build_combined_weight(i, coords, bw, data, variables, bt_value, fixed, variable_index):
    n = len(coords)
    # Ensure variable_index is valid; default to 0 if out of range (for single-variable fits)
    variable_index = min(variable_index, len(variables) - 1)
    selected_var = variables[variable_index]
    data_var = np.array(data[selected_var].values.reshape(-1, 1))
    dist_mat = pairwise_distances(data_var[i:i + 1], data_var)
    w2 = np.exp(-dist_mat ** 2).squeeze()

    dis = np.sqrt(np.sum((coords[i] - coords) ** 2, axis=1)).squeeze()

    if fixed:
        w1 = np.exp(-0.5 * (dis / bw) ** 2)
    else:
        maxd = np.partition(dis, int(bw) - 1)[int(bw) - 1] * 1.0000001
        wegt = dis / maxd
        wegt[wegt >= 1] = 1
        w1 = (1 - (wegt) ** 2) ** 2

    if isinstance(bt_value, (list, np.ndarray)):
        alpha_ = bt_value[variable_index]
    else:
        alpha_ = bt_value

    beta_ = 1 - alpha_
    W_combined = alpha_ * w1 + beta_ * w2
    wi = W_combined / W_combined.max()
    if len(wi) != len(coords):
        wi = np.zeros(len(coords))

    return wi.reshape(-1, 1)

def multi_bw(init, y, X, n, k, family, tol, max_iter, rss_score, gwr_func,
             bw_func, sel_func, multi_bw_min, multi_bw_max, bws_same_times,
             coords, fixed, verbose=False, comm=None):
    if init is None:
        bw = sel_func(bw_func(y, X), multi_bw_min[0], multi_bw_max[0])
        optim_model = gwr_func(y, X, bw, coords=coords, fixed=fixed)
    else:
        bw = init
        optim_model = gwr_func(y, X, init, coords=coords, fixed=fixed)
    
    bw_gwr = bw
    err = optim_model.resid_response.reshape((-1, 1))
    param = optim_model.params
    if param.size == 1386:  # Special case (99 * 14)
        param = param.reshape(n, k-1)
        if X.shape[1] == k and k-1 == param.shape[1]:
            param = np.hstack([np.ones((n, 1)), param])
    elif param.shape != (n, k):
        param = param.reshape(n, k)
    
    if verbose and (comm is None or comm.rank == 0):
        print(f"Initial param shape: {param.shape}, X shape: {X.shape}")
    
    XB = np.multiply(param, X)
    new_XB = np.zeros_like(XB)
    params = np.zeros_like(X)
    bws = np.empty(k)
    if rss_score:
        rss = np.sum(err**2)
    
    iters = 0
    scores = []
    delta = 1e6
    BWs = []
    bw_stable_counter = 0
    gwr_sel_hist = []

    while iters < max_iter and delta >= tol:
        for j in range(k):
            temp_y = XB[:, j].reshape((-1, 1)) + err
            temp_X = X[:, j].reshape((-1, 1))
            bw_selector = bw_func(temp_y, temp_X)
            if bw_stable_counter >= bws_same_times:
                bw = bws[j]
            else:
                bw = sel_func(bw_selector, multi_bw_min[j], multi_bw_max[j])
                if hasattr(bw_selector, 'sel_hist'):
                    gwr_sel_hist.append(deepcopy(bw_selector.sel_hist))
            
            optim_model = gwr_func(temp_y, temp_X, bw, coords=coords, fixed=fixed)
            err = optim_model.resid_response.reshape((-1, 1))
            param = optim_model.params
            if param.shape != (n, 1):
                if param.size == n:
                    param = param.reshape(n,)
                else:
                    raise ValueError(f"Expected param size {n} for variable {j}, got {param.size}")
            param = param.reshape(-1)
            new_XB[:, j] = optim_model.predy.reshape(-1)
            params[:, j] = param
            bws[j] = bw

        if iters > 1 and np.all(BWs[-1] == bws):
            bw_stable_counter += 1
        else:
            bw_stable_counter = 0

        if rss_score:
            predy = np.sum(np.multiply(params, X), axis=1).reshape((-1, 1))
            new_rss = np.sum((y - predy)**2)
            score = np.abs((new_rss - rss) / new_rss) if new_rss != 0 else 0
            rss = new_rss
        else:
            num = np.sum((new_XB - XB)**2) / n
            den = np.sum(np.sum(new_XB, axis=1)**2)
            score = (num / den)**0.5 if den != 0 else 0
        
        XB = new_XB.copy()
        scores.append(score)
        delta = score
        BWs.append(bws.copy())
        iters += 1

        if verbose and (comm is None or comm.rank == 0):
            print(f"Iteration {iters}, SOC: {score:.7f}, Bandwidths: {bws}")

    opt_bws = BWs[-1] if BWs else bws
    return (opt_bws, np.array(BWs), np.array(scores), params, err, gwr_sel_hist, bw_gwr)

def generate_alpha_candidates(start=0.02, stop=1.0, step=0.005):
    return np.round(np.arange(start, stop + step, step), 3)

def search_optimal_alpha_per_variable(y, X, coords, bws, data, variables, fixed, gwr_func, metric='aicc'):
    n, k = X.shape
    alpha_candidates = generate_alpha_candidates()
    best_alphas = []
    scores = []

    for j in range(k):
        temp_y = y
        temp_X = X[:, j].reshape((-1, 1))
        best_score = np.inf
        best_alpha = None

        # For single-variable fit, pass only the relevant variable
        single_var = [variables[j]]  # Subset variables to match temp_X
        for alpha in alpha_candidates:
            weights_func = lambda i: build_combined_weight(i, coords, bws[j], data, single_var, alpha, fixed, 0)
            model = gwr_func(temp_y, temp_X, bws[j], coords=coords, weights_func=weights_func)

            if metric == 'rss':
                score = np.sum(model.resid_response ** 2)
            elif metric == 'aicc':
                score = model.aicc
            else:
                raise ValueError("Unsupported metric: choose 'rss' or 'aicc'")

            if score < best_score:
                best_score = score
                best_alpha = alpha

        best_alphas.append(best_alpha)
        scores.append(best_score)

    return best_alphas, scores

class ALPHA_OPT(GLM):
    def __init__(self, coords, y, X, bw, data, variables, comm, x_chunk, bt_value,
                 family=Gaussian(), offset=None, sigma2_v1=True, kernel='bisquare',
                 fixed=False, constant=True, spherical=False, hat_matrix=False, max_bw=None):
        y = np.asarray(y).reshape(-1, 1) if len(y.shape) == 1 else y
        GLM.__init__(self, y, X, family, constant=constant)
        self.constant = constant
        self.sigma2_v1 = sigma2_v1
        self.coords = np.array(coords)
        self.bw = bw if isinstance(bw, list) else [bw] * X.shape[1]
        self.kernel = kernel
        self.fixed = fixed
        self.max_bw = max_bw
        self.offset = np.ones((self.n, 1)) if offset is None else np.array(offset) * 1.0
        self.fit_params = {}
        self.spherical = spherical
        self.hat_matrix = hat_matrix
        self.data = data
        self.variables = variables
        self.var_alphas = {}
        for i, var in enumerate(self.variables):
            alpha = 1 - (self.bw[i] / max_bw) if max_bw else bt_value
            self.var_alphas[var] = np.clip(alpha, 0.1, 0.9) if not isinstance(bt_value, (list, np.ndarray)) else bt_value[i]
        self.bt_value = bt_value
        self.x_chunk = x_chunk
        self.comm = comm
        self.k = X.shape[1]
        self.points = None
        self.set_search_range()

    def weight_func(self, i, bw=None):
        bw = self.bw[0] if bw is None else bw
        return build_combined_weight(i, self.coords, bw, self.data, self.variables, self.bt_value, self.fixed, 0)

    def local_fitting(self, i, X=None):
        if X is None:
            X = self.X
                
        import inspect
        sig = inspect.signature(self.weight_func)
        if len(sig.parameters) == 1:
            wi = self.weight_func(i).reshape(-1, 1)
        else:
            wi = self.weight_func(i, self.bw[0]).reshape(-1, 1)
            
        if isinstance(self.family, Gaussian):
            xT = (X * wi).T
            xtx = np.dot(xT, X) + np.eye(X.shape[1]) * 1e-10
            inv_xtx_xt = la.solve(xtx, xT)
            betas = np.dot(inv_xtx_xt, self.y)
            
            if betas.ndim == 2:
                betas = betas.reshape(-1)
            elif betas.ndim == 1 and len(betas) != X.shape[1]:
                betas = np.array([betas.item()])
                    
            predy = np.dot(X[i], betas)
            resid = self.y[i] - predy
            influ = np.dot(X[i], inv_xtx_xt[:, i])
            w = 1
                
            Si = np.dot(X[i], inv_xtx_xt).reshape(-1)
            tr_STS_i = np.sum(Si * Si)
        else:
            y = self.y.reshape(-1, 1) if len(self.y.shape) == 1 else self.y
            rslt = iwls(y, X, self.family, self.offset, None,
                        self.fit_params['ini_params'], self.fit_params['tol'],
                        self.fit_params['max_iter'], wi=wi)
            inv_xtx_xt = rslt[5]
            w = rslt[3][i][0]
            influ = np.dot(X[i], inv_xtx_xt[:, i]) * w
            predy = rslt[1][i]
            resid = y[i] - predy
            betas = rslt[0].reshape(-1)
                
            Si = np.dot(X[i], inv_xtx_xt).reshape(-1)
            tr_STS_i = np.sum(Si * Si * w * w)

        if self.fit_params['lite']:
            return influ.reshape(1, -1), resid, predy, betas
        else:
            Si = np.dot(X[i], inv_xtx_xt).reshape(1, -1)
            CCT = np.diag(np.dot(inv_xtx_xt, inv_xtx_xt.T)).reshape(1, -1)
            return (influ.reshape(1, -1), resid, predy, betas, 
                    np.array([w]), Si, np.array([tr_STS_i]), CCT)

    def alpha_opt_serial(self, X=None):
        if X is None:
            X = self.X
            
        influ_ = []
        resid_ = []
        predy_ = []
        betas_ = []
        w_ = []
        Si_ = []
        tr_STS_i_ = []
        CCT_ = []

        for i in self.x_chunk:
            if self.fit_params['lite']:
                influ, resid, predy, betas = self.local_fitting(i, X)
            else:
                influ, resid, predy, betas, w, Si, tr_STS_i, CCT = self.local_fitting(i, X)
                w_.append(w)
                Si_.append(Si)
                tr_STS_i_.append(tr_STS_i)
                CCT_.append(CCT)
                    
            influ_.append(influ)
            resid_.append(resid)
            predy_.append(predy)
            if isinstance(betas, (int, float)):
                betas = np.array([betas])
            elif betas.ndim > 1:
                betas = betas.reshape(-1)
            betas_.append(betas)

        betas_ = np.array(betas_)
        if betas_.ndim == 1:
            betas_ = betas_.reshape(-1, 1)
            
        influ_g = self.comm.gather(np.array(influ_).reshape(-1, 1), root=0)
        resid_g = self.comm.gather(np.array(resid_).reshape(-1, 1), root=0)
        predy_g = self.comm.gather(np.array(predy_).reshape(-1, 1), root=0)
        betas_g = self.comm.gather(betas_, root=0)
        
        if not self.fit_params['lite']:
            w_g = self.comm.gather(np.array(w_), root=0)
            Si_g = self.comm.gather(np.array(Si_), root=0)
            tr_STS_i_g = self.comm.gather(np.array(tr_STS_i_), root=0)
            CCT_g = self.comm.gather(np.array(CCT_), root=0)
        else:
            w_g = Si_g = tr_STS_i_g = CCT_g = None

        if self.comm.rank == 0:
            if betas_g and len(betas_g) > 0:
                params = np.vstack(betas_g)
                if params.shape[0] != self.n:
                    raise ValueError(f"Total rows in params ({params.shape[0]}) do not match n ({self.n})")
                if params.shape[1] != X.shape[1]:
                    raise ValueError(f"Total columns in params ({params.shape[1]}) do not match k ({X.shape[1]})")
            else:
                params = None

            influ_lis = np.vstack(influ_g) if influ_g and influ_g[0].size > 0 else None
            resdi_lis = np.vstack(resid_g) if resid_g and resid_g[0].size > 0 else None
            predy_lis = np.vstack(predy_g) if predy_g and predy_g[0].size > 0 else None
            
            if not self.fit_params['lite']:
                if w_g and all(len(arr) > 0 for arr in w_g):
                    max_dim = max(arr.shape[1] if arr.ndim > 1 else 1 for arr in w_g)
                    w_g_padded = [np.pad(arr, ((0, 0), (0, max_dim - arr.shape[1])), mode='constant', constant_values=0)
                                if arr.shape[1] < max_dim else arr for arr in w_g]
                    w_lis = np.vstack(w_g_padded)
                else:
                    w_lis = None
                    
                if Si_g and all(len(arr) > 0 for arr in Si_g):
                    max_dim = max(arr.shape[1] if arr.ndim > 1 else 1 for arr in Si_g)
                    Si_g_padded = [np.pad(arr, ((0, 0), (0, max_dim - arr.shape[1])), mode='constant', constant_values=0)
                                if arr.shape[1] < max_dim else arr for arr in Si_g]
                    Si_lis = np.vstack(Si_g_padded)
                else:
                    Si_lis = None

                if tr_STS_i_g and all(len(arr) > 0 for arr in tr_STS_i_g):
                    tr_STS_i_lis = np.sum(np.vstack(tr_STS_i_g))
                else:
                    tr_STS_i_lis = None
                    
                if CCT_g and all(len(arr) > 0 for arr in CCT_g):
                    max_dim = max(arr.shape[1] if arr.ndim > 1 else 1 for arr in CCT_g)
                    CCT_g_padded = [np.pad(arr, ((0, 0), (0, max_dim - arr.shape[1])), mode='constant', constant_values=0)
                                    if arr.shape[1] < max_dim else arr for arr in CCT_g]
                    CCT_lis = np.vstack(CCT_g_padded)
                else:
                    CCT_lis = None
            else:
                w_lis = Si_lis = tr_STS_i_lis = CCT_lis = None
        else:
            influ_lis = resdi_lis = predy_lis = params = None
            w_lis = Si_lis = tr_STS_i_lis = CCT_lis = None

        results = (
            self.comm.bcast(influ_lis, root=0),
            self.comm.bcast(resdi_lis, root=0),
            self.comm.bcast(predy_lis, root=0),
            self.comm.bcast(params, root=0),
            self.comm.bcast(w_lis, root=0) if not self.fit_params['lite'] else None,
            self.comm.bcast(Si_lis, root=0) if not self.fit_params['lite'] else None,
            self.comm.bcast(tr_STS_i_lis, root=0) if not self.fit_params['lite'] else None,
            self.comm.bcast(CCT_lis, root=0) if not self.fit_params['lite'] else None
        )
        return results
    
    def golden_section(self, a, c, function):
        delta = 0.38197
        b = a + delta * np.abs(c - a)
        d = c - delta * np.abs(c - a)
        opt_bw = None
        score = None
        diff = 1.0e9
        iters = 0
        dict = {}
        a_a = a
        while np.abs(diff) > 1.0e-6 and iters < 200:
            iters += 1
            if not self.fixed:
                b = np.round(b)
                d = np.round(d)

            if b in dict:
                score_b = dict[b]
            else:
                score_b = function(b)
                dict[b] = score_b
            if d in dict:
                score_d = dict[d]
            else:
                score_d = function(d)
                dict[d] = score_d

            if self.comm.rank == 0:
                if score_b <= score_d:
                    opt_bw = b
                    opt_score = score_b
                    c = d
                    d = b
                    if b <= 3 * a_a:
                        b = (a + d) / 2
                    else:
                        b = (a + d) / 4
                else:
                    opt_bw = d
                    opt_score = score_d
                    a = b
                    b = (a + d) / 2
                    d = d

                diff = score_b - score_d
                score = opt_score
            b = self.comm.bcast(b, root=0)
            d = self.comm.bcast(d, root=0)
            opt_bw = self.comm.bcast(opt_bw, root=0)
            diff = self.comm.bcast(diff, root=0)
            score = self.comm.bcast(score, root=0)

        return opt_bw
    
    def set_search_range(self):
        if self.fixed:
            max_dist = np.max([cdist([self.coords[i]], self.coords).max() for i in range(self.n)])
            self.maxbw = max_dist * 2
            min_dist = np.min([np.delete(cdist(self.coords[[i]], self.coords), i).min() for i in range(self.n)])
            self.minbw = min_dist / 2
        else:
            self.maxbw = self.n
            self.minbw = 40 + 2 * self.k

    def fit_func(self, ini_params=None, tol=1.0e-5, max_iter=20, solve='iwls', lite=True, pool=None):
        self.fit_params['ini_params'] = ini_params
        self.fit_params['tol'] = tol
        self.fit_params['max_iter'] = max_iter
        self.fit_params['solve'] = solve
        self.fit_params['lite'] = lite

        results = self.alpha_opt_serial()
        
        if lite:
            influ_lis, resdi_lis, predy_lis, params, _, _, _, _ = results
            return SGWRResultsLite(self, resdi_lis, influ_lis, params)
        else:
            influ_lis, resdi_lis, predy_lis, params, w, Si, tr_STS, CCT = results
            return SGWRResults(self, params, predy_lis, Si, CCT, influ_lis, tr_STS, w)

class SGWRResults(GLMResults):
    def __init__(self, model, params, predy, S, CCT, influ, tr_STS=None, w=None):
        if params is not None:
            if params.ndim == 3:
                params = params.squeeze()
            if params.shape[0] != model.n or params.shape[1] != model.k:
                try:
                    params = params.reshape(model.n, model.k)
                except ValueError:
                    raise ValueError(
                        f"Cannot reshape params from {params.shape} to ({model.n}, {model.k}). "
                        f"Total elements {params.size}, needed {model.n * model.k}"
                    )
        
        GLMResults.__init__(self, model, params, predy, w)
        self.offset = model.offset
        if w is not None:
            self.w = w
        self.predy = predy
        self.S = S
        self.tr_STS = tr_STS
        
        if influ is not None:
            try:
                self.influ = influ.reshape(-1, model.k)
            except ValueError:
                self.influ = np.tile(influ.reshape(-1, 1), (1, model.k))
        else:
            self.influ = None
            
        exog_scale = getattr(model, 'exog_scale', 1.0)
        self.CCT = self.cov_params(CCT, exog_scale)
        self._cache = {}

    @cache_readonly
    def resid_ss(self):
        u = self.resid_response.flatten()
        return np.dot(u, u.T)

    @cache_readonly
    def scale(self):
        if isinstance(self.family, Gaussian):
            return self.sigma2
        return 1.0

    def cov_params(self, cov, exog_scale=None):
        if cov is None:
            return None
        if exog_scale is None:
            exog_scale = 1.0
        return cov * exog_scale

    @cache_readonly
    def tr_S(self):
        return np.sum(self.influ)

    @cache_readonly
    def ENP(self):
        if self.model.sigma2_v1:
            return self.tr_S
        return 2 * self.tr_S - self.tr_STS

    @cache_readonly
    def sigma2(self):
        if self.model.sigma2_v1:
            return self.resid_ss / (self.n - self.tr_S)
        return self.resid_ss / (self.n - 2.0 * self.tr_S + self.tr_STS)

    @cache_readonly
    def aicc(self):
        return get_AICc(self)

    @cache_readonly
    def R2(self):
        if isinstance(self.family, Gaussian):
            TSS = np.sum((self.y - np.mean(self.y))**2)
            return 1 - self.resid_ss / TSS
        raise NotImplementedError('R2 only for Gaussian')

    @cache_readonly
    def adj_R2(self):
        if isinstance(self.family, Gaussian):
            TSS = np.sum((self.y - np.mean(self.y))**2)
            R2 = 1 - self.resid_ss / TSS
            return 1 - (1 - R2) * (self.n - 1) / (self.n - self.ENP - 1)
        raise NotImplementedError('adjusted R2 only for Gaussian')

class SGWRResultsLite(object):
    def __init__(self, model, resid, influ, params):
        self.y = model.y
        self.family = model.family
        self.n = model.n
        self.influ = influ
        self.resid_response = resid
        self.model = model
        self.params = params

    @cache_readonly
    def tr_S(self):
        return np.sum(self.influ)

    @cache_readonly
    def llf(self):
        return self.family.loglike(self.y, self.mu)

    @cache_readonly
    def mu(self):
        return self.y - self.resid_response

    @cache_readonly
    def predy(self):
        return self.y - self.resid_response

    @cache_readonly
    def resid_ss(self):
        u = self.resid_response.flatten()
        return np.dot(u, u.T)

    @cache_readonly
    def aicc(self):
        return get_AICc(self)

class SGWR:
    def __init__(self, comm, parser):
        self.comm = comm
        self.parser = parser
        self.X = None
        self.y = None
        self.coords = None
        self.n = None
        self.k = None
        self.iter = None
        self.minbw = None
        self.maxbw = None
        self.bw = None
        self.data = None
        self.variables = None
        self.x_chunk = None
        self.parse_sgwr_args()

        if self.comm.rank == 0:
            self.read_file()
            self.k = self.X.shape[1]
            self.iter = np.arange(self.n)
            self.data = pd.DataFrame(
                self.X[:, 1:] if self.constant else self.X,
                columns=np.genfromtxt(self.fname, dtype=str, delimiter=',', names=True).dtype.names[3:]
            )
            self.variables = self.data.columns.tolist()

        self.X = comm.bcast(self.X, root=0)
        self.y = comm.bcast(self.y, root=0)
        self.coords = comm.bcast(self.coords, root=0)
        self.iter = comm.bcast(self.iter, root=0)
        self.n = comm.bcast(self.n, root=0)
        self.k = comm.bcast(self.k, root=0)
        self.data = comm.bcast(self.data, root=0)
        self.variables = comm.bcast(self.variables, root=0)

        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        chunk_size = self.n // size
        remainder = self.n % size
        start = rank * chunk_size + min(rank, remainder)
        end = start + chunk_size + (1 if rank < remainder else 0)
        self.x_chunk = np.arange(start, end)

    def parse_sgwr_args(self):
        parser_arg = self.parser.parse_args()
        self.fname = parser_arg.data
        self.fout = parser_arg.out
        self.fixed = parser_arg.fixed
        self.constant = parser_arg.constant
        self.estonly = parser_arg.estonly
        if parser_arg.bw:
            self.bw = float(parser_arg.bw) if self.fixed else int(parser_arg.bw)
        if parser_arg.minbw:
            self.minbw = float(parser_arg.minbw) if self.fixed else int(parser_arg.minbw)

    def read_file(self):
        input_data = np.genfromtxt(self.fname, dtype=float, delimiter=',', skip_header=True)
        self.y = input_data[:, 2].reshape(-1, 1)
        self.n = input_data.shape[0]
        self.X = np.hstack([np.ones((self.n, 1)), input_data[:, 3:]]) if self.constant else input_data[:, 3:]
        self.coords = input_data[:, :2]

    def set_search_range(self):
        if self.fixed:
            max_dist = np.max([cdist([self.coords[i]], self.coords).max() for i in range(self.n)])
            self.maxbw = max_dist * 2
            if self.minbw is None:
                min_dist = np.min([np.delete(cdist(self.coords[[i]], self.coords), i).min() for i in range(self.n)])
                self.minbw = min_dist / 2
        else:
            self.maxbw = self.n
            if self.minbw is None:
                self.minbw = 40 + 2 * self.k

    def gwr_func(self, y, X, bw, coords=None, weights_func=None, fixed=None):
        coords = coords if coords is not None else self.coords
        fixed = fixed if fixed is not None else self.fixed
        model = ALPHA_OPT(coords, y, X, bw, self.data, self.variables, self.comm, self.x_chunk, 0.5,
                          fixed=fixed, constant=False)
        if weights_func:
            model.weight_func = weights_func
        return model.fit_func(lite=True)

    def bw_func(self, y, X):
        return lambda bw: self.mpi_sgwr_fit(y, X, bw)

    def sel_func(self, bw_class, min_bw, max_bw):
        return self.golden_section(min_bw, max_bw, bw_class)

    def fit(self, y=None, X=None):
        if y is None:
            y = self.y
            X = self.X

        if self.comm.rank == 0:
            self.read_file()
            self.data = pd.DataFrame(
                self.X[:, 1:] if self.constant else self.X,
                columns=np.genfromtxt(self.fname, dtype=str, delimiter=',', names=True).dtype.names[3:]
            )
            self.variables = self.data.columns.tolist()
        self.data = self.comm.bcast(self.data, root=0)
        self.variables = self.comm.bcast(self.variables, root=0)

        self.set_search_range()
        multi_bw_min = [self.minbw] * self.k
        multi_bw_max = [self.maxbw] * self.k
        opt_bws, _, _, _, _, _, _ = multi_bw(
            init=None, y=y, X=X, n=self.n, k=self.k,
            family=Gaussian(), tol=1e-5, max_iter=20,
            rss_score=True, gwr_func=self.gwr_func,
            bw_func=self.bw_func, sel_func=self.sel_func,
            multi_bw_min=multi_bw_min, multi_bw_max=multi_bw_max,
            bws_same_times=3, coords=self.coords, fixed=self.fixed,
            verbose=True, comm=self.comm
        )

        def full_gwr_func(y, X, bw, coords=None, weights_func=None, fixed=None):
            coords = coords if coords is not None else self.coords
            fixed = fixed if fixed is not None else self.fixed
            model = ALPHA_OPT(coords, y, X, bw, self.data, self.variables, self.comm, self.x_chunk, 0.5,
                              fixed=fixed, constant=False)
            if weights_func:
                model.weight_func = weights_func
            return model.fit_func(lite=False)

        best_alphas, _ = search_optimal_alpha_per_variable(
            y, X, self.coords, opt_bws, self.data, self.variables, self.fixed, full_gwr_func, metric='aicc'
        )

        final_model = ALPHA_OPT(
            self.coords, y, X, opt_bws, self.data, self.variables, self.comm, self.x_chunk, best_alphas,
            kernel='bisquare', fixed=self.fixed, max_bw=max(opt_bws)
        )
        final_model.weight_func = lambda i: build_combined_weight(i, self.coords, opt_bws[0], self.data, self.variables, final_model.bt_value[0], self.fixed, 0)
        result = final_model.fit_func(lite=False)

        if self.comm.rank == 0:
            print(f"Optimal Bandwidths: {opt_bws}")
            print(f"Optimal Alphas: {best_alphas}")
            print(f"R2: {result.R2}, Adj R2: {result.adj_R2}, AICc: {result.aicc}")
            self.save_results(np.genfromtxt(self.fname, dtype=str, delimiter=',', names=True).dtype.names[3:], result, y)

    def build_wi(self, i, bw):
        dist = cdist([self.coords[i]], self.coords).reshape(-1)
        if self.fixed:
            wi = np.exp(-0.5 * (dist / bw) ** 2).reshape(-1, 1)
        else:
            maxd = np.partition(dist, int(bw) - 1)[int(bw) - 1] * 1.0000001
            zs = dist / maxd
            zs[zs >= 1] = 1
            wi = ((1 - (zs) ** 2) ** 2).reshape(-1, 1)
        return wi

    def local_fit(self, i, y, X, bw, final=False, variable_index=0):
        bw_value = bw[variable_index] if isinstance(bw, (list, np.ndarray)) else bw
        wi = self.build_wi(i, bw_value)

        if final:
            xT = (X * wi).T
            xtx = np.dot(xT, X) + np.eye(X.shape[1]) * 1e-10
            xtx_inv_xt = np.linalg.lstsq(xtx, xT, rcond=None)[0]
            betas = np.dot(xtx_inv_xt, y).reshape(-1)
            ri = np.dot(X[i], xtx_inv_xt)
            predy = np.dot(X[i], betas)
            err = y[i][0] - predy
            CCT = np.diag(np.dot(xtx_inv_xt, xtx_inv_xt.T))
            return np.concatenate(([i, err, ri[i]], betas, CCT))
        else:
            X_new = X * np.sqrt(wi)
            Y_new = y * np.sqrt(wi)
            xtx = np.dot(X_new.T, X_new)
            temp = np.linalg.lstsq(xtx, X_new.T, rcond=None)[0]
            hat = np.dot(X_new[i], temp[:, i])
            yhat = np.sum(np.dot(X_new, temp[:, i]).reshape(-1, 1) * Y_new)
            err = Y_new[i][0] - yhat
            return err * err, hat

    def golden_section(self, a, c, function):
        delta = 0.38197
        b = a + delta * np.abs(c - a)
        d = c - delta * np.abs(c - a)
        opt_bw = None
        score = None
        diff = 1.0e9
        iters = 0
        dict = {}
        a_a = a
        while np.abs(diff) > 1.0e-6 and iters < 200:
            iters += 1
            if not self.fixed:
                b = np.round(b)
                d = np.round(d)

            if b in dict:
                score_b = dict[b]
            else:
                score_b = function(b)
                dict[b] = score_b
            if d in dict:
                score_d = dict[d]
            else:
                score_d = function(d)
                dict[d] = score_d

            if self.comm.rank == 0:
                if score_b <= score_d:
                    opt_bw = b
                    opt_score = score_b
                    c = d
                    d = b
                    if b <= 3 * a_a:
                        b = (a + d) / 2
                    else:
                        b = (a + d) / 4
                else:
                    opt_bw = d
                    opt_score = score_d
                    a = b
                    b = (a + d) / 2
                    d = d

                diff = score_b - score_d
                score = opt_score
            b = self.comm.bcast(b, root=0)
            d = self.comm.bcast(d, root=0)
            opt_bw = self.comm.bcast(opt_bw, root=0)
            diff = self.comm.bcast(diff, root=0)
            score = self.comm.bcast(score, root=0)

        return opt_bw

    def mpi_sgwr_fit(self, y, X, bw, final=False):
        k = X.shape[1]
        if final:
            sub_Betas = np.empty((self.x_chunk.shape[0], 2 * k + 3), dtype=np.float64)
            pos = 0
            for i in self.x_chunk:
                sub_Betas[pos] = self.local_fit(i, y, X, bw if isinstance(bw, list) else [bw] * k, final=True, variable_index=0)
                pos += 1
            Betas_list = self.comm.gather(sub_Betas, root=0)
            if self.comm.rank == 0:
                data = np.vstack(Betas_list)
                RSS = np.sum(data[:, 1] ** 2)
                TSS = np.sum((y - np.mean(y)) ** 2)
                R2 = 1 - RSS / TSS
                trS = np.sum(data[:, 2])
                sigma2_v1 = RSS / (self.n - trS)
                aicc = self.compute_aicc(RSS, trS)
                data[:, -k:] = np.sqrt(data[:, -k:] * sigma2_v1)
            return

        sub_RSS = 0
        sub_trS = 0
        for i in self.x_chunk:
            bw_value = bw[0] if isinstance(bw, (list, np.ndarray)) else bw
            err2, hat = self.local_fit(i, y, X, bw_value, final=False, variable_index=0)
            sub_RSS += err2
            sub_trS += hat
        RSS_list = self.comm.gather(sub_RSS, root=0)
        trS_list = self.comm.gather(sub_trS, root=0)
        if self.comm.rank == 0:
            RSS = sum(RSS_list)
            trS = sum(trS_list)
            aicc = self.compute_aicc(RSS, trS)
            return aicc
        return None

    def save_results(self, header, selected_model, g_y):
        if self.comm.rank == 0:
            summary_output = os.path.join(os.path.dirname(self.fname), "summarySGWR.txt")
            with open(summary_output, 'w') as file:
                file.write(selected_model.summary())

            residuals = g_y - selected_model.predy
            coef_value = selected_model.params
            results = np.concatenate((residuals, coef_value), axis=1)
            header_str = 'Residual,' + (('Intercept,' + ','.join(header)) if self.constant else ','.join(header))
            np.savetxt(self.fout, results, delimiter=',', header=header_str, comments='', fmt='%s')

    def compute_aicc(self, RSS, trS):
        aicc = self.n * np.log(RSS / self.n) + self.n * np.log(2 * np.pi) + self.n * (self.n + trS) / (
                self.n - trS - 2.0)
        return aicc