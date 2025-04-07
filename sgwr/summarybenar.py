import numpy as np
import pandas as pd
from spglm.family import Gaussian
from spglm.glm import GLM
from diagnostics import get_AICc
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


def summaryModel(self):
    summary = '=' * 75 + '\n'
    summary += "%-54s %20s\n" % ('Model type', self.family.__class__.__name__)
    summary += "%-60s %14d\n" % ('Number of observations:', self.n)
    summary += "%-60s %14d\n\n" % ('Number of covariates:', self.k)
    return summary

def calculate_vif(X):
    X = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    return vif_data


def summaryGLM(self):
    XNames = getattr(self.model, 'variables', ["X" + str(i) for i in range(self.k)])
    if self.model.constant:
        XNames = ['Intercept'] + XNames

    glm_rslt = GLM(self.model.y, self.model.X, constant=self.model.constant, family=self.family).fit()

    summary = "%s\n" % ('Global Regression Results')
    summary += '-' * 75 + '\n'

    if isinstance(self.family, Gaussian):
        summary += "%-62s %12.3f\n" % ('Residual sum of squares:', glm_rslt.deviance)
        summary += "%-62s %12.3f\n" % ('Log-likelihood:', glm_rslt.llf)
        summary += "%-62s %12.3f\n" % ('AIC:', glm_rslt.aic)
        summary += "%-62s %12.3f\n" % ('AICc:', get_AICc(glm_rslt))
        summary += "%-62s %12.3f\n" % ('BIC:', glm_rslt.bic)
        summary += "%-62s %12.3f\n" % ('R2:', glm_rslt.D2)
        summary += "%-62s %12.3f\n\n" % ('Adj. R2:', glm_rslt.adj_D2)
    else:
        summary += "%-62s %12.3f\n" % ('Deviance:', glm_rslt.deviance)
        summary += "%-62s %12.3f\n" % ('Log-likelihood:', glm_rslt.llf)
        summary += "%-62s %12.3f\n" % ('AIC:', glm_rslt.aic)
        summary += "%-62s %12.3f\n" % ('AICc:', get_AICc(glm_rslt))
        summary += "%-62s %12.3f\n" % ('BIC:', glm_rslt.bic)
        summary += "%-62s %12.3f\n" % ('Percent deviance explained:', glm_rslt.D2)
        summary += "%-62s %12.3f\n\n" % ('Adj. percent deviance explained:', glm_rslt.adj_D2)

    # Coefficient table
    summary += "%s\n" % ('Coefficient Estimates')
    summary += "%-31s %10s %10s %10s %10s\n" % ('Variable', 'Est.', 'SE', 't(Est/SE)', 'p-value')
    summary += "%-31s %10s %10s %10s %10s\n" % ('-' * 31, '-' * 10, '-' * 10, '-' * 10, '-' * 10)
    n_params = len(glm_rslt.params)
    for i in range(min(n_params, len(XNames))):
        summary += "%-31s %10.3f %10.3f %10.3f %10.3f\n" % (
            XNames[i], glm_rslt.params[i], glm_rslt.bse[i], glm_rslt.tvalues[i], glm_rslt.pvalues[i])
    if n_params < len(XNames):
        summary += f"\nNote: {len(XNames) - n_params} variable(s) dropped in global fit (possibly due to collinearity).\n"

    # VIF table
    X_cols = self.model.X.shape[1]
    if X_cols != len(XNames):
        summary += f"\nWarning: X has {X_cols} columns, but XNames has {len(XNames)} elements. Adjusting column names.\n"
        XNames = XNames[:X_cols]  # Truncate or pad XNames to match X
    X = pd.DataFrame(self.model.X, columns=XNames)
    vif_data = calculate_vif(X)
    summary += "\n%s\n" % ('Variance Inflation Factor (VIF)')
    summary += '-' * 75 + '\n'
    summary += vif_data.to_string(index=False)
    summary += "\n"

    return summary


def summarySGWR(self):
    XNames = getattr(self.model, 'variables', ["X" + str(i) for i in range(self.k)])
    if self.model.constant:
        XNames = ['Intercept'] + XNames

    summary = "%s\n" % ('Similarity Geographically Weighted Regression (SGWR) Results')
    summary += '-' * 75 + '\n'

    kernel_type = 'Fixed' if self.model.fixed else 'Adaptive'
    summary += "%-54s %20s\n" % ('Spatial kernel:', f"{kernel_type} {self.model.kernel}")

    # Use self.opt_bws instead of self.model.bw
    if hasattr(self, 'opt_bws') and isinstance(self.opt_bws, (list, np.ndarray)):
        summary += "\n%s\n" % ('Variable-Specific Parameters')
        summary += '-' * 75 + '\n'
        summary += "%-20s %12s %12s %12s %12s %12s\n" % (
            'Variable', 'Bandwidth', 'Alpha', 'ENP_j', 'Adj t-val(95%)', 'DoD_j')
        summary += "%-20s %12s %12s %12s %12s %12s\n" % (
            '-' * 20, '-' * 12, '-' * 12, '-' * 12, '-' * 12, '-' * 12)

        for i in range(min(self.k, len(XNames))):
            var_name = XNames[i]
            bw_item = self.opt_bws[i] if i < len(self.opt_bws) else np.nan
            if isinstance(bw_item, np.ndarray):
                if bw_item.size > 1:
                    bw_val = float(bw_item[0])
                else:
                    bw_val = float(bw_item.item())
            else:
                bw_val = float(bw_item) if bw_item is not np.nan else np.nan
            
            alpha_val = (self.model.var_alphas.get(var_name, self.model.bt_value[i] if isinstance(self.model.bt_value, (list, np.ndarray)) and i < len(self.model.bt_value) else self.model.bt_value)
                        if hasattr(self.model, 'var_alphas') else np.nan)
            if isinstance(alpha_val, np.ndarray):
                if alpha_val.size > 1:
                    alpha_val = float(alpha_val[0])
                else:
                    alpha_val = float(alpha_val.item())
            else:
                alpha_val = float(alpha_val) if alpha_val is not None else np.nan

            # ENP_j (effective number of parameters per variable)
            try:
                enp_j = float(np.mean(self.influ[:, i])) if self.influ is not None and i < self.influ.shape[1] else np.nan
            except:
                enp_j = np.nan

            # DoD_j (degree of dependence)
            dod_j = float(1 - (enp_j / self.ENP)) if enp_j is not None and not np.isnan(enp_j) and self.ENP != 0 else np.nan

            # Adjusted t-value
            adj_t = float(self.critical_tval(self.adj_alpha[1])) if hasattr(self, 'adj_alpha') and len(self.adj_alpha) > 1 else np.nan

            summary += "%-20s %12.3f %12.3f %12.3f %12.3f %12.3f\n" % (
                var_name, bw_val, alpha_val, enp_j, adj_t, dod_j)

    # Diagnostic information
    summary += "\n%s\n" % ('Diagnostic Information')
    summary += '-' * 75 + '\n'

    if isinstance(self.family, Gaussian):
        summary += "%-62s %12.3f\n" % ('Residual sum of squares:', self.resid_ss)
        summary += "%-62s %12.3f\n" % ('Effective number of parameters (trace(S)):', self.tr_S)
        summary += "%-62s %12.3f\n" % ('Degree of freedom (n - trace(S)):', self.df_model)
        summary += "%-62s %12.3f\n" % ('Sigma estimate:', np.sqrt(self.sigma2))
        summary += "%-62s %12.3f\n" % ('Log-likelihood:', self.llf)
        summary += "%-62s %12.3f\n" % ('AIC:', self.aic)
        summary += "%-62s %12.3f\n" % ('AICc:', self.aicc)
        summary += "%-62s %12.3f\n" % ('BIC:', self.bic)
        summary += "%-62s %12.3f\n" % ('R2:', self.R2)
        summary += "%-62s %12.3f\n" % ('Adjusted R2:', self.adj_R2)
    else:
        summary += "%-62s %12.3f\n" % ('Effective number of parameters (trace(S)):', self.tr_S)
        summary += "%-62s %12.3f\n" % ('Degree of freedom (n - trace(S)):', self.df_model)
        summary += "%-62s %12.3f\n" % ('Log-likelihood:', self.llf)
        summary += "%-62s %12.3f\n" % ('AIC:', self.aic)
        summary += "%-62s %12.3f\n" % ('AICc:', self.aicc)
        summary += "%-62s %12.3f\n" % ('BIC:', self.bic)
        summary += "%-62s %12.3f\n" % ('Percent deviance explained:', self.D2)
        summary += "%-62s %12.3f\n" % ('Adjusted percent deviance explained:', self.adj_D2)

    summary += "%-62s %12.3f\n" % ('Adj. alpha (95%):', self.adj_alpha[1] if hasattr(self, 'adj_alpha') else np.nan)
    summary += "%-62s %12.3f\n" % ('Adj. critical t value (95%):', 
                                   self.critical_tval(self.adj_alpha[1]) if hasattr(self, 'adj_alpha') else np.nan)

    # Alpha testing results
    if hasattr(self.model, 'alpha_values') and hasattr(self.model, 'aiccs'):
        summary += "\n%s\n" % ('Alpha Testing Results')
        summary += '-' * 75 + '\n'
        summary += "%-20s %12s\n" % ('Alpha', 'AICc')
        summary += "%-20s %12s\n" % ('-' * 20, '-' * 12)
        for alpha, aicc in zip(self.model.alpha_values, self.model.aiccs):
            summary += "%-20.3f %12.3f\n" % (alpha, aicc)

    # Summary statistics for parameter estimates
    summary += "\n%s\n" % ('Summary Statistics For SGWR Parameter Estimates')
    summary += '-' * 75 + '\n'
    summary += "%-20s %12s %12s %12s %12s %12s\n" % ('Variable', 'Mean', 'STD', 'Min', 'Median', 'Max')
    summary += "%-20s %12s %12s %12s %12s %12s\n" % ('-' * 20, '-' * 12, '-' * 12, '-' * 12, '-' * 12, '-' * 12)
    for i in range(min(self.k, len(XNames))):
        params_col = self.params[:, i] if i < self.params.shape[1] else np.array([np.nan])
        summary += "%-20s %12.3f %12.3f %12.3f %12.3f %12.3f\n" % (
            XNames[i], np.mean(params_col), np.std(params_col), 
            np.min(params_col), np.median(params_col), np.max(params_col))

    summary += '=' * 75 + '\n'
    return summary