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

    XNames = ["X" + str(i) for i in range(self.k)]
    glm_rslt = GLM(self.model.y, self.model.X, constant=False,
                   family=self.family).fit()

    summary = "%s\n" % ('Global Regression Results')
    summary += '-' * 75 + '\n'

    if isinstance(self.family, Gaussian):
        summary += "%-62s %12.3f\n" % ('Residual sum of squares:',
                                       glm_rslt.deviance)
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
        summary += "%-62s %12.3f\n" % ('Percent deviance explained:',
                                       glm_rslt.D2)
        summary += "%-62s %12.3f\n\n" % ('Adj. percent deviance explained:',
                                         glm_rslt.adj_D2)

    summary += "%-31s %10s %10s %10s %10s\n" % ('Variable', 'Est.', 'SE',
                                                't(Est/SE)', 'p-value')
    summary += "%-31s %10s %10s %10s %10s\n" % ('-' * 31, '-' * 10, '-' * 10,
                                                '-' * 10, '-' * 10)
    for i in range(self.k):
        summary += "%-31s %10.3f %10.3f %10.3f %10.3f\n" % (
            XNames[i], glm_rslt.params[i], glm_rslt.bse[i],
            glm_rslt.tvalues[i], glm_rslt.pvalues[i])
    summary += "\n"

    X = pd.DataFrame(self.model.X, columns=XNames)
    X = sm.add_constant(X)
    vif_data = calculate_vif(X)
    summary += "%s\n" % ('Variance Inflation Factor (VIF)')
    '-' * 75 + '\n'
    summary += vif_data.to_string(index=False)
    summary += "\n"

    return summary


def summarySGWR(self):
    XNames = ["X" + str(i) for i in range(self.k)]

    summary = "%s\n" % ('Similarity Geographically Weighted Regression (SGWR) results')
    summary += '-' * 75 + '\n'

    if self.model.fixed:
        summary += "%-50s %20s\n" % ('Spatial kernel:',
                                     'Fixed ' + self.model.kernel)
    else:
        summary += "%-54s %20s\n" % ('Spatial kernel:',
                                     'Adaptive ' + self.model.kernel)

    summary += "%-62s %12.3f\n" % ('Bandwidth used:', self.model.bw)

    summary += "\n%s\n" % ('Diagnostic information')
    summary += '-' * 75 + '\n'

    if isinstance(self.family, Gaussian):
        summary += "%-62s %12.3f\n" % ('Residual sum of squares:',
                                       self.resid_ss)
        summary += "%-62s %12.3f\n" % (
            'Effective number of parameters (trace(S)):', self.tr_S)
        summary += "%-62s %12.3f\n" % ('Degree of freedom (n - trace(S)):',
                                       self.df_model)
        summary += "%-62s %12.3f\n" % ('Sigma estimate:', np.sqrt(self.sigma2))
        summary += "%-62s %12.3f\n" % ('Log-likelihood:', self.llf)
        summary += "%-62s %12.3f\n" % ('AIC:', self.aic)
        summary += "%-62s %12.3f\n" % ('AICc:', self.aicc)
        summary += "%-62s %12.3f\n" % ('BIC:', self.bic)
        summary += "%-62s %12.3f\n" % ('R2:', self.R2)
        summary += "%-62s %12.3f\n" % ('Adjusted R2:', self.adj_R2)

    else:
        summary += "%-62s %12.3f\n" % (
            'Effective number of parameters (trace(S)):', self.tr_S)
        summary += "%-62s %12.3f\n" % ('Degree of freedom (n - trace(S)):',
                                       self.df_model)
        summary += "%-62s %12.3f\n" % ('Log-likelihood:', self.llf)
        summary += "%-62s %12.3f\n" % ('AIC:', self.aic)
        summary += "%-62s %12.3f\n" % ('AICc:', self.aicc)
        summary += "%-62s %12.3f\n" % ('BIC:', self.bic)
        summary += "%-60s %12.3f\n" % ('Percent deviance explained:', self.D2)
        summary += "%-60s %12.3f\n" % ('Adjusted percent deviance explained:',
                                       self.adj_D2)

    summary += "%-62s %12.3f\n" % ('Adj. alpha (95%):', self.adj_alpha[1])
    summary += "%-62s %12.3f\n" % ('Adj. critical t value (95%):',
                                   self.critical_tval(self.adj_alpha[1]))

    # Tambahkan informasi nilai alpha dan AICc
    if hasattr(self.model, 'alpha_values') and hasattr(self.model, 'aiccs'):
        summary += "\n%s\n" % ('Alpha Testing Results')
        summary += '-' * 75 + '\n'
        summary += "%-20s %10s\n" % ('Alpha', 'AICc')
        summary += "%-20s %10s\n" % ('-' * 20, '-' * 10)
        for alpha, aicc in zip(self.model.alpha_values, self.model.aiccs):
            summary += "%-20.2f %10.3f\n" % (alpha, aicc)

    summary += "\n%s\n" % ('Summary Statistics For SGWR Parameter Estimates')
    summary += '-' * 75 + '\n'
    summary += "%-20s %10s %10s %10s %10s %10s\n" % ('Variable', 'Mean', 'STD',
                                                     'Min', 'Median', 'Max')
    summary += "%-20s %10s %10s %10s %10s %10s\n" % (
        '-' * 20, '-' * 10, '-' * 10, '-' * 10, '-' * 10, '-' * 10)
    for i in range(self.k):
        summary += "%-20s %10.3f %10.3f %10.3f %10.3f %10.3f\n" % (
            XNames[i], np.mean(self.params[:, i]), np.std(self.params[:, i]),
            np.min(self.params[:, i]), np.median(self.params[:, i]),
            np.max(self.params[:, i]))


    summary += '=' * 75 + '\n'
    return summary