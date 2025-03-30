import numpy as np
import pandas as pd
from spglm.family import Gaussian
from spglm.glm import GLM
from diagnostics import get_AICc
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


def summaryModel(self):
    """Summary of basic model information"""
    summary = '=' * 80 + '\n'
    summary += "MODEL SUMMARY\n"
    summary += '=' * 80 + '\n'
    summary += "%-25s %25s\n" % ('Model type:', 'SGWR with ' + self.family.__class__.__name__)
    summary += "%-25s %25d\n" % ('Number of observations:', self.n)
    summary += "%-25s %25d\n" % ('Number of covariates:', self.k)
    if hasattr(self.model, 'variables') and self.model.variables:
        summary += "\nVariables:\n"
        for i, var in enumerate(self.model.variables):
            summary += f"{i+1}. {var}\n"
    summary += '\n'
    return summary

def calculate_vif(X):
    """Calculate Variance Inflation Factors"""
    X = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def summaryGLM(self):
    """Summary of global regression results"""
    # Get variable names
    if hasattr(self.model, 'variables') and self.model.variables:
        XNames = self.model.variables.copy()  # Make a copy to avoid modifying original
        if self.model.constant:
            XNames = ['Intercept'] + XNames
    else:
        XNames = ["X" + str(i) for i in range(self.k)]

    # Ensure we have the right number of variables
    n_vars = len(XNames)
    if n_vars != self.k:
        if self.model.constant:
            XNames = ['Intercept'] + ["X" + str(i) for i in range(self.k-1)]
        else:
            XNames = ["X" + str(i) for i in range(self.k)]
    
    glm_rslt = GLM(self.model.y, self.model.X, constant=self.model.constant,
                   family=self.family).fit()

    summary = "GLOBAL REGRESSION RESULTS\n"
    summary += '=' * 80 + '\n'

    # Model diagnostics
    if isinstance(self.family, Gaussian):
        summary += "%-40s %20.3f\n" % ('Residual sum of squares:', glm_rslt.deviance)
        summary += "%-40s %20.3f\n" % ('R-squared:', glm_rslt.D2)
        summary += "%-40s %20.3f\n" % ('Adj. R-squared:', glm_rslt.adj_D2)
    else:
        summary += "%-40s %20.3f\n" % ('Deviance:', glm_rslt.deviance)
        summary += "%-40s %20.3f\n" % ('Percent deviance explained:', glm_rslt.D2)
        summary += "%-40s %20.3f\n" % ('Adj. percent deviance explained:', glm_rslt.adj_D2)
    
    summary += "%-40s %20.3f\n" % ('Log-likelihood:', glm_rslt.llf)
    summary += "%-40s %20.3f\n" % ('AIC:', glm_rslt.aic)
    summary += "%-40s %20.3f\n" % ('AICc:', get_AICc(glm_rslt))
    summary += "%-40s %20.3f\n\n" % ('BIC:', glm_rslt.bic)

    # Coefficient table - ensure we don't exceed bounds
    summary += "COEFFICIENT ESTIMATES\n"
    summary += '-' * 80 + '\n'
    summary += "%-20s %10s %10s %10s %10s\n" % ('Variable', 'Estimate', 'Std.Error', 
                                               't-value', 'p-value')
    summary += '-' * 80 + '\n'
    
    for i in range(min(len(XNames), len(glm_rslt.params))):  # Safeguard against index errors
        summary += "%-20s %10.3f %10.3f %10.3f %10.3f\n" % (
            XNames[i], 
            glm_rslt.params[i], 
            glm_rslt.bse[i] if i < len(glm_rslt.bse) else np.nan,
            glm_rslt.tvalues[i] if i < len(glm_rslt.tvalues) else np.nan,
            glm_rslt.pvalues[i] if i < len(glm_rslt.pvalues) else np.nan)
    
    # VIF table - handle constant term properly
    summary += "\nVARIANCE INFLATION FACTORS\n"
    summary += '-' * 80 + '\n'
    try:
        X_df = pd.DataFrame(self.model.X, columns=XNames[1:] if self.model.constant else XNames)
        vif_data = calculate_vif(X_df)
        summary += vif_data.to_string(index=False)
    except Exception as e:
        summary += f"Could not calculate VIF: {str(e)}\n"
    summary += "\n\n"

    return summary

def summarySGWR(self):
    """Summary of SGWR-specific results"""
    # Get variable names
    if hasattr(self.model, 'variables') and self.model.variables:
        XNames = self.model.variables
        if self.model.constant:
            XNames = ['Intercept'] + XNames
    else:
        XNames = ["X" + str(i) for i in range(self.k)]

    summary = "SGWR MODEL RESULTS\n"
    summary += '=' * 80 + '\n'

    # Model specifications
    summary += "MODEL SPECIFICATIONS\n"
    summary += '-' * 80 + '\n'
    kernel_type = 'Fixed' if self.model.fixed else 'Adaptive'
    summary += "%-30s %20s\n" % ('Kernel type:', f"{kernel_type} {self.model.kernel}")
    
    if hasattr(self.model, 'bw') and isinstance(self.model.bw, (list, np.ndarray)):
        summary += "\nVARIABLE-SPECIFIC PARAMETERS\n"
        summary += '-' * 80 + '\n'
        summary += "%-15s %12s %12s %12s %12s %12s\n" % (
            'Variable', 'Bandwidth', 'Alpha', 'ENP_j', 'DoD_j', 'Adj t-val')
        summary += '-' * 80 + '\n'
        
        for i in range(min(self.k, len(XNames))):  # Ensure we don't exceed bounds
            var_name = XNames[i]
            
            # Safely get bandwidth value
            bw_val = self.model.bw[i] if i < len(self.model.bw) else np.nan
            
            # Safely get alpha value
            alpha_val = (self.model.var_alphas.get(var_name, self.model.bt_value) 
                        if hasattr(self.model, 'var_alphas') else np.nan)
            alpha_val = np.nan if alpha_val is None else alpha_val  # Ensure no None
            
            # Safely get ENP value
            try:
                enp_val = (np.mean(self.influ[:,i]) 
                        if hasattr(self, 'influ') and self.influ is not None 
                        and i < self.influ.shape[1] 
                        else np.mean(self.influ) if hasattr(self, 'influ') else np.nan)
            except:
                enp_val = np.nan
            enp_val = np.nan if enp_val is None else enp_val  # Ensure no None
                
            # Calculate DoD
            dod_val = (1 - (enp_val / self.ENP) 
                    if hasattr(self, 'ENP') and self.ENP is not None 
                    and enp_val is not None and self.ENP != 0 
                    else np.nan)
            dod_val = np.nan if dod_val is None else dod_val  # Ensure no None
            
            # Get adjusted t-value
            adj_t_val = (self.critical_tval(self.adj_alpha[1]) 
                        if hasattr(self, 'adj_alpha') and hasattr(self, 'critical_tval') 
                        and len(self.adj_alpha) > 1 
                        else np.nan)
            adj_t_val = np.nan if adj_t_val is None else adj_t_val  # Ensure no None
            
            # Format the string with guaranteed numeric values
            summary += "%-15s %12.3f %12.3f %12.3f %12.3f %12.3f\n" % (
                var_name, 
                float(bw_val if bw_val is not None else np.nan),  # Convert to float
                float(alpha_val if alpha_val is not None else np.nan),
                float(enp_val if enp_val is not None else np.nan),
                float(dod_val if dod_val is not None else np.nan),
                float(adj_t_val if adj_t_val is not None else np.nan))

    # Model diagnostics
    summary += "\nMODEL DIAGNOSTICS\n"
    summary += '-' * 80 + '\n'
    
    if isinstance(self.family, Gaussian):
        summary += "%-40s %20.3f\n" % ('Residual sum of squares:', self.resid_ss if hasattr(self, 'resid_ss') else np.nan)
        summary += "%-40s %20.3f\n" % ('R-squared:', self.R2 if hasattr(self, 'R2') else np.nan)
        summary += "%-40s %20.3f\n" % ('Adj. R-squared:', self.adj_R2 if hasattr(self, 'adj_R2') else np.nan)
        summary += "%-40s %20.3f\n" % ('Sigma estimate:', np.sqrt(self.sigma2) if hasattr(self, 'sigma2') else np.nan)
    else:
        summary += "%-40s %20.3f\n" % ('Deviance:', self.deviance if hasattr(self, 'deviance') else np.nan)
        summary += "%-40s %20.3f\n" % ('Percent deviance explained:', self.D2 if hasattr(self, 'D2') else np.nan)
        summary += "%-40s %20.3f\n" % ('Adj. percent deviance explained:', self.adj_D2 if hasattr(self, 'adj_D2') else np.nan)
    
    # Safely format remaining diagnostics
    summary += "%-40s %20.3f\n" % ('Effective parameters (ENP):', self.ENP if hasattr(self, 'ENP') else np.nan)
    summary += "%-40s %20.3f\n" % ('Degrees of freedom:', self.df_model if hasattr(self, 'df_model') else np.nan)
    summary += "%-40s %20.3f\n" % ('Log-likelihood:', self.llf if hasattr(self, 'llf') else np.nan)
    summary += "%-40s %20.3f\n" % ('AIC:', self.aic if hasattr(self, 'aic') else np.nan)
    summary += "%-40s %20.3f\n" % ('AICc:', self.aicc if hasattr(self, 'aicc') else np.nan)
    summary += "%-40s %20.3f\n" % ('BIC:', self.bic if hasattr(self, 'bic') else np.nan)

    if hasattr(self, 'adj_alpha'):
        summary += "%-40s %20.3f\n" % ('Adj. alpha (95%):', self.adj_alpha[1] if len(self.adj_alpha) > 1 else np.nan)
        summary += "%-40s %20.3f\n" % ('Adj. critical t-value:', 
                                      self.critical_tval(self.adj_alpha[1]) if len(self.adj_alpha) > 1 else np.nan)
    else:
        summary += "%-40s %20s\n" % ('Adj. alpha (95%):', 'N/A')
        summary += "%-40s %20s\n" % ('Adj. critical t-value:', 'N/A')

    # Alpha testing results if available
    if hasattr(self.model, 'alpha_values') and hasattr(self.model, 'aiccs'):
        summary += "\nALPHA TESTING RESULTS\n"
        summary += '-' * 80 + '\n'
        summary += "%-15s %15s\n" % ('Alpha', 'AICc')
        summary += '-' * 80 + '\n'
        for alpha, aicc in zip(self.model.alpha_values, self.model.aiccs):
            summary += "%-15.3f %15.3f\n" % (alpha, aicc)
        best_idx = np.argmin(self.model.aiccs)
        summary += "\nBest alpha: %.3f (AICc = %.3f)\n" % (
            self.model.alpha_values[best_idx], self.model.aiccs[best_idx])

    # Parameter statistics
    summary += "\nPARAMETER ESTIMATES SUMMARY\n"
    summary += '-' * 80 + '\n'
    summary += "%-15s %10s %10s %10s %10s %10s\n" % (
        'Variable', 'Mean', 'StdDev', 'Min', 'Median', 'Max')
    summary += '-' * 80 + '\n'
    
    for i in range(self.k):
        params_col = self.params[:, i] if i < self.params.shape[1] else np.array([np.nan])
        summary += "%-15s %10.3f %10.3f %10.3f %10.3f %10.3f\n" % (
            XNames[i], 
            np.nanmean(params_col), 
            np.nanstd(params_col),
            np.nanmin(params_col), 
            np.nanmedian(params_col),
            np.nanmax(params_col))

    summary += '=' * 80 + '\n'
    return summary