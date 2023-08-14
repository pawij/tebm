import numpy as np
from scipy import linalg, stats

def log_multivariate_normal_density(X, means, covars, covariance_type='diag'):
    """Compute the log probability under a multivariate Gaussian distribution.
    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds to a
        single data point.
    means : array_like, shape (n_components, n_features)
        List of n_features-dimensional mean vectors for n_components Gaussians.
        Each row corresponds to a single mean vector.
    covars : array_like
        List of n_components covariance parameters for each Gaussian. The shape
        depends on `covariance_type`:
            (n_components, n_features)      if 'spherical',
            (n_features, n_features)    if 'tied',
            (n_components, n_features)    if 'diag',
            (n_components, n_features, n_features) if 'full'
    covariance_type : string
        Type of the covariance parameters.  Must be one of
        'spherical', 'tied', 'diag', 'full'.  Defaults to 'diag'.
    Returns
    -------
    lpr : array_like, shape (n_samples, n_components)
        Array containing the log probabilities of each data point in
        X under each of the n_components multivariate Gaussian distributions.
    """
    log_multivariate_normal_density_dict = {
        'spherical': _log_multivariate_normal_density_spherical,
        'tied': _log_multivariate_normal_density_tied,
        'diag': _log_multivariate_normal_density_diag,
        'full': _log_multivariate_normal_density_full}

    return log_multivariate_normal_density_dict[covariance_type](
        X, means, covars
    )

def _log_multivariate_normal_density_diag(X, means, covars):
    """Compute Gaussian log-density at X for a diagonal model."""
    # X: (ns, nf); means: (nc, nf); covars: (nc, nf) -> (ns, nc)
    n_samples, n_dim = X.shape
    # Avoid 0 log 0 = nan in degenerate covariance case.
    covars = np.maximum(covars, np.finfo(float).tiny)
    with np.errstate(over="ignore"):
        return -0.5 * (n_dim * np.log(2 * np.pi)
                       + np.log(covars).sum(axis=-1)
                       + ((X[:, None, :] - means) ** 2 / covars).sum(axis=-1))
    """
    with np.errstate(over="ignore"):
        S = weights[1]
        weights = weights[0]
        def calc_coeff(sig):
            return 1./np.sqrt(np.pi*2.0)*1./sig
        def calc_exp(x,mu,sig):
            x = (x-mu)/sig
            return np.exp(-.5*x*x)
        def normPdf(x,mu,sig):
            return calc_coeff(sig)*calc_exp(x,mu,sig)
        prob = normPdf(X[:, None, :], means, np.sqrt(covars))*weights
        #        prob = stats.norm.pdf(X[:, None, :], loc=means, scale=np.sqrt(covars))*weights
        # normalise
        for i in range(prob.shape[0]):
            for j in range(prob.shape[2]):
                bm_pos = np.where(S == j)[0][0]
                prob_h = prob[i,:,j][0]
                prob_d = prob[i,:,j][-1]
                if prob_h==0 and prob_d==0:
                    print (X)
                    print (means)
                    print (np.sqrt(covars))
                    print (weights)
                    print (prob)
                    quit()
                if np.isnan(prob_h) or np.isnan(prob_d):
                    prob_h = .5
                else:
                    prob_h = prob_h / (prob_h+prob_d)
                prob_d = 1-prob_h
                prob[i,:bm_pos+1,j] = prob_h
                prob[i,bm_pos+1:,j] = prob_d
        prob[prob == 0] = np.finfo(float).eps
        like = np.nansum(np.log(prob),axis=-1)
        return like
    """
def _log_multivariate_normal_density_spherical(X, means, covars):
    """Compute Gaussian log-density at X for a spherical model."""
    cv = covars.copy()
    if covars.ndim == 1:
        cv = cv[:, np.newaxis]
    if cv.shape[1] == 1:
        cv = np.tile(cv, (1, X.shape[-1]))
    return _log_multivariate_normal_density_diag(X, means, cv)


def _log_multivariate_normal_density_tied(X, means, covars):
    """Compute Gaussian log-density at X for a tied model."""
    cv = np.tile(covars, (means.shape[0], 1, 1))
    return _log_multivariate_normal_density_full(X, means, cv)


def _log_multivariate_normal_density_full(X, means, covars, min_covar=1.e-7):
    """Log probability for full covariance matrices."""
    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            try:
                cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim),
                                          lower=True)
            except linalg.LinAlgError:
                raise ValueError("'covars' must be symmetric, "
                                 "positive-definite")

        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) +
                                 n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob
