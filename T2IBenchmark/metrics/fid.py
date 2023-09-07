import numpy as np
from scipy import linalg


class FIDStats:
    
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma
    
    def to_npz(self, path: str):
        np.savez(path, mu=self.mu, sigma=self.sigma)
        
    @classmethod
    def from_features(cls, features: np.ndarray) -> "FIDStats":
        mu, sigma = np.mean(features, axis=0), np.cov(features, rowvar=False)
        return cls(mu, sigma)
    
    @classmethod
    def from_npz(cls, path: str) -> "FIDStats":
        data = np.load(path)
        mu = data['mu']
        sigma = data['sigma']
        return cls(mu, sigma)
    
    
def frechet_distance(stat1, stat2, eps=1e-6):
    mu1, sigma1 = stat1.mu, stat1.sigma
    mu2, sigma2 = stat2.mu, stat2.sigma
    
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
