//! Fisher-Rao metric: the fundamental Riemannian metric on statistical manifolds.
//!
//! The Fisher information matrix gᵢⱼ(θ) = E[∂ᵢ log p(x|θ) · ∂ⱼ log p(x|θ)]
//! defines a metric tensor that makes the space of probability distributions
//! into a Riemannian manifold.

use serde::{Deserialize, Serialize};

use crate::types::{Distribution, Matrix, Params};

/// The Fisher-Rao metric tensor for parametric families.
///
/// This is the core abstraction: given a parametric family of distributions,
/// compute the Fisher information matrix which serves as the Riemannian metric
/// on the statistical manifold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FisherRaoMetric {
    /// The distribution family.
    pub distribution: Distribution,
}

impl FisherRaoMetric {
    /// Create a new Fisher-Rao metric for the given distribution family.
    pub fn new(distribution: Distribution) -> Self {
        Self { distribution }
    }

    /// Compute the Fisher information matrix gᵢⱼ(θ) at the current parameters.
    ///
    /// For each supported distribution, this returns the analytically known
    /// Fisher information matrix.
    pub fn fisher_matrix(&self) -> Matrix {
        match &self.distribution {
            Distribution::Normal { mu: _, sigma } => self.fisher_normal(*sigma),
            Distribution::Bernoulli { p } => self.fisher_bernoulli(*p),
            Distribution::Multinomial { n: _, probs } => self.fisher_multinomial(probs),
            Distribution::Exponential { lambda } => self.fisher_exponential(*lambda),
        }
    }

    /// Compute the Fisher information matrix element gᵢⱼ by numerical integration.
    ///
    /// Uses Gauss-Hermite quadrature for Normal, direct computation otherwise.
    /// This is a fallback for distributions where analytical forms aren't implemented.
    pub fn fisher_matrix_numerical(&self) -> Matrix {
        match &self.distribution {
            Distribution::Normal { mu, sigma } => self.fisher_normal_numerical(*mu, *sigma),
            Distribution::Bernoulli { p } => self.fisher_bernoulli(*p),
            Distribution::Multinomial { n: _, probs } => self.fisher_multinomial(probs),
            Distribution::Exponential { lambda } => self.fisher_exponential(*lambda),
        }
    }

    /// Compute the Fisher information matrix at an arbitrary parameter point.
    pub fn fisher_matrix_at(&self, params: &Params) -> Matrix {
        match &self.distribution {
            Distribution::Normal { .. } => {
                let sigma = params.values[1];
                self.fisher_normal(sigma)
            }
            Distribution::Bernoulli { .. } => {
                let p = params.values[0];
                self.fisher_bernoulli(p)
            }
            Distribution::Multinomial { n: _, .. } => {
                let k = self.distribution.param_dim() + 1;
                let mut probs = params.values.clone();
                probs.push(1.0 - probs.iter().sum::<f64>());
                if probs.len() < k {
                    // shouldn't happen but be safe
                }
                self.fisher_multinomial(&probs)
            }
            Distribution::Exponential { .. } => {
                let lambda = params.values[0];
                self.fisher_exponential(lambda)
            }
        }
    }

    /// Verify positive semi-definiteness of the Fisher information matrix.
    pub fn is_psd(&self) -> bool {
        self.fisher_matrix().is_psd()
    }

    // -- Analytical implementations --

    fn fisher_normal(&self, sigma: f64) -> Matrix {
        // I(μ, σ) = [[1/σ², 0], [0, 2/σ²]]
        let s2 = sigma * sigma;
        Matrix::from_2x2(1.0 / s2, 0.0, 0.0, 2.0 / s2)
    }

    fn fisher_bernoulli(&self, p: f64) -> Matrix {
        // I(p) = 1 / (p(1-p))
        Matrix::from_scalar(1.0 / (p * (1.0 - p)))
    }

    fn fisher_multinomial(&self, probs: &[f64]) -> Matrix {
        // Iᵢⱼ = n·(δᵢⱼ/pᵢ + 1/pₖ) where pₖ = 1 - Σpᵢ
        let k = probs.len();
        let n_params = k - 1;
        if n_params == 0 {
            return Matrix::from_scalar(0.0);
        }
        let p_k = 1.0 - probs.iter().sum::<f64>();
        let mut m = Matrix::zeros(n_params);
        for (i, row) in m.data.chunks_mut(n_params).enumerate().take(n_params) {
            for (j, elem) in row.iter_mut().enumerate() {
                if i == j {
                    *elem = 1.0 / probs[i] + 1.0 / p_k;
                } else {
                    *elem = 1.0 / p_k;
                }
            }
        }
        m
    }

    fn fisher_exponential(&self, lambda: f64) -> Matrix {
        // I(λ) = 1/λ²
        Matrix::from_scalar(1.0 / (lambda * lambda))
    }

    /// Numerical Fisher matrix for Normal via Gauss-Hermite quadrature.
    fn fisher_normal_numerical(&self, mu: f64, sigma: f64) -> Matrix {
        // 15-point Gauss-Hermite quadrature
        let nodes: [f64; 15] = [
            -4.499_990_898_291_425,
            -3.669_950_530_413_66,
            -2.967_166_928_057_565,
            -2.325_605_580_550_055,
            -1.719_762_093_236_616,
            -1.136_115_665_210_31,
            -0.565_069_389_045_071,
            0.0,
            0.565_069_389_045_071,
            1.136_115_665_210_31,
            1.719_762_093_236_616,
            2.325_605_580_550_055,
            2.967_166_928_057_565,
            3.669_950_530_413_66,
            4.499_990_898_291_425,
        ];
        let weights: [f64; 15] = [
            1.034_401_848_045_048e-9,
            5.465_077_734_905_64e-7,
            4.034_689_492_466_9e-5,
            1.017_363_073_032_734e-3,
            1.019_315_379_618_724e-2,
            5.758_446_351_015_426e-2,
            1.694_681_286_652_763e-1,
            2.660_316_395_064_925e-1,
            1.694_681_286_652_763e-1,
            5.758_446_351_015_426e-2,
            1.019_315_379_618_724e-2,
            1.017_363_073_032_734e-3,
            4.034_689_492_466_9e-5,
            5.465_077_734_905_64e-7,
            1.034_401_848_045_048e-9,
        ];

        let mut g = Matrix::zeros(2);
        for (i, &node) in nodes.iter().enumerate() {
            let x = mu + sigma * std::f64::consts::SQRT_2 * node;

            // Score for Normal(μ, σ): ∂log p/∂μ = (x-μ)/σ², ∂log p/∂σ = ((x-μ)²-σ²)/σ³
            let dx = x - mu;
            let s_mu = dx / (sigma * sigma);
            let s_sigma = (dx * dx - sigma * sigma) / (sigma * sigma * sigma);

            let w = weights[i];
            g.set(0, 0, g.get(0, 0) + w * s_mu * s_mu);
            g.set(0, 1, g.get(0, 1) + w * s_mu * s_sigma);
            g.set(1, 0, g.get(1, 0) + w * s_sigma * s_mu);
            g.set(1, 1, g.get(1, 1) + w * s_sigma * s_sigma);
        }

        // Gauss-Hermite integration: multiply by 1/√π
        let scale = 1.0 / std::f64::consts::SQRT_2;
        for i in 0..4 {
            g.data[i] *= scale;
        }
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_fisher_matrix() {
        let d = Distribution::Normal {
            mu: 0.0,
            sigma: 2.0,
        };
        let m = FisherRaoMetric::new(d);
        let fim = m.fisher_matrix();
        assert!((fim.get(0, 0) - 0.25).abs() < 1e-10); // 1/σ² = 1/4
        assert!((fim.get(0, 1)).abs() < 1e-10);
        assert!((fim.get(1, 0)).abs() < 1e-10);
        assert!((fim.get(1, 1) - 0.5).abs() < 1e-10); // 2/σ² = 2/4
    }

    #[test]
    fn test_bernoulli_fisher_matrix() {
        let d = Distribution::Bernoulli { p: 0.5 };
        let m = FisherRaoMetric::new(d);
        let fim = m.fisher_matrix();
        assert!((fim.get(0, 0) - 4.0).abs() < 1e-10); // 1/(0.5*0.5)
    }

    #[test]
    fn test_exponential_fisher_matrix() {
        let d = Distribution::Exponential { lambda: 3.0 };
        let m = FisherRaoMetric::new(d);
        let fim = m.fisher_matrix();
        assert!((fim.get(0, 0) - 1.0 / 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_psd_normal() {
        let d = Distribution::Normal {
            mu: 0.0,
            sigma: 1.0,
        };
        let m = FisherRaoMetric::new(d);
        assert!(m.is_psd());
    }

    #[test]
    fn test_psd_bernoulli() {
        let d = Distribution::Bernoulli { p: 0.3 };
        let m = FisherRaoMetric::new(d);
        assert!(m.is_psd());
    }

    #[test]
    fn test_symmetric() {
        let d = Distribution::Normal {
            mu: 5.0,
            sigma: 3.0,
        };
        let m = FisherRaoMetric::new(d);
        let fim = m.fisher_matrix();
        assert!(fim.is_symmetric(1e-10));
    }

    #[test]
    fn test_fisher_at_different_params() {
        let d = Distribution::Normal {
            mu: 0.0,
            sigma: 1.0,
        };
        let m = FisherRaoMetric::new(d);
        let params = Params::new(vec!["mu", "sigma"], vec![0.0, 5.0]);
        let fim = m.fisher_matrix_at(&params);
        assert!((fim.get(0, 0) - 0.04).abs() < 1e-10); // 1/25
    }
}
