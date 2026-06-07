//! Cramér-Rao bound: the fundamental limit on estimator variance.
//!
//! For any unbiased estimator θ̂ of parameter θ:
//! Var(θ̂) ≥ I(θ)⁻¹
//!
//! An estimator that achieves this bound is called efficient.

use serde::{Deserialize, Serialize};

use crate::metric::FisherRaoMetric;
use crate::types::{Distribution, Matrix};

/// Cramér-Rao bound computation and estimator efficiency analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CramerRaoBound {
    /// The distribution family.
    pub distribution: Distribution,
}

impl CramerRaoBound {
    /// Create for a given distribution.
    pub fn new(distribution: Distribution) -> Self {
        Self { distribution }
    }

    /// Compute the Cramér-Rao lower bound: I(θ)⁻¹.
    ///
    /// Returns None if the Fisher matrix is singular.
    pub fn bound(&self) -> Option<Matrix> {
        let fim = self.fisher_matrix();
        fim.inverse()
    }

    /// Compute the bound at specific parameter values.
    pub fn bound_at(&self, params: &crate::types::Params) -> Option<Matrix> {
        let metric = FisherRaoMetric::new(self.distribution.clone());
        let fim = metric.fisher_matrix_at(params);
        fim.inverse()
    }

    /// Get the Fisher information matrix.
    pub fn fisher_matrix(&self) -> Matrix {
        FisherRaoMetric::new(self.distribution.clone()).fisher_matrix()
    }

    /// Check if an estimator is efficient (achieves the CRB).
    ///
    /// For a scalar parameter: efficient iff Var(θ̂) = 1/I(θ).
    /// For vector: efficient iff Cov(θ̂) = I(θ)⁻¹.
    pub fn is_efficient(&self, estimator_cov: &Matrix) -> EfficiencyResult {
        let fim = self.fisher_matrix();
        let crb = match fim.inverse() {
            Some(crb) => crb,
            None => {
                return EfficiencyResult {
                    is_efficient: false,
                    crb_matrix: Matrix::zeros(0),
                    estimator_cov: estimator_cov.clone(),
                    efficiency_ratio: vec![f64::NAN],
                };
            }
        };

        // Check if Cov(θ̂) ≈ I⁻¹
        let dim = fim.n;
        let mut ratios = Vec::with_capacity(dim);
        let mut all_efficient = true;

        for i in 0..dim {
            let cov_ii = estimator_cov.get(i, i);
            let crb_ii = crb.get(i, i);
            if crb_ii > 1e-15 {
                let ratio = crb_ii / cov_ii;
                ratios.push(ratio);
                if (ratio - 1.0).abs() > 0.01 {
                    all_efficient = false;
                }
            } else {
                ratios.push(f64::NAN);
                all_efficient = false;
            }
        }

        EfficiencyResult {
            is_efficient: all_efficient,
            crb_matrix: crb,
            estimator_cov: estimator_cov.clone(),
            efficiency_ratio: ratios,
        }
    }

    /// Check if an estimator is efficient against the CRB for a sample of size n.
    pub fn is_efficient_with_sample(&self, estimator_cov: &Matrix, n: usize) -> EfficiencyResult {
        let sample_crb = match self.bound_for_sample(n) {
            Some(crb) => crb,
            None => {
                return EfficiencyResult {
                    is_efficient: false,
                    crb_matrix: Matrix::zeros(0),
                    estimator_cov: estimator_cov.clone(),
                    efficiency_ratio: vec![f64::NAN],
                };
            }
        };

        let dim = estimator_cov.n;
        let mut ratios = Vec::with_capacity(dim);
        let mut all_efficient = true;

        for i in 0..dim {
            let cov_ii = estimator_cov.get(i, i);
            let crb_ii = sample_crb.get(i, i);
            if crb_ii > 1e-15 {
                let ratio = crb_ii / cov_ii;
                ratios.push(ratio);
                if (ratio - 1.0).abs() > 0.05 {
                    all_efficient = false;
                }
            } else {
                ratios.push(f64::NAN);
                all_efficient = false;
            }
        }

        EfficiencyResult {
            is_efficient: all_efficient,
            crb_matrix: sample_crb,
            estimator_cov: estimator_cov.clone(),
            efficiency_ratio: ratios,
        }
    }

    /// Compute the scalar CRB for a single parameter.
    /// CRB = 1 / I(θ)
    pub fn scalar_bound(&self) -> f64 {
        let fim = self.fisher_matrix();
        assert!(fim.n == 1, "scalar_bound only for 1-parameter families");
        let i = fim.get(0, 0);
        if i.abs() < 1e-15 {
            f64::INFINITY
        } else {
            1.0 / i
        }
    }

    /// Check if an MVUE (Minimum Variance Unbiased Estimator) can exist.
    ///
    /// An MVUE exists iff the score function is a linear function of
    /// sufficient statistics (exponential family condition).
    pub fn mvue_exists(&self) -> bool {
        // All distributions in our library are exponential families
        matches!(
            self.distribution,
            Distribution::Normal { .. }
                | Distribution::Bernoulli { .. }
                | Distribution::Multinomial { .. }
                | Distribution::Exponential { .. }
        )
    }

    /// Compute the Fisher information number (total information).
    /// This is n · I(θ) for n i.i.d. observations.
    pub fn information_for_sample(&self, n: usize) -> Matrix {
        self.fisher_matrix().scale(n as f64)
    }

    /// Compute the CRB for a sample of size n.
    pub fn bound_for_sample(&self, n: usize) -> Option<Matrix> {
        self.information_for_sample(n).inverse()
    }
}

/// Result of checking estimator efficiency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyResult {
    /// Whether the estimator achieves the CRB.
    pub is_efficient: bool,
    /// The CRB matrix I(θ)⁻¹.
    pub crb_matrix: Matrix,
    /// The estimator's covariance matrix.
    pub estimator_cov: Matrix,
    /// Ratio CRB[i] / Var(θ̂[i]) for each parameter.
    /// Values close to 1.0 indicate efficiency.
    pub efficiency_ratio: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Params;

    #[test]
    fn test_bernoulli_crb() {
        let crb = CramerRaoBound::new(Distribution::Bernoulli { p: 0.5 });
        let bound = crb.scalar_bound();
        // I(p=0.5) = 4, CRB = 0.25
        assert!((bound - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_normal_crb() {
        let crb = CramerRaoBound::new(Distribution::Normal {
            mu: 0.0,
            sigma: 2.0,
        });
        let bound = crb.bound().expect("should be invertible");
        // I⁻¹ = [[σ², 0], [0, σ²/2]] = [[4, 0], [0, 2]]
        assert!((bound.get(0, 0) - 4.0).abs() < 1e-10);
        assert!((bound.get(1, 1) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_efficient_estimator() {
        // For Normal(μ, σ²), the sample mean X̄ is efficient for μ
        // Var(X̄) = σ²/n, CRB(μ) for sample = σ²/n
        let sigma = 2.0;
        let n = 10;
        let crb = CramerRaoBound::new(Distribution::Normal { mu: 0.0, sigma });
        // The CRB for a sample of size n
        let sample_bound = crb.bound_for_sample(n).unwrap();
        let cov = Matrix::from_2x2(
            sigma * sigma / n as f64,
            0.0,
            0.0,
            sigma * sigma / (2.0 * n as f64),
        );
        let result = crb.is_efficient_with_sample(&cov, n);
        assert!(result.is_efficient);
    }

    #[test]
    fn test_inefficient_estimator() {
        // Use a covariance larger than CRB
        let crb = CramerRaoBound::new(Distribution::Bernoulli { p: 0.5 });
        let cov = Matrix::from_scalar(0.5); // CRB is 0.25, so 0.5 > CRB
        let result = crb.is_efficient(&cov);
        assert!(!result.is_efficient);
        assert!(result.efficiency_ratio[0] < 1.0);
    }

    #[test]
    fn test_mvue_exists() {
        assert!(
            CramerRaoBound::new(Distribution::Normal {
                mu: 0.0,
                sigma: 1.0
            })
            .mvue_exists()
        );
        assert!(CramerRaoBound::new(Distribution::Bernoulli { p: 0.5 }).mvue_exists());
        assert!(CramerRaoBound::new(Distribution::Exponential { lambda: 1.0 }).mvue_exists());
    }

    #[test]
    fn test_information_scales_with_sample() {
        let crb = CramerRaoBound::new(Distribution::Bernoulli { p: 0.5 });
        let info_n1 = crb.fisher_matrix();
        let info_n10 = crb.information_for_sample(10);
        assert!((info_n10.get(0, 0) - 10.0 * info_n1.get(0, 0)).abs() < 1e-10);
    }

    #[test]
    fn test_crb_at_params() {
        let crb = CramerRaoBound::new(Distribution::Normal {
            mu: 0.0,
            sigma: 1.0,
        });
        let params = Params::new(vec!["mu", "sigma"], vec![0.0, 3.0]);
        let bound = crb.bound_at(&params).unwrap();
        // I⁻¹ = [[9, 0], [0, 4.5]]
        assert!((bound.get(0, 0) - 9.0).abs() < 1e-10);
        assert!((bound.get(1, 1) - 4.5).abs() < 1e-10);
    }

    #[test]
    fn test_efficiency_ratios() {
        let crb = CramerRaoBound::new(Distribution::Normal {
            mu: 0.0,
            sigma: 1.0,
        });
        let cov = Matrix::from_2x2(1.0, 0.0, 0.0, 0.6); // CRB diagonal = [1.0, 0.5]
        let result = crb.is_efficient(&cov);
        assert!(!result.is_efficient);
        assert!((result.efficiency_ratio[0] - 1.0).abs() < 0.01); // μ is efficient
        assert!(result.efficiency_ratio[1] < 0.9); // σ is not efficient
    }
}
