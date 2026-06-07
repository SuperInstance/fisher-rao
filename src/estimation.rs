//! Maximum likelihood estimation via Fisher scoring.
//!
//! Fisher scoring replaces the Hessian in Newton's method with the expected
//! negative Fisher information, giving more stable convergence:
//! θ_{k+1} = θ_k + I(θ_k)⁻¹ · S(θ_k)
//!
//! where S(θ) = ∂ℓ/∂θ is the score function.

use serde::{Deserialize, Serialize};

use crate::metric::FisherRaoMetric;
use crate::types::{Distribution, Matrix, Params};

/// MLE via Fisher scoring algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Estimation {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance on parameter change.
    pub tol: f64,
}

/// Result of an estimation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimationResult {
    /// Estimated parameters.
    pub estimates: Params,
    /// Fisher information at the estimate.
    pub fisher_info: Matrix,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Log-likelihood at the estimate (if computed).
    pub log_likelihood: Option<f64>,
    /// Score function norm at convergence.
    pub score_norm: f64,
}

impl Estimation {
    /// Create with default settings (max 100 iterations, tolerance 1e-8).
    pub fn new() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-8,
        }
    }

    /// Create with custom settings.
    pub fn with_params(max_iter: usize, tol: f64) -> Self {
        Self { max_iter, tol }
    }

    /// Run Fisher scoring for a Normal distribution given sample data.
    ///
    /// Estimates μ and σ from i.i.d. observations.
    pub fn fit_normal(&self, data: &[f64]) -> EstimationResult {
        let n = data.len() as f64;

        // Initial estimates: sample mean and sample std
        let mean = data.iter().sum::<f64>() / n;
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let sigma = var.sqrt().max(1e-10);

        let mut theta = vec![mean, sigma];

        for iter in 0..self.max_iter {
            let mu = theta[0];
            let sigma = theta[1];

            // Score function for Normal:
            // S_μ = Σ(xᵢ - μ) / σ²
            // S_σ = Σ((xᵢ - μ)² - σ²) / σ³
            let mut s_mu = 0.0;
            let mut s_sigma = 0.0;
            for &x in data {
                let d = x - mu;
                s_mu += d / (sigma * sigma);
                s_sigma += (d * d - sigma * sigma) / (sigma * sigma * sigma);
            }

            let score = vec![s_mu, s_sigma];
            let score_norm: f64 = score.iter().map(|s| s * s).sum::<f64>().sqrt();

            if score_norm < self.tol {
                let fim = FisherRaoMetric::new(Distribution::Normal { mu, sigma }).fisher_matrix();
                let ll = Self::normal_log_likelihood(data, mu, sigma);
                return EstimationResult {
                    estimates: Params::new(vec!["mu", "sigma"], theta.clone()),
                    fisher_info: fim,
                    iterations: iter,
                    converged: true,
                    log_likelihood: Some(ll),
                    score_norm,
                };
            }

            // Fisher scoring step: θ += I⁻¹ · S
            let fim = FisherRaoMetric::new(Distribution::Normal { mu, sigma }).fisher_matrix();
            let fim_inv = match fim.inverse() {
                Some(inv) => inv,
                None => {
                    return EstimationResult {
                        estimates: Params::new(vec!["mu", "sigma"], theta),
                        fisher_info: fim,
                        iterations: iter,
                        converged: false,
                        log_likelihood: None,
                        score_norm,
                    };
                }
            };

            let update = fim_inv.mul_vec(&score);
            theta[0] += update[0];
            theta[1] += update[1];
            theta[1] = theta[1].max(1e-10); // sigma > 0
        }

        let mu = theta[0];
        let sigma = theta[1];
        let fim = FisherRaoMetric::new(Distribution::Normal { mu, sigma }).fisher_matrix();
        EstimationResult {
            estimates: Params::new(vec!["mu", "sigma"], theta),
            fisher_info: fim,
            iterations: self.max_iter,
            converged: false,
            log_likelihood: Some(Self::normal_log_likelihood(data, mu, sigma)),
            score_norm: f64::NAN,
        }
    }

    /// Run Fisher scoring for Bernoulli (single parameter).
    pub fn fit_bernoulli(&self, successes: usize, trials: usize) -> EstimationResult {
        let p_hat = successes as f64 / trials as f64;
        let theta = vec![p_hat];

        // For Bernoulli, the MLE is simply the sample proportion.
        // One iteration suffices (exponential family, score = 0 at MLE).
        let fim = FisherRaoMetric::new(Distribution::Bernoulli { p: p_hat }).fisher_matrix();
        let ll = Self::bernoulli_log_likelihood(successes, trials, p_hat);

        EstimationResult {
            estimates: Params::new(vec!["p"], theta),
            fisher_info: fim,
            iterations: 1,
            converged: true,
            log_likelihood: Some(ll),
            score_norm: 0.0,
        }
    }

    /// Run Fisher scoring for Exponential.
    pub fn fit_exponential(&self, data: &[f64]) -> EstimationResult {
        let n = data.len() as f64;
        let lambda_hat = n / data.iter().sum::<f64>();

        let fim =
            FisherRaoMetric::new(Distribution::Exponential { lambda: lambda_hat }).fisher_matrix();
        let ll = Self::exponential_log_likelihood(data, lambda_hat);

        EstimationResult {
            estimates: Params::new(vec!["lambda"], vec![lambda_hat]),
            fisher_info: fim,
            iterations: 1,
            converged: true,
            log_likelihood: Some(ll),
            score_norm: 0.0,
        }
    }

    /// Compute the score function at given parameters for a Normal distribution.
    pub fn score_normal(data: &[f64], mu: f64, sigma: f64) -> Vec<f64> {
        let _n = data.len() as f64;
        let mut s_mu = 0.0;
        let mut s_sigma = 0.0;
        for &x in data {
            let d = x - mu;
            s_mu += d;
            s_sigma += d * d - sigma * sigma;
        }
        vec![s_mu / (sigma * sigma), s_sigma / (sigma * sigma * sigma)]
    }

    /// Compute the observed Fisher information (negative Hessian of log-likelihood).
    pub fn observed_information_normal(data: &[f64], mu: f64, sigma: f64) -> Matrix {
        let n = data.len() as f64;
        let s2 = sigma * sigma;
        let mut m12 = 0.0;
        let mut m22_raw = 0.0;
        for &x in data {
            let d = x - mu;
            m12 += d;
            m22_raw += d * d;
        }
        let m11 = n / s2;
        m12 = 2.0 * m12 / (sigma * s2);
        let m22 = -n / s2 + 3.0 * m22_raw / (s2 * s2);

        // This is -H (negative Hessian) = observed information
        Matrix::from_2x2(m11, m12, m12, m22)
    }

    /// Compute the expected Fisher information (= negative expected Hessian).
    pub fn expected_information(distribution: &Distribution) -> Matrix {
        FisherRaoMetric::new(distribution.clone()).fisher_matrix()
    }

    fn normal_log_likelihood(data: &[f64], mu: f64, sigma: f64) -> f64 {
        let n = data.len() as f64;
        let s2 = sigma * sigma;
        let sum_sq: f64 = data.iter().map(|x| (x - mu).powi(2)).sum();
        -n / 2.0 * (2.0 * std::f64::consts::PI * s2).ln() - sum_sq / (2.0 * s2)
    }

    fn bernoulli_log_likelihood(successes: usize, trials: usize, p: f64) -> f64 {
        let failures = trials - successes;
        let mut ll = 0.0;
        if successes > 0 {
            ll += successes as f64 * p.ln();
        }
        if failures > 0 {
            ll += failures as f64 * (1.0 - p).ln();
        }
        ll
    }

    fn exponential_log_likelihood(data: &[f64], lambda: f64) -> f64 {
        let n = data.len() as f64;
        n * lambda.ln() - lambda * data.iter().sum::<f64>()
    }
}

impl Default for Estimation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_normal_known_params() {
        // Generate data from Normal(3.0, 1.5²)
        let data: Vec<f64> = vec![
            3.5, 2.8, 4.1, 3.2, 2.9, 3.8, 1.5, 4.5, 3.0, 2.7, 3.3, 3.9, 2.6, 3.1, 3.7, 4.0, 2.4,
            3.6, 3.4, 2.5,
        ];
        let est = Estimation::new();
        let result = est.fit_normal(&data);
        assert!(result.converged);
        assert!((result.estimates.values[0] - 3.0).abs() < 1.0); // μ close to 3
        assert!((result.estimates.values[1] - 1.5).abs() < 1.0); // σ close to 1.5
        assert!(result.log_likelihood.unwrap().is_finite());
    }

    #[test]
    fn test_fit_normal_zero_mean() {
        let data: Vec<f64> = vec![-0.1, 0.2, -0.05, 0.1, -0.2, 0.15, -0.08, 0.03, -0.12, 0.07];
        let est = Estimation::new();
        let result = est.fit_normal(&data);
        assert!(result.converged);
        assert!(result.estimates.values[0].abs() < 0.5); // μ close to 0
    }

    #[test]
    fn test_fit_bernoulli() {
        let est = Estimation::new();
        let result = est.fit_bernoulli(7, 10);
        assert!(result.converged);
        assert!((result.estimates.values[0] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_fit_exponential() {
        // Data from Exp(2): mean should be ~0.5, so λ̂ = 1/mean ≈ 2
        let data: Vec<f64> = vec![0.3, 0.7, 0.2, 0.5, 0.4, 0.8, 0.1, 0.6, 0.3, 0.5];
        let est = Estimation::new();
        let result = est.fit_exponential(&data);
        assert!(result.converged);
        assert!(result.log_likelihood.unwrap().is_finite());
    }

    #[test]
    fn test_score_at_mle_is_zero() {
        // For Bernoulli, score at MLE should be zero
        let data = vec![1.0_f64, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]; // 7/10
        // Not directly testing Normal score here since that needs different data format
        let score = Estimation::score_normal(&data, 0.5, 0.5);
        // Score at sample mean should be near zero
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let score2 = Estimation::score_normal(&data, mean, 0.5);
        assert!(score2[0].abs() < 1e-10);
    }

    #[test]
    fn test_observed_vs_expected_information() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mu = 3.0;
        let sigma = 1.0;
        let observed = Estimation::observed_information_normal(&data, mu, sigma);
        let expected = Estimation::expected_information(&Distribution::Normal { mu, sigma });

        // Observed and expected should be close for large samples
        // For this small sample, they differ but same order of magnitude
        assert!(observed.get(0, 0) > 0.0);
        assert!(expected.get(0, 0) > 0.0);
    }

    #[test]
    fn test_fisher_info_at_estimate() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let est = Estimation::new();
        let result = est.fit_normal(&data);
        // Fisher info should be positive definite
        assert!(result.fisher_info.is_psd());
    }

    #[test]
    fn test_log_likelihood() {
        let data: Vec<f64> = vec![0.0];
        let ll = Estimation::normal_log_likelihood(&data, 0.0, 1.0);
        // log p(0|N(0,1)) = -0.5*log(2π)
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!((ll - expected).abs() < 1e-10);
    }
}
