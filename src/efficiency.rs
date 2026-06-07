//! Efficiency: asymptotic relative efficiency, one-step estimators, and Bhattacharyya bounds.
//!
//! The asymptotic relative efficiency (ARE) compares two estimators by the
//! ratio of their asymptotic variances. The Bhattacharyya bound generalizes
//! the Cramér-Rao bound using higher-order derivatives of the score.

use serde::{Deserialize, Serialize};

use crate::types::Matrix;

/// Efficiency analysis for comparing estimators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Efficiency;

impl Efficiency {
    /// Compute asymptotic relative efficiency: ARE(T₁, T₂) = Var(T₂) / Var(T₁).
    ///
    /// ARE > 1 means T₁ is more efficient than T₂.
    /// ARE = 1 means they are equally efficient.
    /// ARE < 1 means T₂ is more efficient.
    pub fn are(var_estimator1: f64, var_estimator2: f64) -> f64 {
        var_estimator2 / var_estimator1
    }

    /// Compute ARE from sample variances (consistent estimators).
    pub fn are_from_samples(estimator1: &[f64], estimator2: &[f64]) -> f64 {
        let var1 = Self::sample_variance(estimator1);
        let var2 = Self::sample_variance(estimator2);
        Self::are(var1, var2)
    }

    /// Compute the one-step estimator improvement ratio.
    ///
    /// Starting from an initial estimator T₀ with variance v₀,
    /// one Fisher scoring step gives:
    /// T₁ = T₀ + I⁻¹(T₀) · S(T₀)
    ///
    /// The efficiency gain is the ratio v₀ / v₁.
    pub fn one_step_improvement(initial_variance: f64, fisher_info: f64) -> f64 {
        // After one step, variance approaches CRB = 1/I
        let crb = 1.0 / fisher_info;
        initial_variance / crb
    }

    /// Compute the Bhattacharyya bound (tighter than CRB).
    ///
    /// Uses second-order information:
    /// Var(T) ≥ B₁ + B₂
    ///
    /// where B₁ = I⁻¹ (CRB) and B₂ involves the second derivative matrix.
    ///
    /// For a scalar parameter θ:
    /// B₂ = (I₂₂ - I₁₁²) / (I · (I₂₂ - I₁₁²) - I₁₂²)
    ///
    /// Parameters:
    /// - `fisher_info`: I(θ) — first-order Fisher information
    /// - `second_order`: second-order Fisher information matrix elements
    ///
    /// Returns (crb, bhattacharyya_bound)
    pub fn bhattacharyya_bound(
        fisher_info: f64,
        second_order: &BhattacharyyaSecondOrder,
    ) -> (f64, f64) {
        let crb = 1.0 / fisher_info;

        // Second-order Bhattacharyya matrix
        // G = [[I₁₁, I₁₂], [I₁₂, I₂₂]]
        // where I₁₁ = I(θ), I₁₂ = E[∂log p/∂θ · ∂²log p/∂θ²] + I(θ)²
        // I₂₂ = E[(∂²log p/∂θ²)²]
        let g = Matrix::from_2x2(
            fisher_info,
            second_order.cross_term,
            second_order.cross_term,
            second_order.second_derivative_info,
        );

        let det_g = g.det();
        if det_g.abs() < 1e-15 {
            return (crb, crb);
        }

        // The Bhattacharyya bound is g^11 (the (1,1) element of G⁻¹)
        let bhattacharyya = g.get(1, 1) / det_g;

        (crb, bhattacharyya.abs())
    }

    /// Compute Bhattacharyya bound for Normal(μ, σ²) estimating σ².
    ///
    /// For σ² estimation: CRB = 2σ⁴/n, Bhattacharyya gives same (Normal is
    /// already efficient).
    pub fn bhattacharyya_normal_sigma(sigma: f64, n: usize) -> (f64, f64) {
        let s4 = sigma.powi(4);
        let crb = 2.0 * s4 / (n as f64);
        // For Normal, the sample variance achieves CRB, so Bhattacharyya = CRB
        (crb, crb)
    }

    /// Compute the efficiency of the sample median relative to the sample mean
    /// for Normal(μ, σ²).
    ///
    /// ARE(median, mean) = 2/π ≈ 0.637
    pub fn median_vs_mean_normal() -> f64 {
        2.0 / std::f64::consts::PI
    }

    /// Compute the efficiency of the sample mean relative to the best estimator
    /// for Laplace distribution.
    ///
    /// ARE(mean, median) for Laplace = 0.5
    pub fn mean_vs_median_laplace() -> f64 {
        0.5
    }

    /// Compute the Pitman efficiency (closeness probability).
    ///
    /// Given two estimators T₁ and T₂ of θ, Pitman efficiency is:
    /// P(|T₁ - θ| < |T₂ - θ|)
    pub fn pitman_efficiency(
        estimator1_samples: &[f64],
        estimator2_samples: &[f64],
        true_value: f64,
    ) -> f64 {
        let n = estimator1_samples.len().min(estimator2_samples.len());
        if n == 0 {
            return f64::NAN;
        }
        let mut count = 0;
        for i in 0..n {
            if (estimator1_samples[i] - true_value).abs()
                < (estimator2_samples[i] - true_value).abs()
            {
                count += 1;
            }
        }
        count as f64 / n as f64
    }

    fn sample_variance(data: &[f64]) -> f64 {
        let n = data.len() as f64;
        if n < 2.0 {
            return 0.0;
        }
        let mean = data.iter().sum::<f64>() / n;
        data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
    }
}

/// Second-order information for the Bhattacharyya bound.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BhattacharyyaSecondOrder {
    /// E[∂log p/∂θ · ∂²log p/∂θ²] — cross term between first and second derivatives.
    pub cross_term: f64,
    /// E[(∂²log p/∂θ²)²] — second derivative information.
    pub second_derivative_info: f64,
}

impl BhattacharyyaSecondOrder {
    /// Compute second-order information for Normal(μ, σ²) estimating σ.
    pub fn normal_sigma(sigma: f64) -> Self {
        // ∂²log p/∂σ² = ((x-μ)²/σ⁴) - 3/σ²
        // E[∂²log p/∂σ²] = -2/σ² (expected)
        // E[(∂²log p/∂σ²)²] = ... (computed analytically)
        let s2 = sigma * sigma;
        Self {
            cross_term: 0.0, // symmetric family, cross terms vanish
            second_derivative_info: 6.0 / (s2 * s2),
        }
    }

    /// Compute second-order information for Bernoulli(p).
    pub fn bernoulli(p: f64) -> Self {
        let q = 1.0 - p;
        let _info = 1.0 / (p * q);
        // For Bernoulli: ∂²log p/∂p² = -1/p² - 1/q²
        // E[(∂²log p/∂p²)²] = 1/p³ + 1/q³ - 2/(pq)
        let second_deriv_info = 1.0 / (p * p * p) + 1.0 / (q * q * q) - 2.0 / (p * q);
        // Cross term: E[∂log p · ∂²log p] involves E[(x-p)(-1/p² + 1/q²)/(p·q)]
        let cross = 0.0; // By Stein's lemma for exponential families
        Self {
            cross_term: cross,
            second_derivative_info: second_deriv_info,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_are_identical() {
        let are = Efficiency::are(1.0, 1.0);
        assert!((are - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_are_better() {
        let are = Efficiency::are(0.5, 1.0);
        assert!((are - 2.0).abs() < 1e-10); // T1 is twice as efficient
    }

    #[test]
    fn test_are_from_samples() {
        let est1 = vec![1.0, 1.1, 0.9, 1.0, 1.05];
        let est2 = vec![0.5, 1.5, 0.8, 1.2, 1.0];
        let are = Efficiency::are_from_samples(&est1, &est2);
        assert!(are > 1.0); // est1 has lower variance
    }

    #[test]
    fn test_one_step_improvement() {
        // For Normal(0, 1): I = 1, CRB = 1
        // If initial variance = 2, improvement = 2/1 = 2
        let improvement = Efficiency::one_step_improvement(2.0, 1.0);
        assert!((improvement - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_median_vs_mean() {
        let are = Efficiency::median_vs_mean_normal();
        assert!((are - 2.0 / std::f64::consts::PI).abs() < 1e-10);
        assert!(are < 1.0); // Median is less efficient than mean for Normal
    }

    #[test]
    fn test_mean_vs_median_laplace() {
        let are = Efficiency::mean_vs_median_laplace();
        assert!((are - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_bhattacharyya_normal() {
        let (crb, bhat) = Efficiency::bhattacharyya_normal_sigma(1.0, 10);
        assert!((crb - 0.2).abs() < 1e-10); // 2/(10) = 0.2
        assert!((bhat - crb).abs() < 1e-10); // Same for Normal
    }

    #[test]
    fn test_bhattacharyya_bernoulli() {
        let info = 1.0 / (0.5 * 0.5); // = 4
        let second_order = BhattacharyyaSecondOrder::bernoulli(0.5);
        let (crb, bhat) = Efficiency::bhattacharyya_bound(info, &second_order);
        assert!(bhat >= crb - 1e-10); // Bhattacharyya ≥ CRB
    }

    #[test]
    fn test_pitman_efficiency() {
        let est1 = vec![1.0, 1.1, 0.9, 1.0, 1.05];
        let est2 = vec![0.5, 1.5, 0.8, 1.2, 1.0];
        let pe = Efficiency::pitman_efficiency(&est1, &est2, 1.0);
        assert!(pe >= 0.0 && pe <= 1.0);
    }

    #[test]
    fn test_bhattacharyya_bound_singularity() {
        // When G is singular, should fall back to CRB
        let info = 1.0;
        let second_order = BhattacharyyaSecondOrder {
            cross_term: 0.0,
            second_derivative_info: 0.0, // Makes G singular
        };
        let (crb, bhat) = Efficiency::bhattacharyya_bound(info, &second_order);
        assert!((bhat - crb).abs() < 1e-10);
    }

    #[test]
    fn test_second_order_normal() {
        let so = BhattacharyyaSecondOrder::normal_sigma(2.0);
        assert!((so.cross_term).abs() < 1e-10);
        assert!(so.second_derivative_info > 0.0);
    }

    #[test]
    fn test_second_order_bernoulli() {
        let so = BhattacharyyaSecondOrder::bernoulli(0.5);
        assert!(so.second_derivative_info > 0.0);
    }
}
