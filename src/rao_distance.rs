//! Rao distance: geodesic distance on the Fisher-Rao manifold.
//!
//! The Rao distance measures the intrinsic distance between two probability
//! distributions using the geodesic curve on the Riemannian manifold equipped
//! with the Fisher information metric.

use serde::{Deserialize, Serialize};

use crate::types::Distribution;

/// Rao distance computation between distributions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaoDistance;

impl RaoDistance {
    /// Compute the Rao distance between two distributions of the same family.
    ///
    /// Uses closed-form expressions where available, numerical integration
    /// otherwise.
    pub fn distance(d1: &Distribution, d2: &Distribution) -> f64 {
        match (d1, d2) {
            (
                Distribution::Normal {
                    mu: mu1,
                    sigma: sigma1,
                },
                Distribution::Normal {
                    mu: mu2,
                    sigma: sigma2,
                },
            ) => Self::distance_normal(*mu1, *sigma1, *mu2, *sigma2),
            (Distribution::Bernoulli { p: p1 }, Distribution::Bernoulli { p: p2 }) => {
                Self::distance_bernoulli(*p1, *p2)
            }
            (
                Distribution::Exponential { lambda: l1 },
                Distribution::Exponential { lambda: l2 },
            ) => Self::distance_exponential(*l1, *l2),
            (
                Distribution::Multinomial { n: _, probs: p1 },
                Distribution::Multinomial { n: _, probs: p2 },
            ) => Self::distance_multinomial(p1, p2),
            _ => panic!("Rao distance requires same distribution family"),
        }
    }

    /// Rao distance for Normal(μ, σ²):
    ///
    /// d = √2 · |ln(σ₂/σ₁)|
    ///
    /// For the full Normal manifold (μ, σ), the distance is:
    /// d = 2·√2 · arccosh(1 + (μ₁-μ₂)² / (2σ₁σ₂) + (σ₁²+σ₂²)/(2σ₁σ₂) - 1)
    ///
    /// Simplified: d² = 2·ln(σ₂/σ₁)² + 2·(1 - (μ₁-μ₂)² / ((σ₁²+σ₂²)))·...
    ///
    /// The exact closed form for the full (μ,σ) Normal manifold:
    /// d = 2·arccos( (σ₁σ₂) / ((σ₁² + σ₂² + (μ₁-μ₂)²)/2)^(1/2) · √2 )
    ///
    /// Actually the standard formula:
    /// d(Normal(μ₁,σ₁²), Normal(μ₂,σ₂²)) = √2 · arccosh( (σ₁²+σ₂²+(μ₁-μ₂)²) / (2σ₁σ₂) )
    pub fn distance_normal(mu1: f64, sigma1: f64, mu2: f64, sigma2: f64) -> f64 {
        let s1sq = sigma1 * sigma1;
        let s2sq = sigma2 * sigma2;
        let dmu = mu1 - mu2;
        let arg = (s1sq + s2sq + dmu * dmu) / (2.0 * sigma1 * sigma2);
        std::f64::consts::SQRT_2 * arg.max(1.0).acosh()
    }

    /// Rao distance for Bernoulli: d(p₁, p₂) = 2·arccos(√(p₁p₂) + √((1-p₁)(1-p₂))).
    pub fn distance_bernoulli(p1: f64, p2: f64) -> f64 {
        let inner = (p1 * p2).sqrt() + ((1.0 - p1) * (1.0 - p2)).sqrt();
        2.0 * inner.acos()
    }

    /// Rao distance for Exponential: d(λ₁, λ₂) = |ln(λ₂/λ₁)| · √2.
    /// The exponential family with rate λ has Fisher info 1/λ², so the metric is ds = dλ/λ.
    /// Geodesic distance: √2 · |ln(λ₂/λ₁)|.
    pub fn distance_exponential(lambda1: f64, lambda2: f64) -> f64 {
        std::f64::consts::SQRT_2 * (lambda2 / lambda1).ln().abs()
    }

    /// Rao distance for Multinomial (Hellinger-based approximation).
    /// For the probability simplex with Fisher metric, the geodesic is:
    /// d = 2·arccos(Σ√(pᵢqᵢ))
    pub fn distance_multinomial(p: &[f64], q: &[f64]) -> f64 {
        assert_eq!(p.len(), q.len(), "Probabilities must have same length");
        let dot: f64 = p.iter().zip(q.iter()).map(|(a, b)| (a * b).sqrt()).sum();
        2.0 * dot.acos()
    }

    /// Compute Rao distance via numerical integration along a curve.
    ///
    /// This uses the general formula:
    /// d = ∫₀¹ √(θ̇(t)ᵀ · I(θ(t)) · θ̇(t)) dt
    ///
    /// Parameters:
    /// - `fisher_at`: function that returns Fisher matrix at parameter vector
    /// - `curve`: function that returns θ(t) for t ∈ [0, 1]
    /// - `n_steps`: number of integration steps
    pub fn distance_numerical<F, G>(fisher_at: F, curve: G, n_steps: usize) -> f64
    where
        F: Fn(&[f64]) -> crate::types::Matrix,
        G: Fn(f64) -> Vec<f64>,
    {
        let dt = 1.0 / (n_steps as f64);
        let mut total = 0.0;

        let mut prev_theta = curve(0.0);
        for i in 1..=n_steps {
            let t = (i as f64) * dt;
            let theta = curve(t);
            let dtheta: Vec<f64> = theta
                .iter()
                .zip(prev_theta.iter())
                .map(|(a, b)| (a - b) / dt)
                .collect();

            let fim = fisher_at(&theta);

            // Compute √(dθᵀ · I · dθ)
            let idt = fim.mul_vec(&dtheta);
            let norm: f64 = dtheta.iter().zip(idt.iter()).map(|(a, b)| a * b).sum();

            total += norm.sqrt() * dt;
            prev_theta = theta;
        }

        total
    }

    /// Check if the distance satisfies the triangle inequality.
    pub fn satisfies_triangle(d12: f64, d23: f64, d13: f64) -> bool {
        d13 <= d12 + d23 + 1e-10 && d12 <= d13 + d23 + 1e-10 && d23 <= d12 + d13 + 1e-10
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_same_distribution() {
        let d = RaoDistance::distance_normal(0.0, 1.0, 0.0, 1.0);
        assert!(d.abs() < 1e-10, "Distance to self should be 0, got {}", d);
    }

    #[test]
    fn test_normal_different_means() {
        let d = RaoDistance::distance_normal(0.0, 1.0, 3.0, 1.0);
        assert!(d > 0.0);
    }

    #[test]
    fn test_normal_different_sigmas() {
        let d = RaoDistance::distance_normal(0.0, 1.0, 0.0, 2.0);
        assert!(d > 0.0);
    }

    #[test]
    fn test_bernoulli_same() {
        let d = RaoDistance::distance_bernoulli(0.5, 0.5);
        assert!(d.abs() < 1e-10);
    }

    #[test]
    fn test_bernoulli_different() {
        let d = RaoDistance::distance_bernoulli(0.3, 0.7);
        assert!(d > 0.0);
    }

    #[test]
    fn test_exponential_same() {
        let d = RaoDistance::distance_exponential(2.0, 2.0);
        assert!(d.abs() < 1e-10);
    }

    #[test]
    fn test_exponential_different() {
        let d = RaoDistance::distance_exponential(1.0, 3.0);
        assert!(d > 0.0);
    }

    #[test]
    fn test_normal_symmetry() {
        let d1 = RaoDistance::distance_normal(1.0, 2.0, 3.0, 4.0);
        let d2 = RaoDistance::distance_normal(3.0, 4.0, 1.0, 2.0);
        assert!((d1 - d2).abs() < 1e-10);
    }

    #[test]
    fn test_bernoulli_symmetry() {
        let d1 = RaoDistance::distance_bernoulli(0.2, 0.8);
        let d2 = RaoDistance::distance_bernoulli(0.8, 0.2);
        assert!((d1 - d2).abs() < 1e-10);
    }

    #[test]
    fn test_triangle_inequality() {
        let d12 = RaoDistance::distance_normal(0.0, 1.0, 2.0, 1.5);
        let d23 = RaoDistance::distance_normal(2.0, 1.5, 4.0, 2.0);
        let d13 = RaoDistance::distance_normal(0.0, 1.0, 4.0, 2.0);
        assert!(RaoDistance::satisfies_triangle(d12, d23, d13));
    }

    #[test]
    fn test_multinomial_distance() {
        let p = vec![0.3, 0.4, 0.3];
        let q = vec![0.3, 0.4, 0.3];
        let d = RaoDistance::distance_multinomial(&p, &q);
        assert!(d.abs() < 1e-10);
    }

    #[test]
    fn test_distance_via_distribution_enum() {
        let d1 = Distribution::Normal {
            mu: 0.0,
            sigma: 1.0,
        };
        let d2 = Distribution::Normal {
            mu: 1.0,
            sigma: 1.0,
        };
        let d = RaoDistance::distance(&d1, &d2);
        assert!(d > 0.0);
    }

    #[test]
    fn test_numerical_integration_normal() {
        use crate::metric::FisherRaoMetric;
        use crate::types::Params;

        let d = RaoDistance::distance_numerical(
            |theta| {
                let params = Params::new(vec!["mu", "sigma"], theta.to_vec());
                let metric = FisherRaoMetric::new(Distribution::Normal {
                    mu: 0.0,
                    sigma: 1.0,
                });
                metric.fisher_matrix_at(&params)
            },
            |t| {
                // Linear interpolation from (0, 1) to (1, 2)
                vec![t, 1.0 + t]
            },
            1000,
        );
        // Should be positive and finite
        assert!(d > 0.0 && d.is_finite());
    }
}
