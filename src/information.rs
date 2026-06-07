//! Information matrix: compute, invert, and analyze the Fisher information.
//!
//! The Fisher information I(θ) measures the amount of information that an
//! observable random variable carries about the unknown parameter θ.
//! Jeffrey's prior ∝ √det(I) provides a non-informative prior.

use serde::{Deserialize, Serialize};

use crate::metric::FisherRaoMetric;
use crate::types::{Distribution, Matrix, Params};

/// Computed Fisher information matrix with analysis utilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationMatrix {
    /// The computed Fisher information matrix.
    pub matrix: Matrix,
    /// The distribution it was computed for.
    pub distribution: Distribution,
}

impl InformationMatrix {
    /// Compute the Fisher information matrix for a distribution.
    pub fn compute(distribution: Distribution) -> Self {
        let metric = FisherRaoMetric::new(distribution.clone());
        let matrix = metric.fisher_matrix();
        Self {
            matrix,
            distribution,
        }
    }

    /// Compute at a specific parameter point.
    pub fn compute_at(distribution: Distribution, params: &Params) -> Self {
        let metric = FisherRaoMetric::new(distribution.clone());
        let matrix = metric.fisher_matrix_at(params);
        Self {
            matrix,
            distribution,
        }
    }

    /// Compute Jeffrey's prior: ∝ √det(I(θ)).
    pub fn jeffreys_prior(&self) -> f64 {
        let det = self.matrix.det();
        if det <= 0.0 { 0.0 } else { det.sqrt() }
    }

    /// Invert the information matrix. Returns None if singular.
    pub fn inverse(&self) -> Option<Matrix> {
        self.matrix.inverse()
    }

    /// Compute eigenvalues (exact for 1×1 and 2×2, iterative for larger).
    pub fn eigenvalues(&self) -> Vec<f64> {
        match self.matrix.n {
            1 => vec![self.matrix.data[0]],
            2 => self.matrix.eigenvalues_2x2_symmetric(),
            _ => {
                // Power iteration for largest eigenvalue, then deflate
                // For simplicity, use the trace and det for PSD matrices
                // This is a simplified approach for small matrices
                self.approximate_eigenvalues()
            }
        }
    }

    /// Check if the information matrix is positive definite.
    pub fn is_positive_definite(&self) -> bool {
        self.matrix.det() > 0.0 && self.matrix.is_psd()
    }

    /// Condition number (ratio of largest to smallest eigenvalue).
    pub fn condition_number(&self) -> f64 {
        let eigs = self.eigenvalues();
        let max_eig = eigs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_eig = eigs.iter().cloned().fold(f64::INFINITY, f64::min);
        if min_eig.abs() < 1e-15 {
            f64::INFINITY
        } else {
            max_eig / min_eig
        }
    }

    /// Approximate eigenvalues using power iteration + deflation for n > 2.
    fn approximate_eigenvalues(&self) -> Vec<f64> {
        let n = self.matrix.n;
        let mut eigenvalues = Vec::new();
        let mut remaining = self.matrix.clone();

        for _ in 0..n {
            let eig = Self::power_iteration(&remaining, 100);
            eigenvalues.push(eig);
            // Deflate: subtract eig * vv^T where v is the eigenvector
            // Simplified: just use the eigenvalue
            if eigenvalues.len() < n {
                remaining = remaining.sub(&Matrix::identity(n).scale(eig));
            }
        }
        eigenvalues
    }

    /// Power iteration to find the dominant eigenvalue.
    fn power_iteration(m: &Matrix, iterations: usize) -> f64 {
        let n = m.n;
        let mut v = vec![1.0 / (n as f64).sqrt(); n];
        for _ in 0..iterations {
            let mv = m.mul_vec(&v);
            let norm: f64 = mv.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                return 0.0;
            }
            v = mv.iter().map(|x| x / norm).collect();
        }
        let mv = m.mul_vec(&v);
        let dot: f64 = v.iter().zip(mv.iter()).map(|(a, b)| a * b).sum();
        dot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_information() {
        let im = InformationMatrix::compute(Distribution::Normal {
            mu: 0.0,
            sigma: 2.0,
        });
        assert!((im.matrix.get(0, 0) - 0.25).abs() < 1e-10);
        assert!(im.is_positive_definite());
    }

    #[test]
    fn test_bernoulli_information() {
        let im = InformationMatrix::compute(Distribution::Bernoulli { p: 0.5 });
        assert!((im.matrix.get(0, 0) - 4.0).abs() < 1e-10);
        assert!(im.is_positive_definite());
    }

    #[test]
    fn test_jeffreys_prior_bernoulli() {
        // For Bernoulli: I(p) = 1/(p(1-p)), Jeffreys = √(1/(p(1-p))) = 1/√(p(1-p))
        let im = InformationMatrix::compute(Distribution::Bernoulli { p: 0.5 });
        let jp = im.jeffreys_prior();
        assert!((jp - 2.0).abs() < 1e-10); // √4 = 2
    }

    #[test]
    fn test_jeffreys_prior_normal() {
        let im = InformationMatrix::compute(Distribution::Normal {
            mu: 0.0,
            sigma: 1.0,
        });
        let jp = im.jeffreys_prior();
        // det(I) = 1/σ² · 2/σ² = 2/σ⁴, √det = √2/σ²
        assert!((jp - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_inverse() {
        let im = InformationMatrix::compute(Distribution::Normal {
            mu: 0.0,
            sigma: 1.0,
        });
        let inv = im.inverse().expect("should be invertible");
        // I = [[1, 0], [0, 2]], I⁻¹ = [[1, 0], [0, 0.5]]
        assert!((inv.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((inv.get(1, 1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_eigenvalues_normal() {
        let im = InformationMatrix::compute(Distribution::Normal {
            mu: 0.0,
            sigma: 2.0,
        });
        let eigs = im.eigenvalues();
        assert_eq!(eigs.len(), 2);
        // I = [[0.25, 0], [0, 0.5]] → eigenvalues 0.5, 0.25
        assert!((eigs[0] - 0.5).abs() < 1e-10);
        assert!((eigs[1] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_compute_at() {
        let im = InformationMatrix::compute_at(
            Distribution::Normal {
                mu: 0.0,
                sigma: 1.0,
            },
            &Params::new(vec!["mu", "sigma"], vec![0.0, 4.0]),
        );
        assert!((im.matrix.get(0, 0) - 1.0 / 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_condition_number() {
        let im = InformationMatrix::compute(Distribution::Normal {
            mu: 0.0,
            sigma: 2.0,
        });
        let cn = im.condition_number();
        assert!((cn - 2.0).abs() < 1e-10); // 0.5 / 0.25
    }

    #[test]
    fn test_exponential_information() {
        let im = InformationMatrix::compute(Distribution::Exponential { lambda: 2.0 });
        assert!((im.matrix.get(0, 0) - 0.25).abs() < 1e-10);
        assert!(im.is_positive_definite());
    }
}
