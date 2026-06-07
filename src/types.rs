//! Matrix and vector types used across the crate.
//!
//! Minimal implementations with no external dependencies.

use serde::{Deserialize, Serialize};

/// A square matrix stored in row-major order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Matrix {
    /// Number of rows (and columns).
    pub n: usize,
    /// Row-major data, length n*n.
    pub data: Vec<f64>,
}

impl Matrix {
    /// Create a zero matrix of size `n × n`.
    pub fn zeros(n: usize) -> Self {
        Self {
            n,
            data: vec![0.0; n * n],
        }
    }

    /// Create an identity matrix of size `n × n`.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    /// Create from a slice of rows.
    pub fn from_rows(rows: &[&[f64]]) -> Self {
        let n = rows.len();
        let mut data = Vec::with_capacity(n * n);
        for row in rows {
            data.extend_from_slice(row);
        }
        Self { n, data }
    }

    /// Create a 1×1 matrix from a scalar.
    pub fn from_scalar(v: f64) -> Self {
        Self {
            n: 1,
            data: vec![v],
        }
    }

    /// Create a 2×2 matrix from elements.
    pub fn from_2x2(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self {
            n: 2,
            data: vec![a, b, c, d],
        }
    }

    /// Get element (i, j).
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n + j]
    }

    /// Set element (i, j).
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.n + j] = v;
    }

    /// Matrix addition.
    pub fn add(&self, other: &Matrix) -> Matrix {
        let mut result = self.clone();
        for i in 0..self.data.len() {
            result.data[i] += other.data[i];
        }
        result
    }

    /// Matrix subtraction.
    pub fn sub(&self, other: &Matrix) -> Matrix {
        let mut result = self.clone();
        for i in 0..self.data.len() {
            result.data[i] -= other.data[i];
        }
        result
    }

    /// Scalar multiplication.
    pub fn scale(&self, s: f64) -> Matrix {
        Matrix {
            n: self.n,
            data: self.data.iter().map(|&x| x * s).collect(),
        }
    }

    /// Matrix multiplication.
    pub fn mul(&self, other: &Matrix) -> Matrix {
        let n = self.n;
        let mut result = Matrix::zeros(n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    /// Matrix-vector multiplication.
    pub fn mul_vec(&self, v: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut result = vec![0.0; n];
        for (i, r) in result.iter_mut().enumerate() {
            for (j, vj) in v.iter().enumerate().take(n) {
                *r += self.get(i, j) * vj;
            }
        }
        result
    }

    /// Transpose (identity for symmetric matrices, but provided for completeness).
    pub fn transpose(&self) -> Matrix {
        let n = self.n;
        let mut result = Matrix::zeros(n);
        for i in 0..n {
            for j in 0..n {
                result.set(i, j, self.get(j, i));
            }
        }
        result
    }

    /// Trace (sum of diagonal elements).
    pub fn trace(&self) -> f64 {
        (0..self.n).map(|i| self.get(i, i)).sum()
    }

    /// Determinant via LU decomposition (partial pivot).
    pub fn det(&self) -> f64 {
        let n = self.n;
        if n == 1 {
            return self.data[0];
        }
        if n == 2 {
            return self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0);
        }

        // Gaussian elimination with partial pivoting
        let mut a = self.data.clone();
        let mut sign = 1.0;
        for col in 0..n {
            // Find pivot
            let mut max_val = a[col * n + col].abs();
            let mut max_row = col;
            for row in (col + 1)..n {
                let v = a[row * n + col].abs();
                if v > max_val {
                    max_val = v;
                    max_row = row;
                }
            }
            if max_row != col {
                // Swap rows
                for j in 0..n {
                    a.swap(col * n + j, max_row * n + j);
                }
                sign = -sign;
            }
            if a[col * n + col].abs() < 1e-15 {
                return 0.0;
            }
            for row in (col + 1)..n {
                let factor = a[row * n + col] / a[col * n + col];
                for j in (col + 1)..n {
                    a[row * n + j] -= factor * a[col * n + j];
                }
                a[row * n + col] = 0.0;
            }
        }
        let mut det_val = sign;
        for i in 0..n {
            det_val *= a[i * n + i];
        }
        det_val
    }

    /// Matrix inverse via Gauss-Jordan elimination.
    pub fn inverse(&self) -> Option<Matrix> {
        let n = self.n;
        let mut aug = vec![0.0; n * 2 * n];

        // Set up [A | I]
        for i in 0..n {
            for j in 0..n {
                aug[i * 2 * n + j] = self.get(i, j);
            }
            aug[i * 2 * n + n + i] = 1.0;
        }

        // Forward elimination with partial pivoting
        for col in 0..n {
            let mut max_val = aug[col * 2 * n + col].abs();
            let mut max_row = col;
            for row in (col + 1)..n {
                let v = aug[row * 2 * n + col].abs();
                if v > max_val {
                    max_val = v;
                    max_row = row;
                }
            }
            if max_row != col {
                for j in 0..(2 * n) {
                    aug.swap(col * 2 * n + j, max_row * 2 * n + j);
                }
            }
            let pivot = aug[col * 2 * n + col];
            if pivot.abs() < 1e-15 {
                return None;
            }

            // Scale pivot row
            for j in 0..(2 * n) {
                aug[col * 2 * n + j] /= pivot;
            }

            // Eliminate column
            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = aug[row * 2 * n + col];
                for j in 0..(2 * n) {
                    aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j];
                }
            }
        }

        let mut result = Matrix::zeros(n);
        for i in 0..n {
            for j in 0..n {
                result.set(i, j, aug[i * 2 * n + n + j]);
            }
        }
        Some(result)
    }

    /// Eigenvalues of a symmetric 2×2 matrix.
    /// Returns eigenvalues sorted in descending order.
    pub fn eigenvalues_2x2_symmetric(&self) -> Vec<f64> {
        assert!(self.n == 2, "Only for 2×2 matrices");
        let a = self.get(0, 0);
        let b = self.get(0, 1);
        let d = self.get(1, 1);
        let trace = a + d;
        let det_val = a * d - b * b;
        let disc = (trace * trace / 4.0 - det_val).max(0.0);
        let sqrt_disc = disc.sqrt();
        let mut eigs = vec![trace / 2.0 + sqrt_disc, trace / 2.0 - sqrt_disc];
        eigs.sort_by(|a, b| b.partial_cmp(a).unwrap());
        eigs
    }

    /// Check if the matrix is positive semi-definite.
    pub fn is_psd(&self) -> bool {
        // For small matrices, check via Cholesky-like test
        // Try Cholesky decomposition: if it succeeds, matrix is PD
        // For PSD, we also accept zero eigenvalues
        let n = self.n;
        if n == 1 {
            return self.data[0] >= -1e-10;
        }
        if n == 2 {
            let eigs = self.eigenvalues_2x2_symmetric();
            return eigs.iter().all(|&e| e >= -1e-10);
        }
        // General case: attempt Cholesky
        let mut l = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[i * n + k] * l[j * n + k];
                }
                if i == j {
                    let val = self.get(i, j) - sum;
                    if val < -1e-10 {
                        return false;
                    }
                    l[i * n + j] = val.max(0.0).sqrt();
                } else {
                    l[i * n + j] = (self.get(i, j) - sum) / l[j * n + j].max(1e-15);
                }
            }
        }
        true
    }

    /// Check if the matrix is symmetric.
    pub fn is_symmetric(&self, tol: f64) -> bool {
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if (self.get(i, j) - self.get(j, i)).abs() > tol {
                    return false;
                }
            }
        }
        true
    }
}

/// A parameter vector for a statistical model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Params {
    /// Named parameters (name → value).
    pub names: Vec<String>,
    pub values: Vec<f64>,
}

impl Params {
    /// Create from parallel lists.
    pub fn new(names: Vec<&str>, values: Vec<f64>) -> Self {
        Self {
            names: names.iter().map(|s| s.to_string()).collect(),
            values,
        }
    }

    /// Number of parameters.
    pub fn dim(&self) -> usize {
        self.values.len()
    }

    /// Create a 1D parameter.
    pub fn scalar(name: &str, value: f64) -> Self {
        Self::new(vec![name], vec![value])
    }
}

/// Parametric distribution families supported by this crate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Distribution {
    /// Normal(μ, σ²) — parameters are [μ, σ].
    Normal { mu: f64, sigma: f64 },
    /// Bernoulli(p) — parameter is [p].
    Bernoulli { p: f64 },
    /// Multinomial(n, [p₁, …, pₖ]) — n trials, k categories.
    Multinomial { n: usize, probs: Vec<f64> },
    /// Exponential(λ) — parameter is [λ].
    Exponential { lambda: f64 },
}

impl Distribution {
    /// Dimension of the parameter space.
    pub fn param_dim(&self) -> usize {
        match self {
            Distribution::Normal { .. } => 2,
            Distribution::Bernoulli { .. } => 1,
            Distribution::Multinomial { probs, .. } => probs.len() - 1, // last is determined
            Distribution::Exponential { .. } => 1,
        }
    }

    /// Get parameters as a Params struct.
    pub fn params(&self) -> Params {
        match self {
            Distribution::Normal { mu, sigma } => {
                Params::new(vec!["mu", "sigma"], vec![*mu, *sigma])
            }
            Distribution::Bernoulli { p } => Params::new(vec!["p"], vec![*p]),
            Distribution::Multinomial { probs, .. } => {
                let k = probs.len();
                let names: Vec<&str> = (0..k - 1)
                    .map(|i| Box::leak(format!("p{}", i + 1).into_boxed_str()) as &str)
                    .collect();
                Params::new(names, probs[..k - 1].to_vec())
            }
            Distribution::Exponential { lambda } => Params::new(vec!["lambda"], vec![*lambda]),
        }
    }
}
