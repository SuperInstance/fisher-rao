# fisher-rao

**Fisher-Rao metric, Cramér-Rao bound, information matrix, and Rao distance for parametric statistical families.**

Fisher information is the answer to a deceptively simple question: *how much does the data tell us about the unknown parameter?* If you observe a random variable drawn from a distribution governed by θ, the Fisher information quantifies precisely how sensitive the probability of your observation is to small changes in θ. High sensitivity means your observation is highly informative — you can pin down θ tightly. Low sensitivity means your observation is nearly compatible with a wide range of θ values — you remain uncertain.

The Fisher-Rao manifold takes this idea further. It equips the space of probability distributions with a **Riemannian metric** — a notion of distance — rooted in information theory. Two distributions are "close" if they're hard to distinguish from data. Two distributions are "far" if even a modest sample reveals which one generated it. The Fisher-Rao metric is the **unique** metric (up to scaling) that is invariant under sufficient statistics and reparametrization — a deep result due to Čencov. Think of it as a ruler that measures how distinguishable two distributions *really* are.

## The Metaphor

Imagine you're a detective trying to identify a suspect from eyewitness accounts. Each witness gives you a probability distribution over suspect features (height, hair color, build). Some witnesses are very specific — their distributions are sharply peaked. Others are vague — broad, flat distributions that could match many people.

The Fisher-Rao framework gives you a **ruler** for this probability space:

- **Fisher information** is the ruler's *resolution* — how finely can it distinguish between two nearby distributions?
- **The Rao distance** is the *measurement* — how far apart are two distributions on this intrinsic manifold?
- **The Cramér-Rao bound** is the *speed limit* — no estimator can resolve parameters faster than the Fisher information allows.
- **Efficiency** tells you if your estimator is *running at the speed limit* or wasting information.

This crate implements all of these concepts for four parametric families: Normal, Bernoulli, Multinomial, and Exponential.

## Architecture

```
                    ┌─────────────────────────────┐
                    │        fisher-rao           │
                    │                             │
                    │  ┌───────────────────────┐  │
                    │  │       metric          │  │
                    │  │  FisherRaoMetric      │◄─┼── Entry point: compute gᵢⱼ(θ)
                    │  │  gᵢⱼ = E[∂ᵢℓ · ∂ⱼℓ] │  │
                    │  └───────────┬───────────┘  │
                    │              │               │
                    │              ▼               │
                    │  ┌───────────────────────┐  │
                    │  │     information       │  │
                    │  │  InformationMatrix    │  │  Analyze, invert, eigenvalues,
                    │  │  Jeffrey's prior      │  │  condition number
                    │  └───────┬──────┬────────┘  │
                    │          │      │            │
                    │          ▼      ▼            │
                    │  ┌──────────┐ ┌───────────┐ │
                    │  │  bound   │ │ rao_dist  │ │
                    │  │  CRB     │ │ geodesics │ │  CRB: Var(θ̂) ≥ I⁻¹
                    │  │  I(θ)⁻¹ │ │ d(p,q)    │ │  Rao: intrinsic distance
                    │  └────┬─────┘ └───────────┘ │
                    │       │                      │
                    │       ▼                      │
                    │  ┌───────────────────────┐  │
                    │  │     estimation        │  │  MLE via Fisher scoring:
                    │  │  Fisher scoring       │  │  θ ← θ + I⁻¹·S(θ)
                    │  │  Score function       │  │
                    │  └───────────┬───────────┘  │
                    │              │               │
                    │              ▼               │
                    │  ┌───────────────────────┐  │
                    │  │     efficiency        │  │  ARE, one-step,
                    │  │  Relative efficiency  │  │  Bhattacharyya bound
                    │  │  Bhattacharyya        │  │
                    │  └───────────────────────┘  │
                    └─────────────────────────────┘

    Supported distributions:
    ┌──────────┐  ┌───────────┐  ┌───────────┐  ┌─────────────┐
    │  Normal  │  │ Bernoulli │  │Multinomial│  │ Exponential │
    │ (μ, σ²)  │  │    (p)    │  │(n,[p₁..])│  │    (λ)      │
    └──────────┘  └───────────┘  └───────────┘  └─────────────┘
```

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
fisher-rao = "0.1"
```

### Compute the Fisher Information Matrix

```rust
use fisher_rao::{FisherRaoMetric, Distribution};

// Fisher information for Normal(μ=0, σ=2)
let dist = Distribution::Normal { mu: 0.0, sigma: 2.0 };
let metric = FisherRaoMetric::new(dist);
let fim = metric.fisher_matrix();

// I(μ,σ) = [[1/σ², 0], [0, 2/σ²]]
assert!((fim.get(0, 0) - 0.25).abs() < 1e-10); // 1/4
assert!((fim.get(1, 1) - 0.50).abs() < 1e-10); // 2/4

// Fisher information for Bernoulli(p=0.5)
let dist = Distribution::Bernoulli { p: 0.5 };
let metric = FisherRaoMetric::new(dist);
let fim = metric.fisher_matrix();

// I(p) = 1/(p(1-p)) = 4.0
assert!((fim.get(0, 0) - 4.0).abs() < 1e-10);
```

### Analyze the Information Matrix

```rust
use fisher_rao::{InformationMatrix, Distribution};

let im = InformationMatrix::compute(Distribution::Normal { mu: 0.0, sigma: 1.0 });

// Jeffrey's prior: √det(I(θ))
let jeffreys = im.jeffreys_prior();
// √(1·2) = √2
assert!((jeffreys - std::f64::consts::SQRT_2).abs() < 1e-10);

// Eigenvalues
let eigs = im.eigenvalues();
assert_eq!(eigs.len(), 2);
// I = [[1,0],[0,2]] → eigenvalues [2.0, 1.0]

// Condition number
let cond = im.condition_number();
assert!((cond - 2.0).abs() < 1e-10);

// Inverse
let inv = im.inverse().unwrap();
// I⁻¹ = [[1, 0], [0, 0.5]]
assert!((inv.get(1, 1) - 0.5).abs() < 1e-10);
```

### Compute Rao Distance

```rust
use fisher_rao::RaoDistance;

// Distance between two Normal distributions
let d = RaoDistance::distance_normal(0.0, 1.0, 3.0, 1.0);
// √2 · arccosh((1+1+9)/(2·1·1)) = √2 · arccosh(5.5)
println!("Rao distance: {:.4}", d);

// Distance between two Bernoulli distributions
let d = RaoDistance::distance_bernoulli(0.3, 0.7);
// 2 · arccos(√(0.21) + √(0.21)) ≈ 0.991
println!("Rao distance: {:.4}", d);

// Distance to self is always 0
let d = RaoDistance::distance_normal(0.0, 1.0, 0.0, 1.0);
assert!(d.abs() < 1e-10);
```

### Verify the Cramér-Rao Bound

```rust
use fisher_rao::{CramerRaoBound, Distribution, Matrix};

let crb = CramerRaoBound::new(Distribution::Bernoulli { p: 0.5 });

// The bound: Var(p̂) ≥ 1/I(p) = 1/4 = 0.25
let bound = crb.scalar_bound();
assert!((bound - 0.25).abs() < 1e-10);

// Check if an estimator achieves the CRB (is efficient)
let estimator_cov = Matrix::from_scalar(0.3); // variance = 0.3
let result = crb.is_efficient(&estimator_cov);
assert!(!result.is_efficient); // 0.3 > 0.25, not efficient

// For sample of size n, CRB shrinks by 1/n
let bound_n10 = crb.bound_for_sample(10).unwrap();
assert!((bound_n10.get(0, 0) - 0.025).abs() < 1e-10); // 0.25/10
```

### Maximum Likelihood via Fisher Scoring

```rust
use fisher_rao::Estimation;

// Fit a Normal distribution to data
let data = vec![3.5, 2.8, 4.1, 3.2, 2.9, 3.8, 1.5, 4.5, 3.0, 2.7,
                3.3, 3.9, 2.6, 3.1, 3.7, 4.0, 2.4, 3.6, 3.4, 2.5];
let est = Estimation::new();
let result = est.fit_normal(&data);

println!("Estimated μ: {:.4}", result.estimates.values[0]); // ~3.0
println!("Estimated σ: {:.4}", result.estimates.values[1]); // ~0.6
println!("Converged: {} in {} iterations", result.converged, result.iterations);
println!("Log-likelihood: {:.4}", result.log_likelihood.unwrap());

// Fit Bernoulli
let result = est.fit_bernoulli(7, 10); // 7 successes in 10 trials
println!("Estimated p: {:.4}", result.estimates.values[0]); // 0.7

// Fit Exponential
let data = vec![0.3, 0.7, 0.2, 0.5, 0.4, 0.8, 0.1, 0.6, 0.3, 0.5];
let result = est.fit_exponential(&data);
println!("Estimated λ: {:.4}", result.estimates.values[0]);
```

### Compare Estimator Efficiency

```rust
use fisher_rao::{Efficiency, BhattacharyyaSecondOrder};

// Asymptotic relative efficiency
let are = Efficiency::are(0.5, 1.0);
assert!((are - 2.0).abs() < 1e-10); // Estimator 1 is twice as efficient

// Classic result: median vs mean for Normal
// ARE = 2/π ≈ 0.637 — median wastes ~36% of information
let are = Efficiency::median_vs_mean_normal();
println!("ARE(median, mean) for Normal: {:.4}", are);
// Output: 0.6366

// Bhattacharyya bound (tighter than CRB)
let info = 4.0; // Fisher info for Bernoulli(0.5)
let second_order = BhattacharyyaSecondOrder::bernoulli(0.5);
let (crb, bhat) = Efficiency::bhattacharyya_bound(info, &second_order);
println!("CRB: {:.4}, Bhattacharyya: {:.4}", crb, bhat);
assert!(bhat >= crb - 1e-10); // Bhattacharyya ≥ CRB always
```

## Modules

| Module | Purpose | Key Types |
|--------|---------|-----------|
| `metric` | Compute the Fisher information matrix gᵢⱼ(θ) for parametric families | `FisherRaoMetric` |
| `information` | Analyze Fisher information: invert, eigenvalues, Jeffrey's prior | `InformationMatrix` |
| `rao_distance` | Geodesic distance on the Fisher-Rao manifold (closed-form + numerical) | `RaoDistance` |
| `bound` | Cramér-Rao lower bound Var(θ̂) ≥ I(θ)⁻¹, efficiency verification | `CramerRaoBound`, `EfficiencyResult` |
| `estimation` | MLE via Fisher scoring (Newton with I(θ) instead of Hessian) | `Estimation`, `EstimationResult` |
| `efficiency` | Asymptotic relative efficiency, one-step estimators, Bhattacharyya bound | `Efficiency`, `BhattacharyyaSecondOrder` |

## Mathematical Foundations

### Score Function

The score function is the gradient of the log-likelihood with respect to the parameter:

```
S(θ) = ∂ log p(x|θ) / ∂θ
```

Key properties:
- E[S(θ)] = 0 (the score has zero mean at the true parameter)
- The score measures how sensitive the likelihood is to parameter changes

### Fisher Information Matrix

The Fisher information matrix is the covariance of the score:

```
gᵢⱼ(θ) = E[∂ᵢ log p(x|θ) · ∂ⱼ log p(x|θ)]
```

Equivalently, it's the expected negative Hessian of the log-likelihood:

```
gᵢⱼ(θ) = -E[∂ᵢ∂ⱼ log p(x|θ)]
```

This matrix is always positive semi-definite and symmetric.

### Analytical Forms

**Normal(μ, σ²):**
```
I(μ,σ) = ⎡ 1/σ²    0   ⎤
          ⎣  0    2/σ²  ⎦
```

**Bernoulli(p):**
```
I(p) = 1 / (p(1-p))
```

**Exponential(λ):**
```
I(λ) = 1/λ²
```

**Multinomial(n, [p₁,…,pₖ]):**
```
Iᵢⱼ = n · (δᵢⱼ/pᵢ + 1/pₖ)    where pₖ = 1 - Σpᵢ
```

### Cramér-Rao Bound

For any unbiased estimator θ̂ of parameter θ:

```
Var(θ̂) ≥ I(θ)⁻¹
```

For a sample of n i.i.d. observations, the information scales linearly:

```
I_n(θ) = n · I₁(θ)
```

So the bound tightens: Var(θ̂) ≥ 1/(n·I₁(θ)).

An estimator that achieves this bound is called **efficient**. The sample mean is efficient for the Normal mean. The sample proportion is efficient for the Bernoulli parameter.

### Rao Distance

The Rao distance is the geodesic distance on the Riemannian manifold of distributions equipped with the Fisher metric:

```
d(p, q) = ∫₀¹ √(θ̇(t)ᵀ · I(θ(t)) · θ̇(t)) dt
```

where θ(t) is the geodesic curve connecting the parameters of p and q.

**Closed forms:**

For Normal(μ₁,σ₁²) and Normal(μ₂,σ₂²):
```
d = √2 · arccosh((σ₁² + σ₂² + (μ₁-μ₂)²) / (2σ₁σ₂))
```

For Bernoulli(p₁) and Bernoulli(p₂):
```
d = 2 · arccos(√(p₁p₂) + √((1-p₁)(1-p₂)))
```

For Exponential(λ₁) and Exponential(λ₂):
```
d = √2 · |ln(λ₂/λ₁)|
```

### Fisher Scoring

Fisher scoring is Newton's method with the expected Hessian replaced by the Fisher information:

```
θ_{k+1} = θ_k + I(θ_k)⁻¹ · S(θ_k)
```

This is guaranteed to increase the log-likelihood at each step (unlike Newton's method with the observed Hessian, which may diverge).

### Bhattacharyya Bound

The Bhattacharyya bound generalizes the CRB using second-order score information:

```
Var(T) ≥ G⁻¹_{11}
```

where G is the extended information matrix:

```
G = ⎡ I₁₁  I₁₂ ⎤
    ⎣ I₁₂  I₂₂ ⎦
```

with I₁₁ = I(θ), I₁₂ = E[∂ℓ·∂²ℓ], I₂₂ = E[(∂²ℓ)²].

The Bhattacharyya bound is always at least as tight as the CRB.

## Design Decisions

1. **Zero external dependencies except serde.** The matrix operations (multiply, invert, eigenvalues, determinant) are implemented from scratch. This keeps the crate lightweight and auditable.

2. **Analytical formulas preferred.** For the four supported distributions, all Fisher information matrices and Rao distances use closed-form expressions. Numerical integration is available as a fallback via Gauss-Hermite quadrature.

3. **All public types are serializable.** Every public struct derives `Serialize` and `Deserialize` from serde, making it easy to embed in larger systems.

4. **The `Matrix` type is minimal but sufficient.** It supports square matrices up to moderate sizes with row-major storage, Gauss-Jordan inversion, and 2×2 eigenvalue decomposition. This is enough for all parametric families in the crate.

5. **Fisher scoring over Newton-Raphson.** The `Estimation` module uses I(θ) instead of the observed Hessian because it's always positive definite (guaranteeing ascent) and often simpler to compute.

## Testing

The crate has **57 tests** covering:

- Fisher matrix computation for Normal, Bernoulli, Exponential, and Multinomial
- Positive semi-definiteness verification
- Cramér-Rao bound computation and efficiency checking
- Rao distance: identity (d=0), symmetry, triangle inequality
- Numerical integration for general geodesic distances
- MLE convergence via Fisher scoring
- Score function properties (zero at MLE)
- Observed vs. expected information
- Jeffrey's prior computation
- Eigenvalues and condition number
- ARE comparison and Bhattacharyya bounds

Run the test suite:

```bash
cargo test
```

## License

Dual-licensed under MIT OR Apache-2.0.
