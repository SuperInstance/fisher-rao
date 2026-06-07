//! # fisher-rao
//!
//! Fisher-Rao metric, Cramér-Rao bound, information matrix, and Rao distance
//! for parametric statistical families.

mod types;

pub mod bound;
pub mod efficiency;
pub mod estimation;
pub mod information;
pub mod metric;
pub mod rao_distance;

pub use bound::CramerRaoBound;
pub use efficiency::Efficiency;
pub use estimation::Estimation;
pub use information::InformationMatrix;
pub use metric::FisherRaoMetric;
pub use rao_distance::RaoDistance;
pub use types::{Distribution, Matrix, Params};
