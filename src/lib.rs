//! Pairwise minimum-free-energy matrices. The Python adapter is optional.
mod cuda;
mod error;
mod folding;
mod matrix;
mod params;
#[cfg(feature = "python")]
mod python;
mod workers;

pub use error::MatrixError;
pub use matrix::{Backend, EnergyMatrix, MatrixCalculator};
pub use params::Model;
