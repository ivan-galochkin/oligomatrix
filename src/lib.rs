//! Pairwise minimum-free-energy matrices. The Python adapter is optional.
mod cuda;
mod error;
mod folding;
mod gpu;
mod matrix;
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod metal;
mod params;
#[cfg(feature = "python")]
mod python;
mod workers;

pub use error::MatrixError;
pub use matrix::{Backend, EnergyMatrix, MatrixCalculator};
pub use params::Model;
