#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
mod driver;
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
pub(crate) use driver::CudaFold;

#[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
mod unavailable {
    use crate::{MatrixError, matrix::Job, params::Params};

    pub(crate) struct CudaFold;

    impl CudaFold {
        pub(crate) fn new(_: &Params) -> Result<Self, MatrixError> {
            Err(MatrixError::Runtime(
                "CUDA is unavailable in this build".into(),
            ))
        }

        pub(crate) fn fold(&self, _: &[Job<'_>]) -> Result<Vec<f32>, MatrixError> {
            unreachable!("CUDA calculator construction failed")
        }
    }
}
#[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
pub(crate) use unavailable::CudaFold;
