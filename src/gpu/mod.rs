use crate::matrix::Job;
use crate::{Backend, MatrixError, cuda::CudaFold, params::Params};

pub(crate) const MAX_BASES: usize = 128;

pub(crate) enum GpuFold {
    Cuda(CudaFold),
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    Metal(crate::metal::MetalFold),
}

impl GpuFold {
    pub(crate) fn new(backend: Backend, params: &Params) -> Result<Option<Self>, MatrixError> {
        match backend {
            Backend::Cpu => Ok(None),
            Backend::Cuda => Ok(Some(Self::Cuda(CudaFold::new(params)?))),
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            Backend::Metal => Ok(Some(Self::Metal(crate::metal::MetalFold::new(params)?))),
            #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
            Backend::Metal => Err(MatrixError::Runtime(
                "Metal requires a macOS arm64 build with the metal feature".into(),
            )),
        }
    }

    pub(crate) fn batch_size(&self) -> usize {
        match self {
            Self::Cuda(_) => 2048,
            // Two triangular tables, at most ~32 MiB per submission.
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            Self::Metal(_) => 512,
        }
    }

    pub(crate) fn fold(&self, jobs: &[Job<'_>]) -> Result<Vec<f32>, MatrixError> {
        match self {
            Self::Cuda(cuda) => cuda.fold(jobs),
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            Self::Metal(metal) => metal.fold(jobs),
        }
    }
}

#[cfg(any(
    all(feature = "cuda", any(target_os = "linux", target_os = "windows")),
    all(feature = "metal", target_os = "macos", target_arch = "aarch64")
))]
pub(crate) fn pack_sequences(jobs: &[Job<'_>]) -> (usize, Vec<u8>) {
    let capacity = jobs
        .iter()
        .map(|j| j.row.len() + j.column.len())
        .max()
        .unwrap();
    debug_assert!(capacity <= MAX_BASES);
    let stride = capacity + 2;
    let mut sequences = vec![0; jobs.len() * stride];
    for (job, seq) in jobs.iter().zip(sequences.chunks_exact_mut(stride)) {
        let n = job.row.len() + job.column.len();
        for (base, target) in job.row.iter().chain(job.column).zip(&mut seq[1..=n]) {
            *target = match base {
                b'A' => 1,
                b'C' => 2,
                b'G' => 3,
                b'T' => 4,
                _ => unreachable!("validated nucleotide"),
            };
        }
        seq[0] = seq[n];
        seq[n + 1] = seq[1];
    }
    (capacity, sequences)
}
