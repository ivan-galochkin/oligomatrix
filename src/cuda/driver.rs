use std::sync::{Arc, Mutex};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg, sys,
};
use cudarc::nvrtc::Ptx;

use crate::{MatrixError, matrix::Job, params::Params};

impl From<DriverError> for MatrixError {
    fn from(error: DriverError) -> Self {
        let message = format!("CUDA: {error}");
        if error.0 == sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY {
            Self::Allocation(message)
        } else {
            Self::Runtime(message)
        }
    }
}

#[derive(Clone)]
struct Kernel {
    context: Arc<CudaContext>,
    function: CudaFunction,
}

impl Kernel {
    fn shared() -> Result<Self, MatrixError> {
        static KERNEL: Mutex<Option<Kernel>> = Mutex::new(None);
        let mut cached = KERNEL
            .lock()
            .map_err(|_| MatrixError::Runtime("CUDA initialization lock poisoned".into()))?;
        if let Some(kernel) = cached.as_ref() {
            return Ok(kernel.clone());
        }
        // SAFETY: probe the system driver before cudarc attempts to resolve its symbols.
        if !unsafe { sys::is_culib_present() } {
            return Err(MatrixError::Runtime("CUDA driver was not found".into()));
        }
        let mut version = 0;
        // SAFETY: the driver writes one integer to a valid host pointer.
        unsafe { sys::cuDriverGetVersion(&mut version) }.result()?;
        if version < 12090 {
            return Err(MatrixError::Runtime(
                "The bundled kernel requires an NVIDIA driver supporting CUDA 12.9 or later".into(),
            ));
        }
        let context = CudaContext::new(0)?;
        if context.compute_capability()? < (7, 5) {
            return Err(MatrixError::Runtime(
                "The bundled kernel requires NVIDIA compute capability 7.5 or later".into(),
            ));
        }
        let module = context.load_module(Ptx::from_src(include_str!("fold.ptx")))?;
        let function = module.load_function("fold")?;
        let kernel = Kernel { context, function };
        *cached = Some(kernel.clone());
        Ok(kernel)
    }
}

pub(crate) struct CudaFold {
    function: CudaFunction,
    stream: Arc<CudaStream>,
    tables: CudaSlice<i32>,
}

impl CudaFold {
    pub(crate) fn new(params: &Params) -> Result<Self, MatrixError> {
        let kernel = Kernel::shared()?;
        let stream = kernel.context.new_stream()?;
        let tables = stream.clone_htod(&params.cuda_tables())?;
        Ok(Self {
            function: kernel.function,
            stream,
            tables,
        })
    }

    pub(crate) fn fold(&self, jobs: &[Job<'_>]) -> Result<Vec<f32>, MatrixError> {
        let capacity = jobs
            .iter()
            .map(|job| job.row.len() + job.column.len())
            .max()
            .unwrap();
        debug_assert!(capacity <= super::MAX_BASES);
        let stride = capacity + 2;
        let mut seqs = vec![0u8; jobs.len() * stride];
        let mut cuts = Vec::with_capacity(jobs.len());
        let mut lengths = Vec::with_capacity(jobs.len());
        let mut monomers = Vec::with_capacity(jobs.len());
        for (job, seq) in jobs.iter().zip(seqs.chunks_exact_mut(stride)) {
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
            cuts.push(job.row.len() as i32);
            lengths.push(n as i32);
            monomers.push(job.column_energy);
        }
        let stream = &self.stream;
        let seqs = stream.clone_htod(&seqs)?;
        let cuts = stream.clone_htod(&cuts)?;
        let lengths = stream.clone_htod(&lengths)?;
        let monomers = stream.clone_htod(&monomers)?;
        let cells = capacity * (capacity + 1) / 2;
        // Long pairs place the branch table in global memory to fit a block's
        // shared-memory budget. The short-pair path uses neither scratch nor its contents.
        let (shared_cells, scratch_cells) = if capacity <= 96 {
            (2 * cells, 1)
        } else {
            (cells, jobs.len() * cells)
        };
        // SAFETY: the kernel writes each output and initializes every scratch cell it reads.
        let mut scratch = unsafe { stream.alloc::<i32>(scratch_cells) }?;
        let mut out = unsafe { stream.alloc::<f32>(jobs.len()) }?;
        let config = LaunchConfig {
            grid_dim: (jobs.len() as u32, 1, 1),
            block_dim: (capacity.next_multiple_of(32) as u32, 1, 1),
            shared_mem_bytes: (shared_cells * size_of::<i32>()) as u32,
        };
        let capacity = capacity as i32;
        let mut launch = stream.launch_builder(&self.function);
        launch
            .arg(&self.tables)
            .arg(&seqs)
            .arg(&cuts)
            .arg(&lengths)
            .arg(&monomers)
            .arg(&mut out)
            .arg(&mut scratch)
            .arg(&capacity);
        // SAFETY: one block owns one job, all arguments follow fold.cu's ABI,
        // and all device buffers live until this stream has finished.
        unsafe { launch.launch(config) }?;
        let result = stream.clone_dtoh(&out)?;
        stream.synchronize()?;
        Ok(result)
    }
}
