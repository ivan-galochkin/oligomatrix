use std::{mem::size_of_val, ptr::NonNull, sync::Mutex};

use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_foundation::NSString;
use objc2_metal::*;

use crate::{MatrixError, gpu::pack_sequences, matrix::Job, params::Params};

const SOURCE: &str = concat!(
    include_str!("../gpu/energy.h"),
    "\n",
    include_str!("../gpu/fold.h"),
    "\n",
    include_str!("fold.metal")
);

type Object<T> = Retained<ProtocolObject<T>>;
type Buffer = Object<dyn MTLBuffer>;

// Uploaded before publication and bound only as a const input to the kernel.
// No CPU or GPU writer can access this allocation after construction.
struct EnergyTables(Buffer);
// SAFETY: Metal resource ownership is thread-safe, and this wrapper never
// exposes the buffer's CPU pointer or binds it for writes.
unsafe impl Send for EnergyTables {}
unsafe impl Sync for EnergyTables {}

#[derive(Clone)]
struct Kernel {
    device: Object<dyn MTLDevice>,
    pipeline: Object<dyn MTLComputePipelineState>,
}

impl Kernel {
    fn shared() -> Result<Self, MatrixError> {
        static CACHE: Mutex<Option<Kernel>> = Mutex::new(None);
        let mut cached = CACHE
            .lock()
            .map_err(|_| MatrixError::Runtime("Metal pipeline lock poisoned".into()))?;
        if let Some(kernel) = &*cached {
            return Ok(kernel.clone());
        }
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| MatrixError::Runtime("No Metal device is available".into()))?;
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), None)
            .map_err(|e| MatrixError::Runtime(format!("Metal shader compilation failed: {e}")))?;
        let function = library
            .newFunctionWithName(&NSString::from_str("fold"))
            .ok_or_else(|| MatrixError::Runtime("Metal fold entry point is missing".into()))?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| MatrixError::Runtime(format!("Metal pipeline creation failed: {e}")))?;
        let kernel = Self { device, pipeline };
        *cached = Some(kernel.clone());
        Ok(kernel)
    }

    fn buffer(&self, size: usize, mode: MTLResourceOptions) -> Result<Buffer, MatrixError> {
        self.device
            .newBufferWithLength_options(size, mode)
            .ok_or_else(|| {
                MatrixError::Allocation(format!("Metal could not allocate {size} bytes"))
            })
    }

    fn upload<T: Copy>(&self, data: &[T]) -> Result<Buffer, MatrixError> {
        let buffer = self.buffer(size_of_val(data), MTLResourceOptions::StorageModeShared)?;
        // SAFETY: the fresh, CPU-visible buffer has exactly the size of the source slice.
        unsafe {
            buffer
                .contents()
                .cast::<T>()
                .as_ptr()
                .copy_from_nonoverlapping(data.as_ptr(), data.len())
        };
        Ok(buffer)
    }
}

pub(crate) struct MetalFold {
    kernel: Kernel,
    queue: Object<dyn MTLCommandQueue>,
    tables: EnergyTables,
}

impl MetalFold {
    pub(crate) fn new(params: &Params) -> Result<Self, MatrixError> {
        autoreleasepool(|_| {
            let kernel = Kernel::shared()?;
            let queue = kernel.device.newCommandQueue().ok_or_else(|| {
                MatrixError::Runtime("Metal command queue creation failed".into())
            })?;
            let tables = EnergyTables(kernel.upload(&params.gpu_tables())?);
            Ok(Self {
                kernel,
                queue,
                tables,
            })
        })
    }

    pub(crate) fn fold(&self, jobs: &[Job<'_>]) -> Result<Vec<f32>, MatrixError> {
        autoreleasepool(|_| self.dispatch(jobs))
    }

    fn dispatch(&self, jobs: &[Job<'_>]) -> Result<Vec<f32>, MatrixError> {
        let (capacity, sequences) = pack_sequences(jobs);
        let sizes: Vec<_> = jobs
            .iter()
            .map(|j| [j.row.len() as i32, (j.row.len() + j.column.len()) as i32])
            .collect();
        let seqs = self.kernel.upload(&sequences)?;
        let sizes = self.kernel.upload(&sizes)?;
        let result = self.kernel.buffer(
            jobs.len() * size_of::<[i32; 2]>(),
            MTLResourceOptions::StorageModeShared,
        )?;
        let scratch = self.kernel.buffer(
            jobs.len() * capacity * (capacity + 1) * size_of::<i32>(),
            MTLResourceOptions::StorageModePrivate,
        )?;
        let cmd = self
            .queue
            .commandBuffer()
            .ok_or_else(|| MatrixError::Runtime("Metal command buffer creation failed".into()))?;
        let encoder = cmd
            .computeCommandEncoder()
            .ok_or_else(|| MatrixError::Runtime("Metal compute encoder creation failed".into()))?;
        encoder.setComputePipelineState(&self.kernel.pipeline);
        let stride_capacity = capacity as i32;
        // SAFETY: all bindings match fold.metal, remain alive through completion,
        // and setBytes copies the scalar before returning.
        unsafe {
            for (i, buffer) in [&self.tables.0, &seqs, &sizes, &result, &scratch]
                .into_iter()
                .enumerate()
            {
                encoder.setBuffer_offset_atIndex(Some(buffer), 0, i);
            }
            encoder.setBytes_length_atIndex(
                NonNull::from(&stride_capacity).cast(),
                size_of::<i32>(),
                5,
            );
        }
        let width = capacity
            .next_multiple_of(self.kernel.pipeline.threadExecutionWidth())
            .min(self.kernel.pipeline.maxTotalThreadsPerThreadgroup());
        let line = |width| MTLSize {
            width,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(line(jobs.len()), line(width));
        encoder.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if let Some(error) = cmd.error() {
            return Err(MatrixError::Runtime(format!("Metal fold failed: {error}")));
        }
        // SAFETY: each group wrote one int2 and the GPU completed its writes.
        let pairs = unsafe {
            std::slice::from_raw_parts(result.contents().cast::<[i32; 2]>().as_ptr(), jobs.len())
        };
        Ok(pairs
            .iter()
            .zip(jobs)
            .map(|(&[connected, first], job)| {
                (connected as f32 / 100.0).min(first as f32 / 100.0 + job.column_energy)
            })
            .collect())
    }
}
