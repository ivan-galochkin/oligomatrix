use std::sync::{Arc, Mutex};

use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::MatrixError;

// Reuse the most recent pool across one-shot calls. Live calculators keep their own Arc.
static LAST_POOL: Mutex<Option<Arc<ThreadPool>>> = Mutex::new(None);

pub(crate) struct Workers {
    limit: usize,
    pool: Option<Arc<ThreadPool>>,
}

impl Workers {
    pub(crate) fn new(threads: usize) -> Self {
        let cpus = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        let limit = cpus.min(if threads == 0 { 8 } else { threads });
        Self { limit, pool: None }
    }

    /// Small batches run on the caller; other batches reuse a fixed-size pool.
    pub(crate) fn pool_for(&mut self, jobs: usize) -> Result<Option<&ThreadPool>, MatrixError> {
        if jobs < 2 || self.limit == 1 {
            return Ok(None);
        }
        if self.pool.is_none() {
            let mut shared = LAST_POOL
                .lock()
                .map_err(|_| MatrixError::Runtime("Worker pool lock poisoned".into()))?;
            let pool = match shared.as_ref() {
                Some(pool) if pool.current_num_threads() == self.limit => pool.clone(),
                _ => {
                    let pool = ThreadPoolBuilder::new()
                        .num_threads(self.limit)
                        .build()
                        .map_err(|e| MatrixError::Runtime(e.to_string()))?;
                    let pool = Arc::new(pool);
                    *shared = Some(pool.clone());
                    pool
                }
            };
            self.pool = Some(pool);
        }
        Ok(self.pool.as_deref())
    }
}
