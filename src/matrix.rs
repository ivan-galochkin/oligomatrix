use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use rayon::prelude::*;

use crate::params::Params;
use crate::workers::Workers;
use crate::{MatrixError, Model, folding};

const BATCH_SIZE: usize = 512;
const MAX_PAIR_LENGTH: usize = 32_700;
type Pair = (usize, usize);

/// An owned, row-major matrix in kcal/mol.
pub struct EnergyMatrix {
    shape: (usize, usize),
    values: Vec<f32>,
}

impl EnergyMatrix {
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    pub fn into_values(self) -> Vec<f32> {
        self.values
    }
}

struct Axis<'a> {
    ids: Vec<usize>,
    unique: Vec<(usize, &'a [u8])>,
}

impl<'a> Axis<'a> {
    fn index(sequences: &'a [String], known: &mut HashMap<String, usize>) -> Self {
        let mut ids = Vec::with_capacity(sequences.len());
        let mut unique = Vec::new();
        let mut seen = HashSet::new();
        for seq in sequences {
            let id = match known.get(seq) {
                Some(&id) => id,
                None => {
                    let id = known.len();
                    known.insert(seq.clone(), id);
                    id
                }
            };
            ids.push(id);
            if seen.insert(id) {
                unique.push((id, seq.as_bytes()));
            }
        }
        Self { ids, unique }
    }
}

struct Job<'a> {
    pair: Pair,
    row: &'a [u8],
    column: &'a [u8],
    column_energy: f32,
}

/// Caches ordered pairs for one temperature and one parameter model.
pub struct MatrixCalculator {
    params: Arc<Params>,
    workers: Workers,
    ids: HashMap<String, usize>,
    monomers: HashMap<usize, f32>,
    pairs: HashMap<Pair, f32>,
}

impl MatrixCalculator {
    /// A thread limit of zero selects up to eight available logical CPUs.
    pub fn new(temperature: f64, model: Model, threads: usize) -> Result<Self, MatrixError> {
        Ok(Self {
            params: Params::shared(model, temperature)?,
            workers: Workers::new(threads),
            ids: HashMap::new(),
            monomers: HashMap::new(),
            pairs: HashMap::new(),
        })
    }

    pub fn matrix(
        &mut self,
        rows: &[String],
        columns: &[String],
    ) -> Result<EnergyMatrix, MatrixError> {
        let shape = (rows.len(), columns.len());
        let longest_row = validate_axis(rows)?;
        let longest_column = validate_axis(columns)?;
        let cells = rows
            .len()
            .checked_mul(columns.len())
            .ok_or_else(|| MatrixError::Allocation("Matrix too large".into()))?;
        if cells == 0 {
            return Ok(EnergyMatrix {
                shape,
                values: Vec::new(),
            });
        }
        if longest_row + longest_column > MAX_PAIR_LENGTH {
            return Err(MatrixError::InvalidInput(
                "Combined fragment length exceeds 32700",
            ));
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(cells)
            .map_err(|e| MatrixError::Allocation(e.to_string()))?;
        let rows = Axis::index(rows, &mut self.ids);
        let columns = Axis::index(columns, &mut self.ids);
        self.compute_missing(&rows, &columns)?;
        for i in rows.ids {
            values.extend(columns.ids.iter().map(|&j| self.pairs[&(i, j)]));
        }
        Ok(EnergyMatrix { shape, values })
    }

    fn compute_missing(&mut self, rows: &Axis<'_>, columns: &Axis<'_>) -> Result<(), MatrixError> {
        let columns: Vec<_> = columns
            .unique
            .iter()
            .map(|&(id, seq)| {
                let energy = *self
                    .monomers
                    .entry(id)
                    .or_insert_with(|| folding::monomer(seq, &self.params));
                (id, seq, energy)
            })
            .collect();
        let mut jobs = Vec::new();
        for &(i, row) in &rows.unique {
            for &(j, column, column_energy) in &columns {
                if !self.pairs.contains_key(&(i, j)) {
                    jobs.push(Job {
                        pair: (i, j),
                        row,
                        column,
                        column_energy,
                    });
                    if jobs.len() == BATCH_SIZE {
                        self.fold_batch(&jobs)?;
                        jobs.clear();
                    }
                }
            }
        }
        self.fold_batch(&jobs)
    }

    fn fold_batch(&mut self, jobs: &[Job<'_>]) -> Result<(), MatrixError> {
        let params = &self.params;
        let fold = |job: &Job<'_>| {
            (
                job.pair,
                folding::dimer(job.row, job.column, job.column_energy, params),
            )
        };
        if let Some(pool) = self.workers.pool_for(jobs.len())? {
            let results: Vec<_> = pool.install(|| jobs.par_iter().map(fold).collect());
            self.pairs.extend(results);
        } else {
            self.pairs.extend(jobs.iter().map(fold));
        }
        Ok(())
    }
}

fn validate_axis(sequences: &[String]) -> Result<usize, MatrixError> {
    let mut longest = 0;
    for sequence in sequences {
        if sequence.is_empty()
            || !sequence
                .bytes()
                .all(|base| matches!(base, b'A' | b'C' | b'G' | b'T'))
        {
            return Err(MatrixError::InvalidInput(
                "Each fragment must be a non-empty uppercase A/C/G/T sequence",
            ));
        }
        longest = longest.max(sequence.len());
    }
    Ok(longest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_matrix_and_cached_pairs_are_available_without_python() {
        let mut calc = MatrixCalculator::new(37.0, Model::Turner2004, 2).unwrap();
        let rows = ["GAT".into(), "TACA".into(), "GAT".into()];
        let cols = ["ATCG".into(), "AGC".into()];
        let first = calc.matrix(&rows, &cols).unwrap();
        assert_eq!(first.shape(), (3, 2));
        assert_eq!(first.as_slice(), &[-0.6, 0.0, 0.0, 0.0, -0.6, 0.0]);
        assert_eq!(
            calc.matrix(&rows[..1], &cols[..1]).unwrap().into_values(),
            vec![-0.6]
        );
    }

    #[test]
    fn empty_axes_keep_their_shape_and_validate_the_other_axis() {
        let mut calc = MatrixCalculator::new(37.0, Model::Mathews2004, 1).unwrap();
        assert_eq!(calc.matrix(&[], &["A".into()]).unwrap().shape(), (0, 1));
        assert!(matches!(
            calc.matrix(&["?".into()], &[]),
            Err(MatrixError::InvalidInput(_))
        ));
    }
}
