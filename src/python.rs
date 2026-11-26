use numpy::{IntoPyArray, PyArray2, ndarray::Array2};
use pyo3::{
    exceptions::{PyMemoryError, PyRuntimeError, PyValueError},
    prelude::*,
};

use crate::{MatrixCalculator, MatrixError};

impl From<MatrixError> for PyErr {
    fn from(error: MatrixError) -> Self {
        match error {
            MatrixError::InvalidInput(msg) => PyValueError::new_err(msg),
            MatrixError::Allocation(msg) => PyMemoryError::new_err(msg),
            MatrixError::Runtime(msg) => PyRuntimeError::new_err(msg),
        }
    }
}

#[pyclass]
struct EnergyCalculator {
    inner: MatrixCalculator,
}

#[pymethods]
impl EnergyCalculator {
    #[new]
    #[pyo3(signature = (*, temperature=37.0, min_loop_size=0, model="turner2004", threads=0, backend="cpu"))]
    fn new(
        py: Python<'_>,
        temperature: f64,
        min_loop_size: i64,
        model: &str,
        threads: u16,
        backend: &str,
    ) -> PyResult<Self> {
        if min_loop_size != 0 {
            return Err(PyValueError::new_err(
                "This implementation supports min_loop_size=0",
            ));
        }
        let model = model.parse()?;
        let backend = backend.parse()?;
        let inner = py.detach(|| {
            MatrixCalculator::with_backend(temperature, model, threads.into(), backend)
        })?;
        Ok(Self { inner })
    }

    /// Return an independent row-major float32 matrix of minimum free energies (kcal/mol).
    fn matrix<'py>(
        &mut self,
        py: Python<'py>,
        rows: Vec<String>,
        columns: Vec<String>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let result = py.detach(|| self.inner.matrix(&rows, &columns))?;
        let array = Array2::from_shape_vec(result.shape(), result.into_values())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(array.into_pyarray(py))
    }
}

/// Build one matrix. Reuse EnergyCalculator to cache pairs across calls.
#[pyfunction]
#[pyo3(signature = (rows, columns, *, temperature=37.0, min_loop_size=0, model="turner2004", threads=0, backend="cpu"))]
#[allow(clippy::too_many_arguments)]
fn energy_matrix<'py>(
    py: Python<'py>,
    rows: Vec<String>,
    columns: Vec<String>,
    temperature: f64,
    min_loop_size: i64,
    model: &str,
    threads: u16,
    backend: &str,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    EnergyCalculator::new(py, temperature, min_loop_size, model, threads, backend)?
        .matrix(py, rows, columns)
}

#[pymodule]
fn oligomatrix(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("OPENMP_ENABLED", false)?;
    m.add_class::<EnergyCalculator>()?;
    m.add_function(wrap_pyfunction!(energy_matrix, m)?)?;
    Ok(())
}
