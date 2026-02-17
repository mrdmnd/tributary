//! PyO3 bindings for the headwater sampler.
//!
//! Exposes `headwater.Sampler` as a Python class via maturin.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::common::TIMESTAMP_DIM;
use crate::embedder::EMBEDDING_DIM;
use crate::sampler::{RawBatch, Sampler, SamplerConfig};

/// Convert a `RawBatch` into a Python dict of numpy arrays.
///
/// Uses `PyArray::from_vec` for zero-copy ownership transfer from Rust to NumPy.
fn batch_to_dict<'py>(py: Python<'py>, batch: RawBatch) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let b = batch.batch_size;
    let s = batch.sequence_length;
    let r = batch.max_rows;
    let u = batch.num_unique_texts;

    // [B, S] tensors
    let semantic_types = PyArray1::from_vec(py, batch.semantic_types).reshape([b, s])?;
    dict.set_item("semantic_types", semantic_types)?;

    let column_ids = PyArray1::from_vec(py, batch.column_ids).reshape([b, s])?;
    dict.set_item("column_ids", column_ids)?;

    let seq_row_ids = PyArray1::from_vec(py, batch.seq_row_ids).reshape([b, s])?;
    dict.set_item("seq_row_ids", seq_row_ids)?;

    let numeric_values = PyArray1::from_vec(py, batch.numeric_values).reshape([b, s])?;
    dict.set_item("numeric_values", numeric_values)?;

    let timestamp_values =
        PyArray1::from_vec(py, batch.timestamp_values).reshape([b, s, TIMESTAMP_DIM])?;
    dict.set_item("timestamp_values", timestamp_values)?;

    let bool_values = PyArray1::from_vec(py, batch.bool_values).reshape([b, s])?;
    dict.set_item("bool_values", bool_values)?;

    let categorical_embed_ids =
        PyArray1::from_vec(py, batch.categorical_embed_ids).reshape([b, s])?;
    dict.set_item("categorical_embed_ids", categorical_embed_ids)?;

    let text_embed_ids = PyArray1::from_vec(py, batch.text_embed_ids).reshape([b, s])?;
    dict.set_item("text_embed_ids", text_embed_ids)?;

    let is_null = PyArray1::from_vec(py, batch.is_null).reshape([b, s])?;
    dict.set_item("is_null", is_null)?;

    let is_target = PyArray1::from_vec(py, batch.is_target).reshape([b, s])?;
    dict.set_item("is_target", is_target)?;

    let is_padding = PyArray1::from_vec(py, batch.is_padding).reshape([b, s])?;
    dict.set_item("is_padding", is_padding)?;

    // [B, R, R] adjacency
    let fk_adj = PyArray1::from_vec(py, batch.fk_adj).reshape([b, r, r])?;
    dict.set_item("fk_adj", fk_adj)?;

    // [B, S] permutations
    let col_perm = PyArray1::from_vec(py, batch.col_perm).reshape([b, s])?;
    dict.set_item("col_perm", col_perm)?;

    let out_perm = PyArray1::from_vec(py, batch.out_perm).reshape([b, s])?;
    dict.set_item("out_perm", out_perm)?;

    let in_perm = PyArray1::from_vec(py, batch.in_perm).reshape([b, s])?;
    dict.set_item("in_perm", in_perm)?;

    // [U, D_t] text embeddings (f16 -> stored as u16 for numpy compatibility)
    let text_emb_u16: Vec<u16> = batch
        .text_batch_embeddings
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let text_emb = PyArray1::from_vec(py, text_emb_u16).reshape([u, EMBEDDING_DIM])?;
    dict.set_item("text_batch_embeddings", text_emb)?;

    // Scalar metadata
    dict.set_item(
        "target_stype",
        PyArray1::from_vec(py, vec![batch.target_stype]),
    )?;
    dict.set_item("task_idx", PyArray1::from_vec(py, vec![batch.task_idx]))?;

    Ok(dict)
}

/// Python-visible Sampler class.
#[pyclass(name = "Sampler")]
struct PySampler {
    inner: Option<Sampler>,
}

#[pymethods]
impl PySampler {
    #[new]
    #[pyo3(signature = (
        db_path,
        rank = 0,
        world_size = 1,
        split_ratios = (0.8, 0.1, 0.1),
        split_seed = 123,
        seed = 42,
        num_prefetch = 3,
        default_batch_size = 32,
        default_sequence_length = 1024,
        bfs_child_width = 16,
        max_rows_per_seq = 200,
    ))]
    fn new(
        py: Python<'_>,
        db_path: String,
        rank: u32,
        world_size: u32,
        split_ratios: (f32, f32, f32),
        split_seed: u64,
        seed: u64,
        num_prefetch: usize,
        default_batch_size: u32,
        default_sequence_length: u32,
        bfs_child_width: u32,
        max_rows_per_seq: u32,
    ) -> PyResult<Self> {
        let config = SamplerConfig {
            db_path,
            rank,
            world_size,
            split_ratios,
            split_seed,
            seed,
            num_prefetch,
            batch_size: default_batch_size,
            sequence_length: default_sequence_length,
            bfs_child_width,
            max_rows_per_seq,
        };

        let sampler = py
            .detach(|| Sampler::new(config))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create sampler: {e}")))?;

        Ok(Self {
            inner: Some(sampler),
        })
    }

    /// Pull the next training batch. Returns a dict of numpy arrays.
    /// Blocks until a batch is available (GIL released during wait).
    fn next_train_batch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let sampler = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Sampler has been shut down"))?;

        let batch = py
            .detach(|| sampler.next_train_batch())
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;

        batch_to_dict(py, batch)
    }

    /// Pull the next validation batch. Returns a dict of numpy arrays.
    /// Blocks until a batch is available (GIL released during wait).
    fn next_val_batch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let sampler = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Sampler has been shut down"))?;

        let batch = py
            .detach(|| sampler.next_val_batch())
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;

        batch_to_dict(py, batch)
    }

    /// Get column-name embeddings: [num_columns, EMBEDDING_DIM] as float16
    /// (returned as uint16 numpy array for f16 compatibility).
    fn column_embeddings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u16>>> {
        let sampler = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Sampler has been shut down"))?;

        let embeddings = py.detach(|| sampler.column_embeddings());
        let num_cols = embeddings.len() / EMBEDDING_DIM;
        let u16_data: Vec<u16> = embeddings.iter().map(|v| v.to_bits()).collect();
        let arr = PyArray1::from_vec(py, u16_data).reshape([num_cols, EMBEDDING_DIM])?;
        Ok(arr)
    }

    /// Get categorical embeddings: [Vc, EMBEDDING_DIM] as float16
    /// (returned as uint16 numpy array for f16 compatibility).
    fn categorical_embeddings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u16>>> {
        let sampler = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Sampler has been shut down"))?;

        let embeddings = py.detach(|| sampler.categorical_embeddings());
        let num_embs = embeddings.len() / EMBEDDING_DIM;
        let u16_data: Vec<u16> = embeddings.iter().map(|v| v.to_bits()).collect();
        let arr = PyArray1::from_vec(py, u16_data).reshape([num_embs, EMBEDDING_DIM])?;
        Ok(arr)
    }

    /// Get database metadata as a Python dict.
    fn database_metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let sampler = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Sampler has been shut down"))?;

        let db = sampler.database();
        let dict = PyDict::new(py);

        dict.set_item("num_tables", db.metadata.table_metadata.len())?;
        dict.set_item("num_columns", db.metadata.column_metadata.len())?;
        dict.set_item("num_tasks", db.metadata.task_metadata.len())?;
        dict.set_item("global_ts_mean_us", db.metadata.global_ts_mean_us)?;
        dict.set_item("global_ts_std_us", db.metadata.global_ts_std_us)?;

        let task_names: Vec<String> = db
            .metadata
            .task_metadata
            .iter()
            .map(|t| t.name.clone())
            .collect();
        dict.set_item("task_names", task_names)?;

        Ok(dict)
    }

    /// Shut down the sampler, draining channels and joining threads.
    fn shutdown(&mut self) {
        if let Some(mut sampler) = self.inner.take() {
            sampler.shutdown();
        }
    }
}

/// Register the headwater Python module.
#[pymodule]
fn headwater(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySampler>()?;
    Ok(())
}
