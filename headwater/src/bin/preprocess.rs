//! Preprocessor binary: transforms a raw parquet database into the binary format
//! used by the sampling pipeline.
//!
//! ## Input
//!
//! A data directory with the following layout:
//! - `metadata/<dataset>.json` — human-annotated schema (tables, columns, semantic types, FKs, tasks)
//! - `raw/<dataset>/<table_name>.parquet` — one parquet file per table
//!
//! ## Output
//!
//! Written to `processed/<dataset>/` under the data directory:
//! - `metadata.json`              — serialized `DatabaseMetadata` (JSON, schema + stats + tasks)
//! - `column_embeddings.bin`      — flat `[C, EMBEDDING_DIM]` f16 array (one per global column)
//! - `categorical_embeddings.bin` — flat `[Vc, EMBEDDING_DIM]` f16 array (all categorical value embeddings)
//! - `text_embeddings.bin`        — flat `[Vt, EMBEDDING_DIM]` f16 array (all text value embeddings)
//! - `graph.bin`                  — bidirectional CSR graph (FK edges)
//! - `tables/<table_name>.bin`    — packed column store per table
//! - `tasks/<task_name>.bin`      — materialized prediction task per task
//!
//! ## Usage
//!
//! ```sh
//! cargo run --release --bin preprocess -- --data-dir data --dataset rel-stack
//! ```

use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::fs::{self, File};
use std::path::PathBuf;

use arrow::array::*;
use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, Int32Type, SchemaRef, TimeUnit};
use arrow::record_batch::RecordBatch;
use chrono::{DateTime, Datelike, Timelike, Utc};
use clap::Parser;
use futures::stream::{self, StreamExt};
use indexmap::IndexMap;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Deserialize;
use tracing::{info, warn};

use half::f16;
use headwater::common::*;
use headwater::embedder::{EMBEDDING_DIM, Embedder, EmbedderConfig, EmbedderError, EmbedderTrait};
use indicatif::{HumanCount, HumanDuration, ProgressBar, ProgressStyle};

// ============================================================================
// CLI
// ============================================================================

#[derive(Parser, Debug)]
#[command(about = "Preprocess a parquet database into binary training format")]
struct Args {
    /// Name of the dataset (e.g. "rel-stack"). Parquets are read from
    /// `<data-dir>/raw/<dataset>/`, metadata from `<data-dir>/metadata/<dataset>.json`,
    /// and output is written to `<data-dir>/processed/<dataset>/`.
    #[arg(long)]
    dataset: String,

    /// Path to the top-level data directory (contains raw/, metadata/, processed/).
    #[arg(long)]
    data_dir: PathBuf,

    /// Embedding batch size (strings per API request). The Baseten embedding
    /// API enforces a limit of 128 strings per request.
    #[arg(long, default_value_t = 128)]
    batch_size: usize,

    /// Maximum number of concurrent embedding API requests.
    #[arg(long, default_value_t = 1024)]
    max_concurrent: usize,

    /// Skip embedding (use zero vectors). Useful for testing the pipeline
    /// without the embedding API.
    #[arg(long, default_value_t = false)]
    skip_embeddings: bool,
}

// ============================================================================
// Raw metadata JSON schema (serde)
// ============================================================================

#[derive(Deserialize, Debug)]
struct RawMetadata {
    #[allow(dead_code)]
    name: String,
    tables: IndexMap<String, RawTable>,
    tasks: IndexMap<String, RawTask>,
}

#[derive(Deserialize, Debug, Clone)]
struct RawTable {
    primary_key: Option<String>,
    temporal_column: Option<String>,
    columns: IndexMap<String, RawColumn>,
}

#[derive(Deserialize, Debug, Clone)]
struct RawColumn {
    stype: String,
    foreign_key: Option<String>,
    description: Option<String>,
}

#[derive(Deserialize, Debug)]
struct RawColumnRef {
    table: String,
    column: String,
}

#[derive(Deserialize, Debug)]
struct RawTask {
    query: String,
    anchor_table: String,
    anchor_key: String,
    target_column: String,
    target_stype: String,
    observation_time_column: Option<String>,
    #[serde(default)]
    columns_to_drop: Vec<RawColumnRef>,
}

// ============================================================================
// Helpers
// ============================================================================

fn parse_stype(s: &str) -> SemanticType {
    match s.to_lowercase().as_str() {
        "identifier" => SemanticType::Identifier,
        "numerical" => SemanticType::Numerical,
        "timestamp" => SemanticType::Timestamp,
        "boolean" => SemanticType::Boolean,
        "categorical" => SemanticType::Categorical,
        "text" => SemanticType::Text,
        "ignored" => SemanticType::Ignored,
        other => {
            warn!("Unknown stype '{other}', treating as Ignored");
            SemanticType::Ignored
        }
    }
}

/// Build a packed little-endian bitmap where bit `i` is 1 iff the value at row `i`
/// is non-null in the given Arrow array.
fn build_validity_bitmap(array: &dyn Array) -> Vec<u8> {
    let n = array.len();
    let byte_len = packed_bit_bytes(n);
    let mut bitmap = vec![0u8; byte_len];
    for i in 0..n {
        if !array.is_null(i) {
            bitmap[i / 8] |= 1 << (i % 8);
        }
    }
    bitmap
}

/// Build a packed little-endian bitmap where bit `i` is 1 iff condition is true
/// for row `i`.
fn build_bitmap_from_fn(n: usize, f: impl Fn(usize) -> bool) -> Vec<u8> {
    let byte_len = packed_bit_bytes(n);
    let mut bitmap = vec![0u8; byte_len];
    for i in 0..n {
        if f(i) {
            bitmap[i / 8] |= 1 << (i % 8);
        }
    }
    bitmap
}

/// Cast an Arrow array to Int64, handling Int32, Int64, UInt32, UInt64.
fn cast_to_i64(array: &dyn Array) -> Option<Int64Array> {
    arrow::compute::cast(array, &DataType::Int64)
        .ok()
        .and_then(|a| a.as_any().downcast_ref::<Int64Array>().cloned())
}

/// Cast an Arrow array to Float64 for numerical stats.
fn cast_to_f64(array: &dyn Array) -> Option<Float64Array> {
    arrow::compute::cast(array, &DataType::Float64)
        .ok()
        .and_then(|a| a.as_any().downcast_ref::<Float64Array>().cloned())
}

/// Extract string values from an Arrow array (handles Utf8, LargeUtf8, Dictionary).
fn array_to_strings(array: &dyn Array) -> Vec<Option<String>> {
    let n = array.len();
    let mut result = Vec::with_capacity(n);

    if let Some(sa) = array.as_any().downcast_ref::<StringArray>() {
        for i in 0..n {
            result.push(if sa.is_null(i) {
                None
            } else {
                Some(sa.value(i).to_string())
            });
        }
    } else if let Some(sa) = array.as_any().downcast_ref::<LargeStringArray>() {
        for i in 0..n {
            result.push(if sa.is_null(i) {
                None
            } else {
                Some(sa.value(i).to_string())
            });
        }
    } else if let Some(da) = array.as_any().downcast_ref::<DictionaryArray<Int32Type>>() {
        let values: &dyn Array = da.values().as_ref();
        if let Some(sv) = values.as_any().downcast_ref::<StringArray>() {
            for i in 0..n {
                if da.is_null(i) {
                    result.push(None);
                } else {
                    let key: usize = da.keys().value(i) as usize;
                    result.push(Some(sv.value(key).to_string()));
                }
            }
        } else {
            // Fallback: try casting the whole thing to StringArray
            if let Ok(cast) = arrow::compute::cast(array, &DataType::Utf8) {
                return array_to_strings(cast.as_ref());
            }
            result.resize(n, None);
        }
    } else {
        // Try generic cast to Utf8
        if let Ok(cast) = arrow::compute::cast(array, &DataType::Utf8) {
            if let Some(sa) = cast.as_any().downcast_ref::<StringArray>() {
                for i in 0..n {
                    result.push(if sa.is_null(i) {
                        None
                    } else {
                        Some(sa.value(i).to_string())
                    });
                }
                return result;
            }
        }
        result.resize(n, None);
    }
    result
}

/// Cast an Arrow array to TimestampMicrosecondArray.
fn cast_to_timestamp_us(array: &dyn Array) -> Option<TimestampMicrosecondArray> {
    let target_dt = DataType::Timestamp(TimeUnit::Microsecond, None);
    arrow::compute::cast(array, &target_dt).ok().and_then(|a| {
        a.as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .cloned()
    })
}

/// Encode a single timestamp (epoch microseconds) into the 15-element feature vector.
fn encode_timestamp(us: i64, mean_us: f64, std_us: f64) -> [f32; TIMESTAMP_DIM] {
    let mut out = [0.0f32; TIMESTAMP_DIM];

    let dt = DateTime::<Utc>::from_timestamp_micros(us)
        .unwrap_or_default()
        .naive_utc();

    let cyclic = |val: f64, period: f64| -> (f32, f32) {
        let angle = 2.0 * PI * val / period;
        (angle.sin() as f32, angle.cos() as f32)
    };

    let (s, c) = cyclic(dt.second() as f64, 60.0);
    out[TS_SECOND_OF_MINUTE] = s;
    out[TS_SECOND_OF_MINUTE + 1] = c;

    let (s, c) = cyclic(dt.minute() as f64, 60.0);
    out[TS_MINUTE_OF_HOUR] = s;
    out[TS_MINUTE_OF_HOUR + 1] = c;

    let (s, c) = cyclic(dt.hour() as f64, 24.0);
    out[TS_HOUR_OF_DAY] = s;
    out[TS_HOUR_OF_DAY + 1] = c;

    let (s, c) = cyclic(dt.weekday().num_days_from_monday() as f64, 7.0);
    out[TS_DAY_OF_WEEK] = s;
    out[TS_DAY_OF_WEEK + 1] = c;

    let (s, c) = cyclic((dt.day() - 1) as f64, 31.0);
    out[TS_DAY_OF_MONTH] = s;
    out[TS_DAY_OF_MONTH + 1] = c;

    let (s, c) = cyclic((dt.month() - 1) as f64, 12.0);
    out[TS_MONTH_OF_YEAR] = s;
    out[TS_MONTH_OF_YEAR + 1] = c;

    let (s, c) = cyclic((dt.ordinal() - 1) as f64, 366.0);
    out[TS_DAY_OF_YEAR] = s;
    out[TS_DAY_OF_YEAR + 1] = c;

    let z = if std_us > 0.0 {
        ((us as f64 - mean_us) / std_us) as f32
    } else {
        0.0
    };
    out[TS_ZSCORE_EPOCH] = z;

    out
}

// ============================================================================
// Per-table loaded data
// ============================================================================

struct LoadedTable {
    name: String,
    batch: RecordBatch,
    schema: SchemaRef,
    raw_table: RawTable,
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();

    let raw_dir = args.data_dir.join("raw").join(&args.dataset);
    let metadata_path = args
        .data_dir
        .join("metadata")
        .join(format!("{}.json", &args.dataset));
    let output_dir = args.data_dir.join("processed").join(&args.dataset);

    info!("Dataset:  {}", args.dataset);
    info!("Raw dir:  {}", raw_dir.display());
    info!("Metadata: {}", metadata_path.display());
    info!("Output:   {}", output_dir.display());

    // Create output directories
    fs::create_dir_all(&output_dir)?;
    fs::create_dir_all(output_dir.join("tables"))?;
    fs::create_dir_all(output_dir.join("tasks"))?;

    let pipeline_start = std::time::Instant::now();

    // ── Step 1: Parse metadata ─────────────────────────────────────────
    info!("Step 1: Parsing metadata...");
    let metadata_text = fs::read_to_string(&metadata_path)?;
    let raw_meta: RawMetadata = serde_json::from_str(&metadata_text)?;
    info!(
        "  Found {} tables, {} tasks",
        raw_meta.tables.len(),
        raw_meta.tasks.len()
    );

    // ── Step 2: Load parquet files and assign global indices ─────────────
    info!("Step 2: Loading parquet files...");
    let num_tables = raw_meta.tables.len();
    let pb_load = ProgressBar::new(num_tables as u64);
    pb_load.set_style(
        ProgressStyle::with_template(
            "  Loading    {bar:40.cyan/blue} {pos}/{len} tables [{elapsed_precise}]",
        )
        .unwrap()
        .progress_chars("##-"),
    );
    let mut loaded_tables: Vec<LoadedTable> = Vec::new();
    let mut global_row_offset: u32 = 0;
    let mut global_col_offset: u32 = 0;

    // These track per-table ranges for metadata
    let mut table_row_ranges: Vec<(RowIdx, RowIdx)> = Vec::new();
    let mut table_col_ranges: Vec<(ColumnIdx, ColumnIdx)> = Vec::new();

    // Column info accumulated in global order
    let mut all_col_names: Vec<String> = Vec::new();
    let mut all_col_table_names: Vec<String> = Vec::new();
    let mut all_col_stypes: Vec<SemanticType> = Vec::new();
    let mut all_col_fkey_targets: Vec<Option<String>> = Vec::new(); // "table.column" strings
    let mut all_col_descriptions: Vec<Option<String>> = Vec::new();

    // Table-level info
    let mut table_names: Vec<String> = Vec::new();
    let mut table_pkey_col_names: Vec<Option<String>> = Vec::new();
    let mut table_temporal_col_names: Vec<Option<String>> = Vec::new();

    for (table_name, raw_table) in &raw_meta.tables {
        let parquet_path = raw_dir.join(format!("{table_name}.parquet"));
        info!("  Loading {}", parquet_path.display());

        let file = File::open(&parquet_path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
            .with_batch_size(1_000_000)
            .build()?;

        let schema = reader.schema().clone();
        let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>()?;
        let batch = if batches.len() == 1 {
            batches.into_iter().next().unwrap()
        } else {
            concat_batches(&schema, &batches)?
        };

        let num_rows = batch.num_rows() as u32;
        let num_cols = raw_table.columns.len() as u32;

        table_row_ranges.push((
            RowIdx(global_row_offset),
            RowIdx(global_row_offset + num_rows),
        ));
        table_col_ranges.push((
            ColumnIdx(global_col_offset),
            ColumnIdx(global_col_offset + num_cols),
        ));

        table_names.push(table_name.clone());
        table_pkey_col_names.push(raw_table.primary_key.clone());
        table_temporal_col_names.push(raw_table.temporal_column.clone());

        for (col_name, raw_col) in &raw_table.columns {
            all_col_names.push(col_name.clone());
            all_col_table_names.push(table_name.clone());
            all_col_stypes.push(parse_stype(&raw_col.stype));
            all_col_fkey_targets.push(raw_col.foreign_key.clone());
            all_col_descriptions.push(raw_col.description.clone());
        }

        info!(
            "    {} rows, {} columns (global rows [{}, {}), global cols [{}, {}))",
            num_rows,
            num_cols,
            global_row_offset,
            global_row_offset + num_rows,
            global_col_offset,
            global_col_offset + num_cols,
        );

        global_row_offset += num_rows;
        global_col_offset += num_cols;

        loaded_tables.push(LoadedTable {
            name: table_name.clone(),
            batch,
            schema,
            raw_table: RawTable {
                primary_key: raw_table.primary_key.clone(),
                temporal_column: raw_table.temporal_column.clone(),
                columns: raw_table.columns.clone(),
            },
        });
        pb_load.inc(1);
    }
    pb_load.finish_and_clear();

    let total_rows = global_row_offset as usize;
    let total_cols = global_col_offset as usize;
    info!(
        "  Loaded {} tables ({} rows, {} cols)",
        loaded_tables.len(),
        HumanCount(total_rows as u64),
        total_cols
    );

    // ── Step 3: Compute column statistics ────────────────────────────────
    info!("Step 3: Computing column statistics...");
    let pb_stats = ProgressBar::new(total_cols as u64);
    pb_stats.set_style(
        ProgressStyle::with_template(
            "  Stats      {bar:40.cyan/blue} {pos}/{len} cols [{elapsed_precise}]",
        )
        .unwrap()
        .progress_chars("##-"),
    );
    let mut all_col_stats: Vec<ColumnStats> = Vec::new();
    let mut global_col_idx = 0usize;

    // Accumulators for global timestamp statistics (across all columns and tables).
    let mut global_ts_sum = 0.0f64;
    let mut global_ts_count = 0u64;

    for (ti, lt) in loaded_tables.iter().enumerate() {
        for (col_name, _raw_col) in &lt.raw_table.columns {
            let stype = all_col_stypes[global_col_idx];

            // Find the Arrow column by name
            let arrow_col = lt
                .schema
                .index_of(col_name)
                .ok()
                .map(|i| lt.batch.column(i));

            let stats = match stype {
                SemanticType::Identifier => {
                    let num_nulls = arrow_col.map(|a| a.null_count() as u64).unwrap_or(0);
                    ColumnStats::Identifier { num_nulls }
                }
                SemanticType::Numerical => {
                    if let Some(array) = arrow_col.and_then(|a| cast_to_f64(a.as_ref())) {
                        let num_nulls = array.null_count() as u64;
                        let n = array.len();
                        let mut sum = 0.0f64;
                        let mut count = 0u64;
                        let mut min_val = f64::MAX;
                        let mut max_val = f64::MIN;
                        for i in 0..n {
                            if !array.is_null(i) {
                                let v = array.value(i);
                                sum += v;
                                count += 1;
                                if v < min_val {
                                    min_val = v;
                                }
                                if v > max_val {
                                    max_val = v;
                                }
                            }
                        }
                        let mean = if count > 0 { sum / count as f64 } else { 0.0 };
                        let mut var_sum = 0.0f64;
                        for i in 0..n {
                            if !array.is_null(i) {
                                let d = array.value(i) - mean;
                                var_sum += d * d;
                            }
                        }
                        let std = if count > 1 {
                            (var_sum / (count - 1) as f64).sqrt()
                        } else {
                            1.0
                        };
                        ColumnStats::Numerical {
                            num_nulls,
                            min: if count > 0 { min_val } else { 0.0 },
                            max: if count > 0 { max_val } else { 0.0 },
                            mean,
                            std,
                        }
                    } else {
                        warn!(
                            "  Column {}.{} could not be cast to f64",
                            table_names[ti], col_name
                        );
                        ColumnStats::Numerical {
                            num_nulls: 0,
                            min: 0.0,
                            max: 0.0,
                            mean: 0.0,
                            std: 1.0,
                        }
                    }
                }
                SemanticType::Timestamp => {
                    if let Some(array) = arrow_col.and_then(|a| cast_to_timestamp_us(a.as_ref())) {
                        let num_nulls = array.null_count() as u64;
                        let n = array.len();
                        let mut sum = 0.0f64;
                        let mut count = 0u64;
                        let mut min_val = i64::MAX;
                        let mut max_val = i64::MIN;
                        for i in 0..n {
                            if !array.is_null(i) {
                                let v = array.value(i);
                                sum += v as f64;
                                count += 1;
                                if v < min_val {
                                    min_val = v;
                                }
                                if v > max_val {
                                    max_val = v;
                                }
                            }
                        }
                        // Accumulate into global timestamp stats
                        global_ts_sum += sum;
                        global_ts_count += count;

                        let mean_us = if count > 0 { sum / count as f64 } else { 0.0 };
                        let mut var_sum = 0.0f64;
                        for i in 0..n {
                            if !array.is_null(i) {
                                let d = array.value(i) as f64 - mean_us;
                                var_sum += d * d;
                            }
                        }
                        let std_us = if count > 1 {
                            (var_sum / (count - 1) as f64).sqrt()
                        } else {
                            1.0
                        };
                        ColumnStats::Timestamp {
                            num_nulls,
                            min_us: if count > 0 { min_val } else { 0 },
                            max_us: if count > 0 { max_val } else { 0 },
                            mean_us,
                            std_us,
                        }
                    } else {
                        warn!(
                            "  Column {}.{} could not be cast to timestamp",
                            table_names[ti], col_name
                        );
                        ColumnStats::Timestamp {
                            num_nulls: 0,
                            min_us: 0,
                            max_us: 0,
                            mean_us: 0.0,
                            std_us: 1.0,
                        }
                    }
                }
                SemanticType::Boolean => {
                    if let Some(arr) = arrow_col {
                        let num_nulls = arr.null_count() as u64;
                        // Try to get as BooleanArray or cast
                        let ba = if let Some(b) = arr.as_any().downcast_ref::<BooleanArray>() {
                            Some(b.clone())
                        } else {
                            arrow::compute::cast(arr.as_ref(), &DataType::Boolean)
                                .ok()
                                .and_then(|a| a.as_any().downcast_ref::<BooleanArray>().cloned())
                        };
                        if let Some(ba) = ba {
                            let mut num_true = 0u64;
                            let mut num_false = 0u64;
                            for i in 0..ba.len() {
                                if !ba.is_null(i) {
                                    if ba.value(i) {
                                        num_true += 1;
                                    } else {
                                        num_false += 1;
                                    }
                                }
                            }
                            ColumnStats::Boolean {
                                num_nulls,
                                num_true,
                                num_false,
                            }
                        } else {
                            ColumnStats::Boolean {
                                num_nulls,
                                num_true: 0,
                                num_false: 0,
                            }
                        }
                    } else {
                        ColumnStats::Boolean {
                            num_nulls: 0,
                            num_true: 0,
                            num_false: 0,
                        }
                    }
                }
                SemanticType::Categorical => {
                    if let Some(arr) = arrow_col {
                        let num_nulls = arr.null_count() as u64;
                        let strings = array_to_strings(arr.as_ref());
                        let mut cats: HashSet<String> = HashSet::new();
                        for s in &strings {
                            if let Some(v) = s {
                                cats.insert(v.clone());
                            }
                        }
                        let mut categories: Vec<String> = cats.into_iter().collect();
                        categories.sort();
                        // cat_emb_start is populated later during vocab collection (step 4b).
                        ColumnStats::Categorical {
                            num_nulls,
                            categories,
                            cat_emb_start: 0,
                        }
                    } else {
                        ColumnStats::Categorical {
                            num_nulls: 0,
                            categories: Vec::new(),
                            cat_emb_start: 0,
                        }
                    }
                }
                SemanticType::Text => {
                    let num_nulls = arrow_col.map(|a| a.null_count() as u64).unwrap_or(0);
                    ColumnStats::Text { num_nulls }
                }
                SemanticType::Ignored => ColumnStats::Ignored,
            };

            all_col_stats.push(stats);
            pb_stats.inc(1);
            global_col_idx += 1;
        }
    }
    pb_stats.finish_and_clear();
    info!("  Computed stats for {total_cols} columns");

    // Compute global timestamp mean, then second pass for global std.
    let global_ts_mean_us = if global_ts_count > 0 {
        global_ts_sum / global_ts_count as f64
    } else {
        0.0
    };
    let global_ts_std_us = {
        let mut var_sum = 0.0f64;
        let mut ci = 0usize;
        for lt in loaded_tables.iter() {
            for (col_name, _) in &lt.raw_table.columns {
                if all_col_stypes[ci] == SemanticType::Timestamp {
                    if let Some(arr) = lt
                        .schema
                        .index_of(col_name)
                        .ok()
                        .map(|i| lt.batch.column(i))
                    {
                        if let Some(ts_arr) = cast_to_timestamp_us(arr.as_ref()) {
                            for i in 0..ts_arr.len() {
                                if !ts_arr.is_null(i) {
                                    let d = ts_arr.value(i) as f64 - global_ts_mean_us;
                                    var_sum += d * d;
                                }
                            }
                        }
                    }
                }
                ci += 1;
            }
        }
        if global_ts_count > 1 {
            (var_sum / (global_ts_count - 1) as f64).sqrt()
        } else {
            1.0
        }
    };
    info!(
        "  Global timestamp stats: count={}, mean={:.0}μs, std={:.0}μs",
        global_ts_count, global_ts_mean_us, global_ts_std_us,
    );

    // ── Step 4: Build embedding vocabulary ───────────────────────────────
    info!("Step 4: Building embedding vocabulary...");

    // Pre-truncate strings to this many characters before embedding.  The
    // tokenizer truncates at MAX_SEQ_LEN tokens anyway (~4 chars/token for
    // English), so anything beyond this is wasted work.  Truncating here also
    // makes the length-sort more representative of actual token counts, which
    // reduces padding waste in GPU batches.
    let max_chars: usize = 512 * 4; // ≈ 512 tokens

    // Vocabulary: ordered lists of unique strings to embed.
    // col_name_strings: one per global column — these become ColumnMetadata.embedding
    let mut col_name_strings: Vec<String> = Vec::with_capacity(total_cols);

    // Categorical embedding vocab: maps formatted string -> CategoricalEmbeddingIdx.
    // Indices are 0-based into categorical_embeddings.bin.
    // Each column's categories form a contiguous block; cat_emb_start is recorded per column.
    let mut cat_map: HashMap<String, CategoricalEmbeddingIdx> = HashMap::new();
    let mut cat_list: Vec<String> = Vec::new();

    // Text embedding vocab: maps raw string -> TextEmbeddingIdx.
    // Indices are 0-based into text_embeddings.bin.
    let mut text_map: HashMap<String, TextEmbeddingIdx> = HashMap::new();
    let mut text_list: Vec<String> = Vec::new();

    let truncate_string = |s: String, limit: usize| -> String {
        if s.len() > limit {
            let end = s.floor_char_boundary(limit);
            s[..end].to_string()
        } else {
            s
        }
    };

    // 4a) Column name embeddings
    for (ti, lt) in loaded_tables.iter().enumerate() {
        for (col_name, raw_col) in &lt.raw_table.columns {
            let s = if let Some(desc) = &raw_col.description {
                format!("{} of {}: {}", col_name, table_names[ti], desc)
            } else {
                format!("{} of {}", col_name, table_names[ti])
            };
            col_name_strings.push(s);
        }
    }

    // 4b) Categorical value embeddings — each column's categories get a contiguous block
    {
        let mut ci = 0usize;
        for lt in loaded_tables.iter() {
            for (col_name, _) in &lt.raw_table.columns {
                if let ColumnStats::Categorical {
                    categories,
                    cat_emb_start,
                    ..
                } = &mut all_col_stats[ci]
                {
                    // Record the start offset for this column's block in the categorical table.
                    *cat_emb_start = cat_list.len() as u32;
                    for cat_val in categories.iter() {
                        let s = truncate_string(format!("{col_name} is {cat_val}"), max_chars);
                        if !cat_map.contains_key(&s) {
                            let idx = CategoricalEmbeddingIdx(cat_list.len() as u32);
                            cat_map.insert(s.clone(), idx);
                            cat_list.push(s);
                        }
                    }
                }
                ci += 1;
            }
        }
    }

    // 4c) Text value embeddings — collect unique non-null text values
    {
        let mut ci = 0usize;
        for lt in loaded_tables.iter() {
            for (col_name, _) in &lt.raw_table.columns {
                if all_col_stypes[ci] == SemanticType::Text {
                    if let Some(arr) = lt
                        .schema
                        .index_of(col_name)
                        .ok()
                        .map(|i| lt.batch.column(i))
                    {
                        let strings = array_to_strings(arr.as_ref());
                        for s in strings {
                            if let Some(v) = s {
                                let v = truncate_string(v, max_chars);
                                if !text_map.contains_key(&v) {
                                    let idx = TextEmbeddingIdx(text_list.len() as u32);
                                    text_map.insert(v.clone(), idx);
                                    text_list.push(v);
                                }
                            }
                        }
                    }
                }
                ci += 1;
            }
        }
    }

    info!(
        "  Column name strings: {}, Categorical vocab: {}, Text vocab: {}",
        col_name_strings.len(),
        cat_list.len(),
        text_list.len()
    );

    // ── Embedding via API ─────────────────────────────────────────────────
    info!("Step 4b: Computing embeddings via API...");

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    // Combine all strings for a single embedding API pass.
    // Layout: [column names | categorical values | text values]
    let all_strings_to_embed: Vec<String> = col_name_strings
        .iter()
        .chain(cat_list.iter())
        .chain(text_list.iter())
        .cloned()
        .collect();

    let total_to_embed = all_strings_to_embed.len();
    info!(
        "  Embedding {} strings (batch_size={}, max_concurrent={})...",
        HumanCount(total_to_embed as u64),
        args.batch_size,
        args.max_concurrent,
    );

    let all_embeddings: Vec<Vec<f16>> = if total_to_embed == 0 {
        Vec::new()
    } else if args.skip_embeddings {
        info!("  --skip-embeddings: using zero vectors");
        vec![vec![f16::ZERO; EMBEDDING_DIM]; total_to_embed]
    } else {
        let config = EmbedderConfig::from_baseten_env()?;
        let embedder = Embedder::new(config)?;
        let refs: Vec<&str> = all_strings_to_embed.iter().map(|s| s.as_str()).collect();

        let pb = ProgressBar::new(total_to_embed as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "  Embedding  {bar:40.cyan/blue} {pos}/{len} strings [{elapsed_precise} elapsed, ETA {eta_precise}]",
            )
            .unwrap()
            .progress_chars("##-"),
        );
        pb.enable_steady_tick(std::time::Duration::from_secs(1));
        let embed_start = std::time::Instant::now();

        let batch_ranges: Vec<(usize, usize)> = {
            let mut ranges = Vec::new();
            let mut start = 0;
            while start < refs.len() {
                let end = (start + args.batch_size).min(refs.len());
                ranges.push((start, end));
                start = end;
            }
            ranges
        };
        let num_batches = batch_ranges.len();

        let result: Vec<Vec<f16>> = rt.block_on(async {
            let embedder_ref = &embedder;
            let pb_ref = &pb;
            let mut all_vecs: Vec<Vec<f16>> = Vec::with_capacity(total_to_embed);
            let mut embedded_so_far: u64 = 0;
            let mut next_log_pct: f64 = 5.0;

            let mut result_stream = stream::iter(batch_ranges.into_iter())
                .map(|(start, end)| {
                    let chunk: Vec<&str> = refs[start..end].to_vec();
                    async move {
                        let n = chunk.len();
                        let flat = embedder_ref.embed_texts(&chunk).await?;
                        Ok::<_, EmbedderError>((n, flat))
                    }
                })
                .buffered(args.max_concurrent.max(1));

            while let Some(batch_result) = result_stream.next().await {
                let (n, flat): (usize, Vec<f16>) = batch_result?;
                for emb in flat.chunks_exact(EMBEDDING_DIM) {
                    all_vecs.push(emb.to_vec());
                }
                embedded_so_far += n as u64;
                pb_ref.inc(n as u64);

                let pct = embedded_so_far as f64 / total_to_embed as f64 * 100.0;
                if pct >= next_log_pct {
                    let elapsed = embed_start.elapsed().as_secs_f64();
                    let rate = embedded_so_far as f64 / elapsed;
                    let remaining = (total_to_embed as f64 - embedded_so_far as f64) / rate;
                    info!(
                        "  Embedding progress: {}/{} ({:.0}%) — {:.0} strings/sec, ~{:.0}s remaining",
                        HumanCount(embedded_so_far),
                        HumanCount(total_to_embed as u64),
                        pct,
                        rate,
                        remaining,
                    );
                    next_log_pct = (pct / 5.0).floor() * 5.0 + 5.0;
                }
            }

            Ok::<_, Box<dyn std::error::Error>>(all_vecs)
        })?;

        pb.finish_with_message("done");
        let embed_elapsed = embed_start.elapsed();
        info!(
            "  Embedding complete: {} strings in {} batches, {} ({:.0} strings/sec)",
            HumanCount(total_to_embed as u64),
            num_batches,
            HumanDuration(embed_elapsed),
            total_to_embed as f64 / embed_elapsed.as_secs_f64(),
        );
        result
    };

    // Split results: column-name | categorical | text.
    let col_count = col_name_strings.len();
    let cat_count = cat_list.len();
    let column_embeddings: &[Vec<f16>] = &all_embeddings[..col_count];
    let categorical_embeddings: &[Vec<f16>] = &all_embeddings[col_count..col_count + cat_count];
    let text_embeddings: &[Vec<f16>] = &all_embeddings[col_count + cat_count..];

    info!(
        "  Got {} column embeddings, {} categorical embeddings, {} text embeddings",
        column_embeddings.len(),
        categorical_embeddings.len(),
        text_embeddings.len()
    );

    // ── Step 5: Encode tables and write table_XXXX.bin ───────────────────
    info!("Step 5: Encoding tables...");
    let pb_enc = ProgressBar::new(num_tables as u64);
    pb_enc.set_style(
        ProgressStyle::with_template(
            "  Encoding   {bar:40.cyan/blue} {pos}/{len} tables [{elapsed_precise}]",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    global_col_idx = 0;
    for lt in loaded_tables.iter() {
        let num_rows = lt.batch.num_rows();
        let mut col_writers_data: Vec<ColumnWriterData> = Vec::new();

        for (col_name, _raw_col) in &lt.raw_table.columns {
            let stype = all_col_stypes[global_col_idx];
            let stats = &all_col_stats[global_col_idx];
            let arrow_col = lt
                .schema
                .index_of(col_name)
                .ok()
                .map(|i| lt.batch.column(i));

            let data = encode_column(
                stype,
                stats,
                arrow_col.map(|a| a.as_ref()),
                num_rows,
                col_name,
                &cat_map,
                &text_map,
                global_ts_mean_us,
                global_ts_std_us,
            );
            col_writers_data.push(data);
            global_col_idx += 1;
        }

        // Convert to ColumnWriter references for write_table_bin
        let writers: Vec<ColumnWriter> = col_writers_data.iter().map(|d| d.as_writer()).collect();
        let path = output_dir.join(format!("tables/{}.bin", lt.name));
        write_table_bin(&path, num_rows as u32, &writers)?;
        pb_enc.inc(1);
    }
    pb_enc.finish_and_clear();
    info!("  Wrote {} table files", num_tables);

    // Build (table_name, column_name) -> global ColumnIdx lookup for FK and columns_to_drop resolution.
    let col_lookup: HashMap<(String, String), ColumnIdx> = {
        let mut map = HashMap::new();
        let mut ci = 0u32;
        for (table_name, raw_table) in &raw_meta.tables {
            for (col_name, _) in &raw_table.columns {
                map.insert((table_name.clone(), col_name.clone()), ColumnIdx(ci));
                ci += 1;
            }
        }
        map
    };

    // ── Step 6: Build PK maps and collect FK edges ─────────────────────
    // Graph construction is deferred to after task tables are created (step 7b)
    // so that task table rows and their one-directional FK edges can be included.
    info!("Step 6: Building PK maps and collecting FK edges...");

    // 6a) Build PK lookup maps: table_name -> HashMap<i64, global_row_idx>
    let mut pk_maps: HashMap<String, HashMap<i64, u32>> = HashMap::new();
    for (ti, lt) in loaded_tables.iter().enumerate() {
        if let Some(pk_col_name) = &lt.raw_table.primary_key {
            if let Some(col_idx) = lt.schema.index_of(pk_col_name).ok() {
                let arr = lt.batch.column(col_idx);
                if let Some(i64_arr) = cast_to_i64(arr.as_ref()) {
                    let row_start = table_row_ranges[ti].0.0;
                    let mut map = HashMap::with_capacity(i64_arr.len());
                    for i in 0..i64_arr.len() {
                        if !i64_arr.is_null(i) {
                            map.insert(i64_arr.value(i), row_start + i as u32);
                        }
                    }
                    pk_maps.insert(lt.name.clone(), map);
                }
            }
        }
    }

    // 6b) Build per-row temporal z-score lookup for the temporal incoming index.
    let mut row_temporal_valid = vec![0u8; total_rows];
    let mut row_temporal_z = vec![0f32; total_rows];
    for (ti, lt) in loaded_tables.iter().enumerate() {
        let Some(tc_name) = &table_temporal_col_names[ti] else {
            continue;
        };
        let Some(col_idx) = lt.schema.index_of(tc_name).ok() else {
            continue;
        };
        let arr = lt.batch.column(col_idx);
        let Some(ts_arr) = cast_to_timestamp_us(arr.as_ref()) else {
            continue;
        };
        let row_start = table_row_ranges[ti].0.0 as usize;
        for i in 0..ts_arr.len() {
            if ts_arr.is_null(i) {
                continue;
            }
            let global_row = row_start + i;
            row_temporal_valid[global_row] = 1;
            row_temporal_z[global_row] = if global_ts_std_us > 0.0 {
                ((ts_arr.value(i) as f64 - global_ts_mean_us) / global_ts_std_us) as f32
            } else {
                0.0
            };
        }
    }

    // 6c) Collect FK edges from real tables (task edges added in step 7b).
    let mut real_edges: Vec<(u32, u32)> = Vec::new();
    for (ti, lt) in loaded_tables.iter().enumerate() {
        let row_start = table_row_ranges[ti].0.0;

        for (col_name, raw_col) in &lt.raw_table.columns {
            if let Some(fk_target) = &raw_col.foreign_key {
                let parts: Vec<&str> = fk_target.splitn(2, '.').collect();
                if parts.len() == 2 {
                    let target_table = parts[0];
                    if let Some(pk_map) = pk_maps.get(target_table) {
                        if let Some(col_arr_idx) = lt.schema.index_of(col_name).ok() {
                            let arr = lt.batch.column(col_arr_idx);
                            if let Some(i64_arr) = cast_to_i64(arr.as_ref()) {
                                for i in 0..i64_arr.len() {
                                    if !i64_arr.is_null(i) {
                                        let fk_val = i64_arr.value(i);
                                        if let Some(&target_row) = pk_map.get(&fk_val) {
                                            let src_row = row_start + i as u32;
                                            real_edges.push((src_row, target_row));
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        warn!(
                            "  FK target table '{}' not found for {}.{}",
                            target_table, table_names[ti], col_name
                        );
                    }
                }
            }
        }
    }
    info!(
        "  PK maps built, {} real FK edges collected",
        HumanCount(real_edges.len() as u64)
    );

    // ── Step 7: Materialize tasks + create task tables ──────────────────
    info!("Step 7: Materializing tasks and creating task tables...");
    let num_tasks = raw_meta.tasks.len();
    let pb_tasks = ProgressBar::new(num_tasks as u64);
    pb_tasks.set_style(
        ProgressStyle::with_template(
            "  Tasks      {bar:40.cyan/blue} {pos}/{len} tasks [{elapsed_precise}] {msg}",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    let mut task_metadata_list: Vec<TaskMetadata> = Vec::new();
    let mut task_edges: Vec<(u32, u32)> = Vec::new();
    let mut task_col_name_strings: Vec<String> = Vec::new();

    // Build table name -> TableIdx lookup
    let table_name_to_idx: HashMap<String, usize> = table_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    rt.block_on(async {
        let session_config = datafusion::prelude::SessionConfig::new()
            .set_bool("datafusion.sql_parser.enable_ident_normalization", false);
        let ctx = datafusion::prelude::SessionContext::new_with_config(session_config);

        for (table_name, _) in &raw_meta.tables {
            let path = raw_dir.join(format!("{table_name}.parquet"));
            let path_str = path.to_str().unwrap();
            let create_sql = format!(
                "CREATE EXTERNAL TABLE \"{table_name}\" STORED AS PARQUET LOCATION '{path_str}'"
            );
            ctx.sql(&create_sql).await?;
        }

        for (task_name, raw_task) in raw_meta.tasks.iter() {
            pb_tasks.set_message(task_name.clone());

            let mut query = raw_task.query.clone();
            for (tname, _) in &raw_meta.tables {
                let from_pat = format!("'{tname}.parquet'");
                let to_pat = format!("\"{tname}\"");
                query = query.replace(&from_pat, &to_pat);
            }

            let df = ctx.sql(&query).await?;
            let batches = df.collect().await?;

            if batches.is_empty() {
                warn!("    Task {} returned no results, skipping", task_name);
                pb_tasks.inc(1);
                continue;
            }

            let result_schema = batches[0].schema();
            let result_batch = if batches.len() == 1 {
                batches.into_iter().next().unwrap()
            } else {
                concat_batches(&result_schema, &batches)?
            };

            let num_seeds = result_batch.num_rows();
            info!("    {} seeds", num_seeds);

            let anchor_table_idx = *table_name_to_idx
                .get(&raw_task.anchor_table)
                .ok_or_else(|| format!("anchor_table '{}' not found", raw_task.anchor_table))?;
            let anchor_pk_map = pk_maps
                .get(&raw_task.anchor_table)
                .ok_or_else(|| format!("no PK map for anchor_table '{}'", raw_task.anchor_table))?;

            let anchor_col_idx = result_schema.index_of(&raw_task.anchor_key)?;
            let anchor_arr = result_batch.column(anchor_col_idx);
            let anchor_i64 = cast_to_i64(anchor_arr.as_ref()).ok_or_else(|| {
                format!("cannot cast anchor_key '{}' to i64", raw_task.anchor_key)
            })?;

            let mut anchor_rows: Vec<u32> = Vec::with_capacity(num_seeds);
            let mut valid_mask: Vec<bool> = Vec::with_capacity(num_seeds);
            for i in 0..num_seeds {
                if anchor_i64.is_null(i) {
                    anchor_rows.push(0);
                    valid_mask.push(false);
                } else {
                    let pk_val = anchor_i64.value(i);
                    if let Some(&global_row) = anchor_pk_map.get(&pk_val) {
                        anchor_rows.push(global_row);
                        valid_mask.push(true);
                    } else {
                        anchor_rows.push(0);
                        valid_mask.push(false);
                    }
                }
            }

            let valid_indices: Vec<usize> = valid_mask
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| if v { Some(i) } else { None })
                .collect();
            let num_valid = valid_indices.len();
            if num_valid == 0 {
                warn!("    No valid seeds for task {}, skipping", task_name);
                pb_tasks.inc(1);
                continue;
            }
            let anchor_rows_valid: Vec<u32> =
                valid_indices.iter().map(|&i| anchor_rows[i]).collect();

            let observation_times: Vec<i64> = if let Some(obs_col_name) =
                &raw_task.observation_time_column
            {
                let obs_col_idx = result_schema.index_of(obs_col_name)?;
                let obs_arr = result_batch.column(obs_col_idx);
                let obs_ts = cast_to_timestamp_us(obs_arr.as_ref()).ok_or_else(|| {
                    format!(
                        "cannot cast observation_time '{}' to timestamp",
                        obs_col_name
                    )
                })?;
                valid_indices
                    .iter()
                    .map(|&i| {
                        if obs_ts.is_null(i) {
                            i64::MAX
                        } else {
                            obs_ts.value(i)
                        }
                    })
                    .collect()
            } else if let Some(tc_name) = &raw_meta.tables[&raw_task.anchor_table].temporal_column {
                let anchor_lt = loaded_tables
                    .iter()
                    .find(|lt| lt.name == raw_task.anchor_table)
                    .ok_or_else(|| {
                        format!(
                            "anchor table '{}' not found in loaded tables",
                            raw_task.anchor_table
                        )
                    })?;
                let tc_arr_idx = anchor_lt.schema.index_of(tc_name).map_err(|_| {
                    format!(
                        "temporal_column '{}' not found in table '{}'",
                        tc_name, raw_task.anchor_table
                    )
                })?;
                let tc_arr = anchor_lt.batch.column(tc_arr_idx);
                let tc_ts = cast_to_timestamp_us(tc_arr.as_ref()).ok_or_else(|| {
                    format!("cannot cast temporal_column '{}' to timestamp", tc_name)
                })?;
                let anchor_row_start = table_row_ranges[anchor_table_idx].0.0;
                valid_indices
                    .iter()
                    .map(|&i| {
                        let global_row = anchor_rows[i];
                        let local_row = (global_row - anchor_row_start) as usize;
                        if tc_ts.is_null(local_row) {
                            i64::MAX
                        } else {
                            tc_ts.value(local_row)
                        }
                    })
                    .collect()
            } else {
                vec![i64::MAX; num_valid]
            };

            let target_col_idx = result_schema.index_of(&raw_task.target_column)?;
            let target_arr = result_batch.column(target_col_idx);
            let target_stype = parse_stype(&raw_task.target_stype);

            let (target_writer_data, target_stats) = encode_task_target(
                target_stype,
                target_arr.as_ref(),
                &valid_indices,
                &raw_task.target_column,
                &cat_map,
                global_ts_mean_us,
                global_ts_std_us,
            );

            // Write the TaskView binary (anchor_rows, observation_times, target).
            let target_writer = target_writer_data.as_writer();
            let task_path = output_dir.join(format!("tasks/{}.bin", task_name));
            write_task_bin(
                &task_path,
                num_valid as u32,
                &anchor_rows_valid,
                &observation_times,
                &target_writer,
            )?;

            // ── Create the task table as a first-class table in the graph ──
            let has_obs_time = raw_task.observation_time_column.is_some();
            let task_num_cols: u32 = if has_obs_time { 2 } else { 1 };

            let task_row_start = RowIdx(global_row_offset);
            let task_row_end = RowIdx(global_row_offset + num_valid as u32);
            let task_col_start = ColumnIdx(global_col_offset);
            let task_col_end = ColumnIdx(global_col_offset + task_num_cols);

            // Build the task table's column writers.
            // Target is always the first column (position 0) by convention.
            let mut task_table_writers: Vec<ColumnWriterData> = Vec::new();

            task_table_writers.push(target_writer_data);

            let target_col_global = ColumnIdx(global_col_offset);
            all_col_names.push(format!("target_{}", raw_task.target_column));
            all_col_table_names.push(task_name.clone());
            all_col_stypes.push(target_stype);
            all_col_fkey_targets.push(None);
            all_col_descriptions.push(Some(format!(
                "{} prediction target",
                raw_task.target_column
            )));
            all_col_stats.push(target_stats.clone());

            task_col_name_strings.push(format!(
                "target of {}: {} prediction target",
                task_name, raw_task.target_column
            ));

            if has_obs_time {
                let obs_validity =
                    build_bitmap_from_fn(num_valid, |i| observation_times[i] != i64::MAX);
                let mut obs_values = vec![0.0f32; num_valid * TIMESTAMP_DIM];
                for i in 0..num_valid {
                    if observation_times[i] != i64::MAX {
                        let encoded = encode_timestamp(
                            observation_times[i],
                            global_ts_mean_us,
                            global_ts_std_us,
                        );
                        obs_values[i * TIMESTAMP_DIM..(i + 1) * TIMESTAMP_DIM]
                            .copy_from_slice(&encoded);
                    }
                }
                task_table_writers.push(ColumnWriterData::Timestamp {
                    validity: obs_validity,
                    values: obs_values,
                });

                all_col_names.push("observation_time".to_string());
                all_col_table_names.push(task_name.clone());
                all_col_stypes.push(SemanticType::Timestamp);
                all_col_fkey_targets.push(None);
                all_col_descriptions.push(Some(format!(
                    "Temporal cutoff for the {} prediction",
                    task_name
                )));

                let obs_stats = compute_obs_time_stats(&observation_times);
                all_col_stats.push(obs_stats);

                task_col_name_strings.push(format!(
                    "observation_time of {}: Temporal cutoff for the prediction",
                    task_name
                ));
            }

            // Write the task table binary file.
            let writers: Vec<ColumnWriter> =
                task_table_writers.iter().map(|d| d.as_writer()).collect();
            let task_table_path = output_dir.join(format!("tables/{}.bin", task_name));
            write_table_bin(&task_table_path, num_valid as u32, &writers)?;

            // Collect FK edges: task_row -> anchor_row (outgoing only).
            for (i, &anchor_row) in anchor_rows_valid.iter().enumerate() {
                let task_global_row = global_row_offset + i as u32;
                task_edges.push((task_global_row, anchor_row));
            }

            // Record table metadata for this task table.
            let task_table_idx = TableIdx((num_tables + task_metadata_list.len()) as u32);
            table_row_ranges.push((task_row_start, task_row_end));
            table_col_ranges.push((task_col_start, task_col_end));
            table_names.push(task_name.clone());
            table_pkey_col_names.push(None);
            table_temporal_col_names.push(if has_obs_time {
                Some("observation_time".to_string())
            } else {
                None
            });

            // Resolve columns_to_drop from raw (table, column) pairs to global ColumnIdx.
            let columns_to_drop: Vec<ColumnIdx> = raw_task
                .columns_to_drop
                .iter()
                .filter_map(|cr| {
                    col_lookup
                        .get(&(cr.table.clone(), cr.column.clone()))
                        .copied()
                })
                .collect();

            task_metadata_list.push(TaskMetadata {
                name: task_name.clone(),
                anchor_table: TableIdx(anchor_table_idx as u32),
                task_table: task_table_idx,
                target_stype,
                target_col: target_col_global,
                columns_to_drop,
                num_seeds: num_valid as u32,
                target_stats,
            });

            global_row_offset += num_valid as u32;
            global_col_offset += task_num_cols;
            pb_tasks.inc(1);
        }

        Ok::<(), Box<dyn std::error::Error>>(())
    })?;
    pb_tasks.finish_and_clear();
    info!(
        "  Materialized {} tasks ({} task FK edges)",
        task_metadata_list.len(),
        HumanCount(task_edges.len() as u64)
    );

    let total_rows_with_tasks = global_row_offset as usize;
    let total_cols_with_tasks = global_col_offset as usize;

    // ── Step 7b: Build FK graph (real + task edges) ─────────────────────
    info!("Step 7b: Building FK graph...");

    // Extend temporal arrays for task table rows (task rows have no temporal z-score for incoming edges).
    row_temporal_valid.resize(total_rows_with_tasks, 0);
    row_temporal_z.resize(total_rows_with_tasks, 0.0);

    // Build outgoing CSR from ALL edges (real + task).
    let mut all_outgoing_edges = real_edges.clone();
    all_outgoing_edges.extend_from_slice(&task_edges);
    all_outgoing_edges.sort_unstable_by_key(|&(src, _)| src);
    let outgoing = CsrGraph::from_sorted_edges(total_rows_with_tasks, &all_outgoing_edges);

    // Build incoming CSR from REAL edges only (task edges are one-directional outgoing).
    real_edges.sort_unstable_by_key(|&(src, _)| src);
    let real_outgoing_for_incoming =
        CsrGraph::from_sorted_edges(total_rows_with_tasks, &real_edges);
    let incoming = real_outgoing_for_incoming.transpose();

    // Build temporal/static incoming split.
    let mut incoming_static_row_ptr: Vec<u32> = Vec::with_capacity(total_rows_with_tasks + 1);
    let mut incoming_static_col_idx: Vec<u32> = Vec::new();
    let mut incoming_temporal_row_ptr: Vec<u32> = Vec::with_capacity(total_rows_with_tasks + 1);
    let mut incoming_temporal_col_idx: Vec<u32> = Vec::new();
    let mut incoming_temporal_z: Vec<f32> = Vec::new();
    incoming_static_row_ptr.push(0);
    incoming_temporal_row_ptr.push(0);
    for dst in 0..total_rows_with_tasks {
        let mut temporal_pairs: Vec<(f32, u32)> = Vec::new();
        for &src in incoming.neighbors(dst as u32) {
            let src_idx = src as usize;
            if row_temporal_valid[src_idx] == 1 {
                temporal_pairs.push((row_temporal_z[src_idx], src));
            } else {
                incoming_static_col_idx.push(src);
            }
        }
        temporal_pairs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for (z, src) in temporal_pairs {
            incoming_temporal_col_idx.push(src);
            incoming_temporal_z.push(z);
        }
        incoming_static_row_ptr.push(incoming_static_col_idx.len() as u32);
        incoming_temporal_row_ptr.push(incoming_temporal_col_idx.len() as u32);
    }
    let graph_path = output_dir.join("graph.bin");
    write_graph_bin(
        &outgoing,
        &incoming,
        &incoming_static_row_ptr,
        &incoming_static_col_idx,
        &incoming_temporal_row_ptr,
        &incoming_temporal_col_idx,
        &incoming_temporal_z,
        &graph_path,
    )?;
    info!(
        "  Wrote {} ({} nodes, {} edges, {} static incoming, {} temporal incoming)",
        graph_path.display(),
        outgoing.num_nodes(),
        outgoing.num_edges(),
        incoming_static_col_idx.len(),
        incoming_temporal_col_idx.len()
    );

    // ── Step 8: Build and write metadata.json + embedding .bin files ─────
    info!("Step 8: Writing metadata and embeddings...");

    let column_metadata: Vec<ColumnMetadata> = (0..total_cols_with_tasks)
        .map(|i| {
            let fkey_target_col = all_col_fkey_targets[i].as_ref().and_then(|fk| {
                let parts: Vec<&str> = fk.splitn(2, '.').collect();
                if parts.len() == 2 {
                    col_lookup
                        .get(&(parts[0].to_string(), parts[1].to_string()))
                        .copied()
                } else {
                    None
                }
            });
            ColumnMetadata {
                name: all_col_names[i].clone(),
                stype: all_col_stypes[i],
                fkey_target_col,
                embedding: Vec::new(), // populated at load time from column_embeddings.bin
            }
        })
        .collect();

    // 8b) Build TableMetadata (real tables + task tables)
    let total_table_count = table_names.len();
    let table_metadata: Vec<TableMetadata> = (0..total_table_count)
        .map(|ti| {
            let pkey_col = table_pkey_col_names[ti].as_ref().and_then(|pk_name| {
                col_lookup
                    .get(&(table_names[ti].clone(), pk_name.clone()))
                    .copied()
            });
            let temporal_col = table_temporal_col_names[ti].as_ref().and_then(|tc_name| {
                // For task tables, temporal_col uses the task table's own column index.
                // The col_lookup only has real table columns, so we look up directly.
                if ti < num_tables {
                    col_lookup
                        .get(&(table_names[ti].clone(), tc_name.clone()))
                        .copied()
                } else {
                    Some(table_col_ranges[ti].0)
                }
            });
            TableMetadata {
                name: table_names[ti].clone(),
                col_range: table_col_ranges[ti],
                row_range: table_row_ranges[ti],
                pkey_col,
                temporal_col,
            }
        })
        .collect();

    let db_metadata = DatabaseMetadata {
        table_metadata,
        column_metadata,
        column_stats: all_col_stats,
        task_metadata: task_metadata_list,
        global_ts_mean_us,
        global_ts_std_us,
    };

    // Write metadata.json
    let metadata_json = serde_json::to_string_pretty(&db_metadata)?;
    fs::write(output_dir.join("metadata.json"), &metadata_json)?;
    info!("  Wrote metadata.json ({} bytes)", metadata_json.len());

    // Compute embeddings for task table column names.
    let task_col_embeddings: Vec<Vec<f16>> = if task_col_name_strings.is_empty() {
        Vec::new()
    } else if args.skip_embeddings {
        vec![vec![f16::ZERO; EMBEDDING_DIM]; task_col_name_strings.len()]
    } else {
        let config = EmbedderConfig::from_baseten_env()?;
        let embedder = Embedder::new(config)?;
        let refs: Vec<&str> = task_col_name_strings.iter().map(|s| s.as_str()).collect();
        info!(
            "  Embedding {} task column names...",
            task_col_name_strings.len()
        );
        let flat = rt.block_on(async { embedder.embed_texts(&refs).await })?;
        flat.chunks_exact(EMBEDDING_DIM)
            .map(|c| c.to_vec())
            .collect()
    };

    // Write column_embeddings.bin — real column embeddings + task column embeddings.
    let total_col_emb_count = column_embeddings.len() + task_col_embeddings.len();
    let mut col_emb_data: Vec<u8> = Vec::with_capacity(total_col_emb_count * EMBEDDING_DIM * 2);
    for emb in column_embeddings.iter().chain(task_col_embeddings.iter()) {
        for &v in emb {
            col_emb_data.extend_from_slice(&f16::to_ne_bytes(v));
        }
    }
    fs::write(output_dir.join("column_embeddings.bin"), &col_emb_data)?;
    info!(
        "  Wrote column_embeddings.bin ({} column embeddings, {} bytes)",
        total_col_emb_count,
        col_emb_data.len()
    );

    // Write categorical_embeddings.bin — categorical value embeddings (GPU-resident).
    let mut cat_emb_data: Vec<u8> =
        Vec::with_capacity(categorical_embeddings.len() * EMBEDDING_DIM * 2);
    for emb in categorical_embeddings.iter() {
        for &v in emb {
            cat_emb_data.extend_from_slice(&f16::to_ne_bytes(v));
        }
    }
    fs::write(output_dir.join("categorical_embeddings.bin"), &cat_emb_data)?;
    info!(
        "  Wrote categorical_embeddings.bin ({} categorical embeddings, {} bytes)",
        categorical_embeddings.len(),
        cat_emb_data.len()
    );

    // Write text_embeddings.bin — text value embeddings (per-batch subsets shipped to GPU).
    let mut text_emb_data: Vec<u8> = Vec::with_capacity(text_embeddings.len() * EMBEDDING_DIM * 2);
    for emb in text_embeddings.iter() {
        for &v in emb {
            text_emb_data.extend_from_slice(&f16::to_ne_bytes(v));
        }
    }
    fs::write(output_dir.join("text_embeddings.bin"), &text_emb_data)?;
    info!(
        "  Wrote text_embeddings.bin ({} text embeddings, {} bytes)",
        text_embeddings.len(),
        text_emb_data.len()
    );

    let elapsed = pipeline_start.elapsed();
    info!("Preprocessing complete in {}!", HumanDuration(elapsed));
    info!("  Output directory: {}", output_dir.display());
    info!(
        "  Tables: {} real + {} task = {}",
        loaded_tables.len(),
        db_metadata.task_metadata.len(),
        total_table_count
    );
    info!("  Tasks: {}", db_metadata.task_metadata.len());
    info!(
        "  Total rows: {} (real) + {} (task) = {}",
        HumanCount(total_rows as u64),
        HumanCount((total_rows_with_tasks - total_rows) as u64),
        HumanCount(total_rows_with_tasks as u64)
    );
    info!("  Total columns: {}", total_cols_with_tasks);
    info!(
        "  Categorical vocab: {}, Text vocab: {}",
        HumanCount(cat_list.len() as u64),
        HumanCount(text_list.len() as u64)
    );

    Ok(())
}

// ============================================================================
// Column encoding helpers
// ============================================================================

/// Owned data for a single encoded column, bridging owned Vecs to borrowed ColumnWriter slices.
enum ColumnWriterData {
    Identifier {
        bits: Vec<u8>,
    },
    Numerical {
        validity: Vec<u8>,
        values: Vec<f32>,
    },
    Timestamp {
        validity: Vec<u8>,
        values: Vec<f32>,
    },
    Boolean {
        validity: Vec<u8>,
        bits: Vec<u8>,
    },
    Embedded {
        validity: Vec<u8>,
        indices: Vec<u32>,
    },
    Ignored,
}

impl ColumnWriterData {
    fn as_writer(&self) -> ColumnWriter<'_> {
        match self {
            ColumnWriterData::Identifier { bits } => ColumnWriter::Identifier { bits },
            ColumnWriterData::Numerical { validity, values } => {
                ColumnWriter::Numerical { validity, values }
            }
            ColumnWriterData::Timestamp { validity, values } => {
                ColumnWriter::Timestamp { validity, values }
            }
            ColumnWriterData::Boolean { validity, bits } => {
                ColumnWriter::Boolean { validity, bits }
            }
            ColumnWriterData::Embedded { validity, indices } => {
                ColumnWriter::Embedded { validity, indices }
            }
            ColumnWriterData::Ignored => ColumnWriter::Ignored,
        }
    }
}

fn encode_column(
    stype: SemanticType,
    stats: &ColumnStats,
    arrow_col: Option<&dyn Array>,
    num_rows: usize,
    col_name: &str,
    cat_map: &HashMap<String, CategoricalEmbeddingIdx>,
    text_map: &HashMap<String, TextEmbeddingIdx>,
    global_ts_mean_us: f64,
    global_ts_std_us: f64,
) -> ColumnWriterData {
    match stype {
        SemanticType::Ignored => ColumnWriterData::Ignored,

        SemanticType::Identifier => {
            if let Some(arr) = arrow_col {
                let bits = build_validity_bitmap(arr);
                ColumnWriterData::Identifier { bits }
            } else {
                ColumnWriterData::Identifier {
                    bits: vec![0u8; packed_bit_bytes(num_rows)],
                }
            }
        }

        SemanticType::Numerical => {
            let (mean, std) = match stats {
                ColumnStats::Numerical { mean, std, .. } => (*mean, *std),
                _ => (0.0, 1.0),
            };
            if let Some(arr) = arrow_col {
                let validity = build_validity_bitmap(arr);
                let f64_arr = cast_to_f64(arr);
                let mut values = vec![0.0f32; num_rows];
                if let Some(fa) = &f64_arr {
                    for i in 0..num_rows {
                        if !fa.is_null(i) {
                            let z = if std > 0.0 {
                                (fa.value(i) - mean) / std
                            } else {
                                0.0
                            };
                            values[i] = z as f32;
                        }
                    }
                }
                ColumnWriterData::Numerical { validity, values }
            } else {
                ColumnWriterData::Numerical {
                    validity: vec![0u8; packed_bit_bytes(num_rows)],
                    values: vec![0.0f32; num_rows],
                }
            }
        }

        SemanticType::Timestamp => {
            if let Some(arr) = arrow_col {
                let validity = build_validity_bitmap(arr);
                let ts_arr = cast_to_timestamp_us(arr);
                let mut values = vec![0.0f32; num_rows * TIMESTAMP_DIM];
                if let Some(ta) = &ts_arr {
                    for i in 0..num_rows {
                        if !ta.is_null(i) {
                            let encoded =
                                encode_timestamp(ta.value(i), global_ts_mean_us, global_ts_std_us);
                            values[i * TIMESTAMP_DIM..(i + 1) * TIMESTAMP_DIM]
                                .copy_from_slice(&encoded);
                        }
                    }
                }
                ColumnWriterData::Timestamp { validity, values }
            } else {
                ColumnWriterData::Timestamp {
                    validity: vec![0u8; packed_bit_bytes(num_rows)],
                    values: vec![0.0f32; num_rows * TIMESTAMP_DIM],
                }
            }
        }

        SemanticType::Boolean => {
            if let Some(arr) = arrow_col {
                let validity = build_validity_bitmap(arr);
                let ba = arr
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .cloned()
                    .or_else(|| {
                        arrow::compute::cast(arr, &DataType::Boolean)
                            .ok()
                            .and_then(|a| a.as_any().downcast_ref::<BooleanArray>().cloned())
                    });
                let bits = if let Some(ba) = &ba {
                    build_bitmap_from_fn(num_rows, |i| !ba.is_null(i) && ba.value(i))
                } else {
                    vec![0u8; packed_bit_bytes(num_rows)]
                };
                ColumnWriterData::Boolean { validity, bits }
            } else {
                let byte_len = packed_bit_bytes(num_rows);
                ColumnWriterData::Boolean {
                    validity: vec![0u8; byte_len],
                    bits: vec![0u8; byte_len],
                }
            }
        }

        SemanticType::Categorical => {
            if let Some(arr) = arrow_col {
                let validity = build_validity_bitmap(arr);
                let strings = array_to_strings(arr);
                let mut indices = vec![0u32; num_rows];
                for i in 0..num_rows {
                    if let Some(ref v) = strings[i] {
                        let key = format!("{col_name} is {v}");
                        if let Some(&emb_idx) = cat_map.get(&key) {
                            indices[i] = emb_idx.0;
                        }
                    }
                }
                ColumnWriterData::Embedded { validity, indices }
            } else {
                ColumnWriterData::Embedded {
                    validity: vec![0u8; packed_bit_bytes(num_rows)],
                    indices: vec![0u32; num_rows],
                }
            }
        }

        SemanticType::Text => {
            if let Some(arr) = arrow_col {
                let validity = build_validity_bitmap(arr);
                let strings = array_to_strings(arr);
                let mut indices = vec![0u32; num_rows];
                for i in 0..num_rows {
                    if let Some(ref v) = strings[i] {
                        if let Some(&emb_idx) = text_map.get(v) {
                            indices[i] = emb_idx.0;
                        }
                    }
                }
                ColumnWriterData::Embedded { validity, indices }
            } else {
                ColumnWriterData::Embedded {
                    validity: vec![0u8; packed_bit_bytes(num_rows)],
                    indices: vec![0u32; num_rows],
                }
            }
        }
    }
}

/// Encode the target column for a task, returning the ColumnWriterData and ColumnStats.
/// `valid_indices` filters which rows from the full result to include.
fn encode_task_target(
    target_stype: SemanticType,
    target_arr: &dyn Array,
    valid_indices: &[usize],
    target_col_name: &str,
    cat_map: &HashMap<String, CategoricalEmbeddingIdx>,
    global_ts_mean_us: f64,
    global_ts_std_us: f64,
) -> (ColumnWriterData, ColumnStats) {
    let n = valid_indices.len();

    match target_stype {
        SemanticType::Numerical => {
            let f64_arr = cast_to_f64(target_arr);
            let mut num_nulls = 0u64;
            let mut sum = 0.0f64;
            let mut count = 0u64;
            let mut min_val = f64::MAX;
            let mut max_val = f64::MIN;

            // First pass: compute stats
            if let Some(ref fa) = f64_arr {
                for &i in valid_indices {
                    if fa.is_null(i) {
                        num_nulls += 1;
                    } else {
                        let v = fa.value(i);
                        sum += v;
                        count += 1;
                        if v < min_val {
                            min_val = v;
                        }
                        if v > max_val {
                            max_val = v;
                        }
                    }
                }
            }
            let mean = if count > 0 { sum / count as f64 } else { 0.0 };
            let mut var_sum = 0.0f64;
            if let Some(ref fa) = f64_arr {
                for &i in valid_indices {
                    if !fa.is_null(i) {
                        let d = fa.value(i) - mean;
                        var_sum += d * d;
                    }
                }
            }
            let std = if count > 1 {
                (var_sum / (count - 1) as f64).sqrt()
            } else {
                1.0
            };

            // Second pass: encode
            let validity = build_bitmap_from_fn(n, |idx| {
                f64_arr
                    .as_ref()
                    .map(|fa| !fa.is_null(valid_indices[idx]))
                    .unwrap_or(false)
            });
            let mut values = vec![0.0f32; n];
            if let Some(ref fa) = f64_arr {
                for (out_idx, &i) in valid_indices.iter().enumerate() {
                    if !fa.is_null(i) {
                        let z = if std > 0.0 {
                            (fa.value(i) - mean) / std
                        } else {
                            0.0
                        };
                        values[out_idx] = z as f32;
                    }
                }
            }

            let stats = ColumnStats::Numerical {
                num_nulls,
                min: if count > 0 { min_val } else { 0.0 },
                max: if count > 0 { max_val } else { 0.0 },
                mean,
                std,
            };
            (ColumnWriterData::Numerical { validity, values }, stats)
        }

        SemanticType::Timestamp => {
            let ts_arr = cast_to_timestamp_us(target_arr);
            let mut num_nulls = 0u64;
            let mut sum = 0.0f64;
            let mut count = 0u64;
            let mut min_val = i64::MAX;
            let mut max_val = i64::MIN;

            if let Some(ref ta) = ts_arr {
                for &i in valid_indices {
                    if ta.is_null(i) {
                        num_nulls += 1;
                    } else {
                        let v = ta.value(i);
                        sum += v as f64;
                        count += 1;
                        if v < min_val {
                            min_val = v;
                        }
                        if v > max_val {
                            max_val = v;
                        }
                    }
                }
            }
            let mean_us = if count > 0 { sum / count as f64 } else { 0.0 };
            let mut var_sum = 0.0f64;
            if let Some(ref ta) = ts_arr {
                for &i in valid_indices {
                    if !ta.is_null(i) {
                        let d = ta.value(i) as f64 - mean_us;
                        var_sum += d * d;
                    }
                }
            }
            let std_us = if count > 1 {
                (var_sum / (count - 1) as f64).sqrt()
            } else {
                1.0
            };

            let validity = build_bitmap_from_fn(n, |idx| {
                ts_arr
                    .as_ref()
                    .map(|ta| !ta.is_null(valid_indices[idx]))
                    .unwrap_or(false)
            });
            let mut values = vec![0.0f32; n * TIMESTAMP_DIM];
            if let Some(ref ta) = ts_arr {
                for (out_idx, &i) in valid_indices.iter().enumerate() {
                    if !ta.is_null(i) {
                        let encoded =
                            encode_timestamp(ta.value(i), global_ts_mean_us, global_ts_std_us);
                        values[out_idx * TIMESTAMP_DIM..(out_idx + 1) * TIMESTAMP_DIM]
                            .copy_from_slice(&encoded);
                    }
                }
            }

            let stats = ColumnStats::Timestamp {
                num_nulls,
                min_us: if count > 0 { min_val } else { 0 },
                max_us: if count > 0 { max_val } else { 0 },
                mean_us,
                std_us,
            };
            (ColumnWriterData::Timestamp { validity, values }, stats)
        }

        SemanticType::Boolean => {
            let ba = target_arr
                .as_any()
                .downcast_ref::<BooleanArray>()
                .cloned()
                .or_else(|| {
                    arrow::compute::cast(target_arr, &DataType::Boolean)
                        .ok()
                        .and_then(|a| a.as_any().downcast_ref::<BooleanArray>().cloned())
                });

            let mut num_nulls = 0u64;
            let mut num_true = 0u64;
            let mut num_false = 0u64;

            if let Some(ref ba) = ba {
                for &i in valid_indices {
                    if ba.is_null(i) {
                        num_nulls += 1;
                    } else if ba.value(i) {
                        num_true += 1;
                    } else {
                        num_false += 1;
                    }
                }
            }

            let validity = build_bitmap_from_fn(n, |idx| {
                ba.as_ref()
                    .map(|b| !b.is_null(valid_indices[idx]))
                    .unwrap_or(false)
            });
            let bits = build_bitmap_from_fn(n, |idx| {
                ba.as_ref()
                    .map(|b| !b.is_null(valid_indices[idx]) && b.value(valid_indices[idx]))
                    .unwrap_or(false)
            });

            let stats = ColumnStats::Boolean {
                num_nulls,
                num_true,
                num_false,
            };
            (ColumnWriterData::Boolean { validity, bits }, stats)
        }

        SemanticType::Categorical => {
            let strings = array_to_strings(target_arr);
            let mut num_nulls = 0u64;
            let mut cats_set: HashSet<String> = HashSet::new();

            for &i in valid_indices {
                if let Some(ref v) = strings[i] {
                    cats_set.insert(v.clone());
                } else {
                    num_nulls += 1;
                }
            }

            let mut categories: Vec<String> = cats_set.into_iter().collect();
            categories.sort();

            // Determine cat_emb_start: look up the first category's index in cat_map.
            // All categories for this task target should already be in cat_map from the
            // source column (task targets are subsets of existing column values).
            let cat_emb_start = if let Some(first_cat) = categories.first() {
                let key = format!("{target_col_name} is {first_cat}");
                cat_map.get(&key).map(|idx| idx.0).unwrap_or(0)
            } else {
                0
            };

            let validity = build_bitmap_from_fn(n, |idx| strings[valid_indices[idx]].is_some());
            let mut indices = vec![0u32; n];
            for (out_idx, &i) in valid_indices.iter().enumerate() {
                if let Some(ref v) = strings[i] {
                    let key = format!("{target_col_name} is {v}");
                    if let Some(&emb_idx) = cat_map.get(&key) {
                        indices[out_idx] = emb_idx.0;
                    }
                }
            }

            let stats = ColumnStats::Categorical {
                num_nulls,
                categories,
                cat_emb_start,
            };
            (ColumnWriterData::Embedded { validity, indices }, stats)
        }

        _ => panic!("invalid target_stype {:?} for task target", target_stype),
    }
}

/// Compute `ColumnStats::Timestamp` for an observation time array (epoch microseconds).
/// Values of `i64::MAX` are treated as null.
fn compute_obs_time_stats(observation_times: &[i64]) -> ColumnStats {
    let n = observation_times.len();
    let mut num_nulls = 0u64;
    let mut sum = 0.0f64;
    let mut count = 0u64;
    let mut min_val = i64::MAX;
    let mut max_val = i64::MIN;

    for &t in observation_times {
        if t == i64::MAX {
            num_nulls += 1;
        } else {
            sum += t as f64;
            count += 1;
            min_val = min_val.min(t);
            max_val = max_val.max(t);
        }
    }

    let mean_us = if count > 0 { sum / count as f64 } else { 0.0 };
    let mut var_sum = 0.0f64;
    for &t in observation_times {
        if t != i64::MAX {
            let d = t as f64 - mean_us;
            var_sum += d * d;
        }
    }
    let std_us = if count > 1 {
        (var_sum / (count - 1) as f64).sqrt()
    } else {
        1.0
    };

    ColumnStats::Timestamp {
        num_nulls,
        min_us: if n > 0 && count > 0 { min_val } else { 0 },
        max_us: if n > 0 && count > 0 { max_val } else { 0 },
        mean_us,
        std_us,
    }
}
