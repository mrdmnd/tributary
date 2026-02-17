# Semantic Types

Database columns have **data types** (the primitive Arrow type a column is stored as) and **semantic types** (the
*meaning* of the data). Semantic types drive how values are normalized, encoded, and stored in the preprocessed binary
format. They also determine what kind of prediction head our model uses for a given task target, and how we compute
loss.

Semantic types are defined in `headwater/src/common.rs` as the `SemanticType` enum (represented as a `u8`).

## Overview

| Semantic Type   | Enum Value | Nullable | Storage Format                     | Normalization / Encoding                          |
|-----------------|:----------:|:--------:|------------------------------------|---------------------------------------------------|
| **Identifier**  | 0          | No       | Packed presence bits               | Presence/absence only (no value stored)           |
| **Numerical**   | 1          | Yes      | Validity bitmap + `[f32]`          | Z-score normalized (`(x - mean) / std`)           |
| **Timestamp**   | 2          | Yes      | Validity bitmap + `[f32; 15/row]`  | 7 cyclic sin/cos pairs + z-scored epoch           |
| **Boolean**     | 3          | Yes      | Validity bitmap + packed bits      | Map to 0 / 1                                      |
| **Categorical** | 4          | Yes      | Validity bitmap + `[u32]`          | Text-embedded (`"column is VALUE"`)               |
| **Text**        | 5          | Yes      | Validity bitmap + `[u32]`          | Text-embedded (`"some document"`)                 |
| **Ignored**     | 6          | N/A      | Nothing stored                     | Column is skipped entirely                        |

## Type Descriptions

### Identifier (`0`)

Identifiers uniquely (or nearly uniquely) distinguish a row. 
Their purpose is entity identity, not attribute description. Examples: `customer_id`, `email`, `display_name`.

- Integer columns named `*_id` or `*Id` are almost always identifiers.
- String columns that are labels/names (not prose) are often identifiers.
- Phone numbers, email addresses, and URLs are typically identifiers.

Identifiers are **not** embedded with the text embedding model. They are included only at the presence/absence
level: if the source value is `NULL`, the model sees absence; otherwise it sees presence. This is stored as a
single packed bitmap on disk (one bit per row) with no validity layer — identifiers are never null in the preprocessed
format.

### Numerical (`1`)

Numeric values where ordering and magnitude carry meaning. Examples: `age`, `price`, `score`, `quantity`.

**Not** for integer columns that happen to be IDs or codes — those should be identifiers or categoricals.

**Normalization:** Per-column z-score normalization: `(x - mean) / std`. The mean and standard deviation are
computed during preprocessing over non-null values and stored in `ColumnStats::Numerical`. Encoded values are
stored as `f32` z-scores.

**Null handling:** A validity bitmap tracks which rows have non-null values. At training time, when the model receives
a numerical cell, it also receives the validity map so that it can learn signals from the presence/absence of data.

### Timestamp (`2`)

Values that represent an instant in time. Examples: `created_at`, `birth_date`, `transaction_time`.
Arrow `timestamp` and `date` types are always assigned this semantic type.

**Normalization:** Timestamps get a rich 15-dimensional float encoding per cell:

1. **7 cyclic sin/cos pairs** (14 floats) for multi-scale periodicity. The values range between [-1, 1].
   - Second of minute (period 60)
   - Minute of hour (period 60)
   - Hour of day (period 24)
   - Day of week (period 7)
   - Day of month (period 31)
   - Month of year (period 12)
   - Day of year (period 366)
2. **1 z-scored epoch value** (float) — the raw epoch microsecond value normalized using *global* timestamp
   statistics (`global_ts_mean_us`, `global_ts_std_us` from `DatabaseMetadata`), so all timestamps across all
   tables in the database are on the same scale.


Statistics per column (`ColumnStats::Timestamp`) store `min_us`, `max_us`, `mean_us`, and `std_us` in epoch
microseconds (`i64` / `f64`).

**Null handling:** A validity bitmap tracks which rows have non-null values. At training time, when the model receives
a timestamp cell, it also receives the validity map so that it can learn signals from the presence/absence of data.

### Boolean (`3`)

True/false values. Examples: `is_active`, `has_discount`.

**Normalization:** Each boolean maps to 0 or 1, stored as packed bits. A separate validity bitmap tracks nulls.

Statistics (`ColumnStats::Boolean`) track `num_nulls`, `num_true`, and `num_false`.

**Null handling:** A validity bitmap tracks which rows have non-null values. At training time, when the model receives
a boolean cell, it also receives the validity map so that it can learn signals from the presence/absence of data.

### Categorical (`4`)

Values drawn from a small-to-medium fixed vocabulary. Examples: `product_category`, `status`, `country_code`.

Key signal: low cardinality (typically < 100 distinct values relative to table size). Integer-flavored columns
with a small number of distinct values are often categorical, not identifiers.

**Encoding:** Each distinct category value is converted to a text string (e.g., `"category is ELECTRONICS"`)
and embedded with a frozen text embedding model (1024-dim output, MRL-truncated to 256-dim, stored as FP16).
The embedding is stored in a dedicated categorical embedding table (`categorical_embeddings.bin`), separate from
text embeddings. Each column's categories occupy a contiguous block in this table. The cell stores a `u32`
`CategoricalEmbeddingIdx` into the categorical embedding table.

Statistics (`ColumnStats::Categorical`) store `num_nulls`, the full list of `categories` (distinct string
values), and `cat_emb_start` (the starting index of this column's block in the categorical embedding table).
Cardinality is `categories.len()`.

**Null handling:** A validity bitmap tracks which rows have non-null values. At training time, when the model receives
a categorical cell, it also receives the validity map so that it can learn signals from the presence/absence of data.

### Text (`5`)

Multi-token strings where the semantic content is important. Examples: product descriptions, reviews, article
bodies, bios.

**Encoding:** Each cell value is embedded with the same frozen text embedding model used for categoricals,
producing a 256-dimensional FP16 vector (MRL-truncated from 1024-dim). Long strings are truncated to 2048
characters before tokenization. Text values are stored in a dedicated text embedding table
(`text_embeddings.bin`), separate from categorical embeddings. The cell stores a `u32` `TextEmbeddingIdx`
into the text embedding table.

The difference from categoricals: text values are typically unique per row (high cardinality), while categoricals
repeat. Both use the same on-disk storage format (`ColumnSlice::Embedded`) but index different embedding tables.

Statistics (`ColumnStats::Text`) track only `num_nulls`.

### Ignored (`6`)

The column is deliberately excluded from processing. No data is stored or encoded. Use for binary blobs, audit
fields, data artifacts, or other columns with no useful signal.

`ColumnStats::Ignored` carries no statistics.

## Assignment

Semantic types are assigned during preprocessing based on a human-annotated `{database}.json` file. The
`generate_metadata.py` script produces a first-draft annotation using an LLM agent that introspects the parquet
files (schemas, cardinalities, value distributions), but a human reviewer makes the final call.

The metadata JSON uses lowercase string names (`"identifier"`, `"numerical"`, `"timestamp"`, `"boolean"`,
`"categorical"`, `"text"`, `"ignored"`) that the preprocessor parses via `parse_stype()`. Unknown strings
default to `Ignored` with a warning.

Most of the time the semantic type can be inferred from the underlying Arrow data type:
- Arrow `timestamp` / `date` types → `Timestamp`
- Arrow `bool` → `Boolean`
- Arrow numeric types → `Numerical` (unless they're IDs or codes)

The metadata file captures information for cases where the data type alone is insufficient — for example,
marking an `int64` column as an `Identifier` or a low-cardinality `string` column as `Categorical` rather than
`Text`.

## Embedding Layout

The preprocessor produces three separate stores of embeddings:

### Column-Name Embeddings (in `column_embeddings.bin`)

A flat `[C, EMBEDDING_DIM]` array of FP16 vectors, one per global column. The embedded text is
`"<col_name> of <table_name>: <description>"` (or `"<col_name> of <table_name>"` if no description is
provided in metadata).

At load time, these are read into `ColumnMetadata.embedding` (a `Vec<f16>`). At training time, they are
collected into a `[num_columns, EMBEDDING_DIM]` tensor and uploaded to GPU shared memory once at the start
of the run.

### Categorical Embeddings (in `categorical_embeddings.bin`)

A flat `[Vc, EMBEDDING_DIM]` array of FP16 vectors containing all categorical value embeddings. Each
column's categories occupy a contiguous block, starting at the offset stored in
`ColumnStats::Categorical::cat_emb_start`. Indices are 0-based.

Because categorical columns are low-cardinality, the total table is small (typically a few thousand entries,
well under 20 MB). At training time, the entire table is uploaded to GPU shared memory once — like column-name
embeddings — and kept resident for the duration of the run. `CategoricalEmbeddingIdx` values in Categorical
columns index directly into this table with no per-batch remapping.

### Text Embeddings (in `text_embeddings.bin`)

A flat `[Vt, EMBEDDING_DIM]` array of FP16 vectors containing all text value embeddings. Each unique text
string value gets one embedding, deduplicated globally. Indices are 0-based.

Text embeddings are high-cardinality (often unique per row) and too large to keep GPU-resident. At training
time, the sampler ships a per-batch subset to the GPU. `TextEmbeddingIdx` values in Text columns are remapped
to batch-local indices during batch construction.

## Usage in Tasks

Prediction tasks (`TaskMetadata`) declare a `target_stype` that determines the prediction head and loss function.
Only `Numerical`, `Categorical`, `Boolean`, and `Timestamp` are valid target types — `Identifier`, `Text`, and
`Ignored` cannot be prediction targets.

The materialized task binary (`<task_name>.bin`) stores target values using the same encoding as regular columns,
with its own `ColumnStats` for normalization and loss computation.
