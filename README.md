# Tributary

A foundation model for learning internal representations of relational databases and making predictions about them.

## Overview

Tributary is HEAVILY based on top of Rishabh Ranjan's work in [Relational Transformer](https://arxiv.org/abs/2510.06377).
The core ideas in this project come predominantly from their work

This is mostly a small re-implementation plus some slight engineer-y modifications.

Tributary is a system designed to train a foundation model to understand relational databases, via prediction.
The underlying model derives signal from human-annotated "tasks", which correspond to regression or classification
problems. As the model learns, it should get better at these tasks across databases.

## Setup

To preprocess your own databases, you need to have a frozen text embedder endpoint available.
The default in this project is to use an OpenAI compatible endpoint; however you want to host this is up to you.
I've found that Baseten is a really good high speed host for throughput-sensitive applications like this preprocessing.

If you want to use a Baseten hosted endpoint, you need to set the following environment variables:

- `BASETEN_EMBEDDER_URL` - the base URL of the Baseten endpoint
- `BASETEN_API_KEY` - the API key for the Baseten endpoint

In additional to generating frozen embeddings, we also will need to provide generated metadata for any given database.
I've put together a small agentic helper script that will generate an appropriate metadata file as a starting point.

If you want to use the OpenRouter endpoint for the "metadata generation agent", you need to set the following variable:

- `OPENROUTER_API_KEY` - the base URL of the OpenRouter endpoint

## Representations

Let's first define the objects we're interested in studying.

First, the relational database.

A relational database is a collection of tables, some of which may be "joinable" to each
other through a primary key/foreign key relationship. We can represent this as a schema
graph, with directed edges between tables.

Each table is also a collection of rows, each of which has some columns. A particular
"cell" is identified by a (table, row, column) tuple. Cells can be "null" or "not null".
(The "null" vs "not null" distinction is not directly supported in the Ranjan work, but
we add it as learning signal here as a useful inductive bias.)

Tables generally will have a primary key column (though this is not required).

Tables may have foreign key columns (relationships to other tables) as well as temporal columns (columns that contain
information points in time). Tables with temporal columns may additionally have a canonical column defined in metadata
that indicates when the observations in the row "came into existence" for the purposes of temporal filtering.

Columns in a table can have different "data types" - these are the primitive types that the column is stored as.
For example, in MySQL, this could be "INT", "FLOAT", "VARCHAR", "BOOLEAN", etc.

However, columns ALSO have meaningful "semantic types" - these encode the meaning of the
column in the context of what it's attempting to represent. For example, a column that
holds a primary key might be an "Identifier" column, whereas a column that holds a date
might be a "Timestamp" column, and a column that holds a string might be a "Text" column
(when the contents are semantically meaningful) or a "Categorical" column (when the
contents do not have meaningful semantics, but serve as an enumeration of some kind.)

The original Ranjan paper did not support the "categorical" semantic type, but we've added
it here as a useful learning target. Open question: would it be better to support booleans
as "categoricals" as well?

Additionally, the Ranjan paper did not support the "Identifier" semantic type - we add
it because the null/not-null-ness of an identifier cell may still be a useful learning
signal to the model.

Our model supports the following semantic types:

- `Identifier`
- `Numerical`
- `Timestamp`
- `Boolean`
- `Categorical`
- `Text`

Every column that is not one of the above semantic types is marked as "Ignored"
and is skipped entirely during preprocessing and training. Users can also manually mark
columns as "Ignored" when they want to skip them entirely from signal inclusion.

## Preprocessing

Assumption: the database comes to us in the form of a collection of parquet files (one per table), and a
special metadata file that has been human-annotated with some information about
the schema. We need information on the semantic types (see above) for each column in the
database, as well as some information about the signal columns that are worth masking
and predicting.

There is a bit of work done here at preprocessing time - we construct a graph
representation of the rows in the database (bidirectional CSR graph), and we also
encode each semantic type cell into a special value.

Identifiers are not encoded at all - merely represented with a validity bitmap indicating "present" / "absent".
The inductive bias here is that there might be some signal in the presence / absence of an identifier, but no signal
from the identifier itself.

Numerical values are encoded as z-scored f32 values, with a validity bitmap (null / present).
The scores are normalized _per-column_ for numerical values.

Timestamp values are cyclically encoded, with a validity bitmap (null / present).
The intention of this inductive bias is that many signals in the world are periodic (holidays, sales, etc), and this
mapping may help the model internalize those patterns better.

- second of minute
- minute of hour
- hour of day
- day of week
- day of month
- month of year
- day of year

These features are turned into pairs (sin(2 _pi_ x / period), cos(2 _pi_ x / period)) to normalize the values.

The timestamp itself (i64 microseconds since epoch) is also part of the feature, but it's
z-score normalized to an f32 value based on all timestamp values across all tables in the database.

Boolean values are encoded as 0 / 1 values, with a validity bitmap (null / present).

Categorical values are encoded as an _index_ into a categorical embedding table, for the string
"column name is X". For example, if the column name is "color", and the value is "red",
we use a frozen text embedder to embed the string literal "color is red", put that embedding into a dedicated
categorical embedding table (`categorical_embeddings.bin`), and store the index into that table.
The categorical table is usually small (low cardinality) and kept GPU-resident at training time.

Text values are similarly encoded — we use the same frozen text embedder for non-identifier
(semantically meaningful) text values, stored in a separate text embedding table
(`text_embeddings.bin`). Text embeddings are usually high-cardinality, and thus cannot live on GPU all the time.
For each batch of sampled trajectories, we identify the complete set of unique text embeddings we need, and ship one
big embeddings tensor to the GPU, along with the indices in the batch.

## Training

We train the model using a masked language model (MLM) objective against a set of "prediction task tables".

For predicting numeric and timestamp values, we use Huber regression loss.
For predicting boolean values, we use binary cross-entropy loss.
For predicting categorical values, we use cross-entropy loss with z-loss regularization

## Building

```bash
cd headwater
cargo build --release
```

## Project structure

```bash
headwater/                  — Rust crate (preprocessing, sampling, inspection)
  src/
    lib.rs                  — crate root (mimalloc global allocator)
    common.rs               — shared types, graph structures, binary format I/O
    embedder.rs             — API-based text embedding (OpenAI-compatible endpoint)
    sampler.rs              — BFS subgraph sampler, batch packing, prefetch pipeline
    python.rs               — PyO3 bindings (Python-callable Sampler with zero-copy NumPy)
    bin/
      preprocess.rs         — data preprocessing binary
      inspect.rs            — preprocessed database inspector / debugger
      single_sample.rs      — single-sample debugging tool
confluence/                 — Python/JAX model (training loop, loss, optimizer)
documentation/              — design docs (architecture, preprocessing, sampling, etc.)
scripts/                    — helper scripts (metadata generation, etc.)
data/
  metadata/                 — human-annotated schema JSON files
  raw/                      — source parquet files
  processed/                — preprocessed binary output
```

## Preprocessed output layout

```bash
data/processed/<dataset>/
  metadata.json              — schema, column stats, task definitions (JSON)
  column_embeddings.bin      — flat [C, 256] f16 array (one per global column)
  categorical_embeddings.bin — flat [Vc, 256] f16 array (all categorical value embeddings, GPU-resident)
  text_embeddings.bin        — flat [Vt, 256] f16 array (all text value embeddings, per-batch subsets)
  graph.bin                  — bidirectional CSR graph (FK edges)
  tables/
    <table_name>.bin        — packed column store per table
  tasks/
    <task_name>.bin         — materialized prediction task
```
