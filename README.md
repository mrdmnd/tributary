# Tributary

A foundation model for learning internal representations of relational databases.

## Overview

Tributary is a system that is design to train a foundation models to understand relational databases, via prediction.
The underlying model derives signal from human-annotated "tasks", which correspond to regression or classification
problems. As the model learns, it should get better at these tasks across databases.

## Setup

To preprocess your own databases, you need to have a frozen text embedder endpoint available.
The default in this project is to use an OpenAI compatible endpoint; how you get this hosted is up to you.
I've found that Baseten is a really good high speed host for throughput-sensitive applications like this preprocessing.

If you want to use the Baseten hosted endpoint, you need to set the following environment variables:
- `BASETEN_EMBEDDER_URL` - the base URL of the Baseten endpoint
- `BASETEN_API_KEY` - the API key for the Baseten endpoint

If you want to use the OpenRouter endpoint for the "metadata generation agent", you need to set the following variable:
- `OPENROUTER_API_KEY` - the base URL of the OpenRouter endpoint



## Representations

We first define the objects we're interested in studying.

First, the relational database.

A relational database is a collection of tables, some of which may be "joinable" to each
other through a primary key/foreign key relationship. We can represent this as a schema
graph, with directed edges between tables.

Each table is also a collection of rows, each of which has some columns. A particular
"cell" is identified by a (table, row, column) tuple. Cells can be "null" or "not null".

These columns in a table can have different "data types" - these are the primitive types
that the column is stored as - for example, in MySQL, this could be "INT", "FLOAT",
"VARCHAR", "BOOLEAN", etc.

However, columns ALSO have meaningful "semantic types" - these encode the meaning of the
column in the context of what it's attempting to represent. For example, a column that
holds a primary key might be an "Identifier" column, whereas a column that holds a date
might be a "Timestamp" column, and a column that holds a string might be a "Text" column
(when the contents are semantically meaningful) or a "Categorical" column (when the
contents do not have meaningful semantics, but serve as an enumeration of some kind.)

Our model supports the following semantic types:

- `Identifier`
- `Numerical`
- `Timestamp`
- `Boolean`
- `Categorical`
- `Text`

Every column that is not one of the above semantic types is considered "Ignored"
and is skipped entirely during preprocessing and training.

## Preprocessing

Assumption: the database comes to us in the form of a collection of parquet files, and a
special "metadata.json" file that has been human-annotated with some information about
the schema. We need information on the semantic types (see above) for each column in the
database, as well as some information about the signal columns that are worth masking
and predicting.

There is a bit of work done here at preprocessing time - we construct a graph
representation of the rows in the database (bidirectional CSR graph), and we also
encode each semantic type cell into a special value.

Identifiers are not encoded at all - merely "present" / "absent". The inductive bias
here is that there might be some signal in the presence / absence of an identifier, but
it's not clear what that signal is - the model will learn.

Numerical values are encoded as z-scored f32 values, with a validity bitmap (null / present).
The scores are normalized _per-column_ for numerical values.

Timestamp values are cyclically encoded, with a validity bitmap (null / present).

- second of minute
- minute of hour
- hour of day
- day of week
- day of month
- month of year
- day of year

The timestamp itself (i64 microseconds since epoch) is also part of the feature, but it's
z-score normalized to an f32 based on all timestamp values across all tables in the database.

Boolean values are encoded as 0 / 1 values, with a validity bitmap (null / present).

Categorical values are encoded as an _index_ into a vocabulary embedding table, for the string
"column name is X". For example, if the column name is "color", and the value is "red",
we use a frozen text embedder to embed "color is red", put that embedding into a shared
vocabulary table (`vocab_embeddings.bin`), and store the index into that table.

Text values are similarly encoded — we use the same frozen text embedder for non-identifier
(semantically meaningful) text values, sharing the same vocabulary table.

## Training

We train the model using a masked language model (MLM) objective.

For numeric and timestamp values, we use Huber regression loss.
For boolean values, we use binary cross-entropy loss.
For categorical values, we use InfoNCE loss

## Building

```bash
cd headwater
cargo build --release
```

## Project structure

```
headwater/                  — Rust crate (preprocessing, sampling, inspection)
  src/
    lib.rs                  — crate root (mimalloc global allocator)
    common.rs               — shared types, graph structures, binary format I/O
    embedder.rs             — API-based text embedding (OpenAI-compatible endpoint)
    sampler.rs              — batch sampler for training (WIP)
    bin/
      preprocess.rs         — data preprocessing binary
      inspect.rs            — preprocessed database inspector / debugger
      single_sample.rs      — single-sample debugging tool (WIP)
confluence/                 — Python/JAX model (WIP)
documentation/              — design docs (architecture, preprocessing, sampling, etc.)
scripts/                    — helper scripts (metadata generation, etc.)
data/
  metadata/                 — human-annotated schema JSON files
  raw/                      — source parquet files
  processed/                — preprocessed binary output
```

## Preprocessed output layout

```
data/processed/<dataset>/
  metadata.json             — schema, column stats, task definitions (JSON)
  column_embeddings.bin     — flat [C, 256] f16 array (one per global column)
  vocab_embeddings.bin      — flat [V, 256] f16 array (one per distinct categorical/text value)
  graph.bin                 — bidirectional CSR graph (FK edges)
  tables/
    <table_name>.bin        — packed column store per table
  tasks/
    <task_name>.bin         — materialized prediction task
```
