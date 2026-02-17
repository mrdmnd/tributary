# Preprocessing: Input and Output Data Formats

## Hierarchy

The directory structure of our data hierarchy is:
data/
- metadata/
  - db1.json
  - db2.json
  - ...
- raw/
  - db1/
    - table1.parquet
    - table2.parquet
    - ...
  - db2/
    - table1.parquet
    - table2.parquet
    - ...
- processed/
  - db1/
    - tables/
      - table1.bin
      - table2.bin
      - ...
    - tasks/
      - task1.bin
      - task2.bin
      - ...
    - metadata.json
    - column_embeddings.bin
    - categorical_embeddings.bin
    - text_embeddings.bin
    - graph.bin

## Raw Input Data 

We get input datasets as collections of .parquet files - these are stored in the `raw` directory, with their database
name as the directory name. Each raw database directory contains one `.parquet` files per table.

The assumptions we make about these files are pretty limited. They are intended to represent a *relational* database,
not just a collection of tables, so having some level of linkage between them is important. If there's a table present
that simply doesn not have any connections to any other tables, it's not particularly relevant to our model, because it
will never be sampled as part of a graph walk. Your database may actually have plenty of tables like this - it's up to
you what you want to do with them.

## Metadata

We also track a separate directory of metadata as JSON - these files are named the same as the database.
The metadata file for a given DB fully describes the relational schema and defines prediction tasks.
`data/metadata/{database}.json` is the sole source of truth for semantic types, keys, relationships, and tasks.

### Generating Metadata Drafts

Creating a full metadata file can be a little bit annoying for humans. We've harnessed an AI agent to write these out
for us! The human will still need to *validate* that all of the semantic types, prediction tasks, etc make sense, but
this helper script (generate_metadata.py) will use an LLM to generate a plausible starting point.

Use the helper script to auto-generate a starting point from parquet schemas:

```
uv run --project scripts scripts/generate_metadata.py data/raw/<dataset_dir>
```

This uses an LLM agent to introspect the parquet files and produce an initial annotation.
The script writes `metadata.json` into the given data directory; you'll need to move it to the
expected location afterwards:

```
mv data/raw/<dataset_dir>/metadata.json data/metadata/<dataset_dir>.json
```

The human annotator then reviews and corrects the output.

This metadata is source-controlled.

### Metadata Structure

```json
{
    "name": "<dataset_name>",
    "tables": { ... },
    "tasks": { ... }
}
```

| Field    | Type   | Required | Description                                 |
|----------|--------|----------|---------------------------------------------|
| `name`   | string | yes      | Dataset identifier.                         |
| `tables` | object | yes      | Map of table name to table object.          |
| `tasks`  | object | yes      | Map of task name to task definition.        |

The table name determines the parquet filename: table `"users"` reads from
`users.parquet` in the same directory.

### Table object

```json
{
    "primary_key": "Id",
    "temporal_column": "CreationDate",
    "columns": { ... }
}
```

| Field             | Type   | Required | Default | Description                                           |
|-------------------|--------|----------|---------|-------------------------------------------------------|
| `primary_key`     | string | no       | none    | Column name of the primary key.                       |
| `temporal_column` | string | no       | none    | Column that represents when this row came into existence. Used by the sampler for temporal filtering during BFS. |
| `columns`         | object | yes      |         | Map of column name to column object. Every column in the parquet file must appear here. |

To skip a column, set its `stype` to `"ignored"` in the `columns` map.

Tables that represent static reference or dimension data (no meaningful
creation timestamp) should omit `temporal_column`. During BFS, rows in
tables without a `temporal_column` are always traversable.

### Column object

```json
{
    "stype": "identifier",
    "foreign_key": "posts.Id",
    "description": "1=Question, 2=Answer"
}
```

| Field         | Type   | Required | Default | Description                                                      |
|---------------|--------|----------|---------|------------------------------------------------------------------|
| `stype`       | string | yes      |         | Semantic type. Must be one of the values listed below.           |
| `foreign_key` | string | no       | none    | Foreign key target in `"table.column"` format.                   |
| `description` | string | no       | none    | Concatenated with column name before embedding.                  |

### Semantic types (`stype`)

Every column must have an explicit `stype`. The valid values map 1:1 to the
`SemanticType` enum in `src/common.rs`:
`"identifier"`, `"numerical"`, `"timestamp"`, `"boolean"`, `"categorical"`,
`"text"`, `"ignored"`.

See [semantic_types.md](semantic_types.md) for full descriptions, encoding
details, and guidance on assigning types.

### Common corrections to look for

- Integer columns like `PostTypeId` or `VoteTypeId` that are really **categoricals**, not identifiers.
- String columns like `DisplayName` that are really **identifiers**, not text.
- String columns like `ContentLicense` that are really **categoricals** with a small fixed vocabulary.
- Numeric columns like `AccountId` that are really **identifiers**.
- Columns that should be **ignored** (audit fields, GUIDs, data artifacts) but aren't.

## Tasks

Prediction tasks are defined as SQL queries executed against the parquet files.
Each task specifies what to predict and which table anchors the subgraph sampling.

See [task_framework.md](task_framework.md) for the conceptual overview of tasks,
the task spectrum (cell masking → derived predictions), temporal correctness,
observation time resolution, and training/evaluation strategy.

### Task object

The key in the `tasks` map is the unique human-readable task name. The value is:

```json
{
    "query": "SELECT order_id, rating FROM 'orders.parquet' WHERE rating IS NOT NULL",
    "anchor_table": "orders",
    "anchor_key": "order_id",
    "target_column": "rating",
    "target_stype": "numerical"
}
```

| Field                      | Type   | Required | Description                                                         |
|----------------------------|--------|----------|---------------------------------------------------------------------|
| `query`                    | string | yes      | SQL query (DataFusion-compatible) against the parquet files. Must return at least `anchor_key` and `target_column`. |
| `anchor_table`             | string | yes      | Table that roots the subgraph sampling on.                          |
| `anchor_key`               | string | yes      | Column in query results that joins back to anchor_table rows.       |
| `target_column`            | string | yes      | Column in query results that the model predicts.                    |
| `target_stype`             | string | yes      | Semantic type of the target. One of: `numerical`, `categorical`, `boolean`, `timestamp`. |
| `observation_time_column`  | string | no       | Column in query results that specifies the temporal cutoff per seed row. Required for derived tasks with temporal bounds. If omitted, the preprocessor falls back to the anchor table's `temporal_column`, or `i64::MAX` for static tables. |

The preprocessor executes each query to materialize ground-truth
`(anchor_row, observation_time, target_value)` tuples. Observation times
are always resolved and stored — see [task_framework.md](task_framework.md)
for the resolution rules.

## Nullability

Column nullability is **not** represented in `metadata.json`. The parquet
files already carry this information in their Arrow schemas, and the
preprocessor reads it directly when writing validity bitmaps. Duplicating it
here would create a source of drift between the metadata and the actual data.

`metadata.json` captures things the parquet schema *cannot* tell you:
semantic types, relationships, and prediction tasks. Physical properties
like nullability and data types are read from the parquet files themselves.

## JSON Schema

```json
{
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Dataset Metadata",
    "description": "Relational schema annotations and prediction tasks for a collection of parquet files.",
    "type": "object",
    "required": ["name", "tables", "tasks"],
    "additionalProperties": false,
    "properties": {
        "name": {
            "type": "string",
            "description": "Dataset identifier."
        },
        "tables": {
            "type": "object",
            "description": "Map of table name to table definition. Table name determines the parquet filename (<name>.parquet).",
            "minProperties": 1,
            "additionalProperties": {
                "$ref": "#/$defs/table"
            }
        },
        "tasks": {
            "type": "object",
            "description": "Map of task name to task definition.",
            "additionalProperties": {
                "$ref": "#/$defs/task"
            }
        }
    },
    "$defs": {
        "table": {
            "type": "object",
            "required": ["columns"],
            "additionalProperties": false,
            "properties": {
                "primary_key": {
                    "type": "string",
                    "description": "Column name of the primary key."
                },
                "temporal_column": {
                    "type": "string",
                    "description": "Column that represents when this row came into existence. Used for temporal filtering during BFS. Omit for static/dimension tables."
                },
                "columns": {
                    "type": "object",
                    "description": "Map of column name to column definition. Every column in the parquet file must appear here.",
                    "minProperties": 1,
                    "additionalProperties": {
                        "$ref": "#/$defs/column"
                    }
                }
            }
        },
        "column": {
            "type": "object",
            "required": ["stype"],
            "additionalProperties": false,
            "properties": {
                "stype": {
                    "type": "string",
                    "enum": ["identifier", "numerical", "timestamp", "boolean", "categorical", "text", "ignored"],
                    "description": "Semantic type of the column."
                },
                "foreign_key": {
                    "type": "string",
                    "pattern": "^[^.]+\\.[^.]+$",
                    "description": "Foreign key target in 'table.column' format."
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable description, concatenated with column name before embedding."
                }
            }
        },
        "task": {
            "type": "object",
            "required": ["query", "anchor_table", "anchor_key", "target_column", "target_stype"],
            "additionalProperties": false,
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL query (DataFusion-compatible) against the parquet files."
                },
                "anchor_table": {
                    "type": "string",
                    "description": "Table that roots the subgraph sampling on."
                },
                "anchor_key": {
                    "type": "string",
                    "description": "Column in query results that joins back to anchor_table rows."
                },
                "target_column": {
                    "type": "string",
                    "description": "Column in query results that the model predicts."
                },
                "target_stype": {
                    "type": "string",
                    "enum": ["numerical", "categorical", "boolean", "timestamp"],
                    "description": "Semantic type of the prediction target."
                },
                "observation_time_column": {
                    "type": "string",
                    "description": "Column in query results specifying the temporal cutoff per seed row. If omitted, the preprocessor falls back to the anchor table's temporal_column, or i64::MAX for static tables."
                }
            }
        }
    }
}
```

## Full example

A fictional online bookstore database with three tables and four tasks:

```json
{
    "name": "example-bookstore",
    "tables": {
        "customers": {
            "primary_key": "customer_id",
            "temporal_column": "signed_up_at",
            "columns": {
                "customer_id":    { "stype": "identifier" },
                "email":          { "stype": "identifier" },
                "display_name":   { "stype": "identifier" },
                "country":        { "stype": "categorical" },
                "age":            { "stype": "numerical" },
                "is_premium":     { "stype": "boolean" },
                "bio":            { "stype": "text", "description": "Free-form customer biography" },
                "internal_notes": { "stype": "ignored" },
                "signed_up_at":   { "stype": "timestamp" }
            }
        },
        "books": {
            "primary_key": "book_id",
            "temporal_column": "published_at",
            "columns": {
                "book_id":      { "stype": "identifier" },
                "title":        { "stype": "text" },
                "genre":        { "stype": "categorical" },
                "description":  { "stype": "text", "description": "Publisher-provided blurb" },
                "price":        { "stype": "numerical" },
                "page_count":   { "stype": "numerical" },
                "published_at": { "stype": "timestamp" }
            }
        },
        "orders": {
            "primary_key": "order_id",
            "temporal_column": "ordered_at",
            "columns": {
                "order_id":    { "stype": "identifier" },
                "customer_id": { "stype": "identifier", "foreign_key": "customers.customer_id" },
                "book_id":     { "stype": "identifier", "foreign_key": "books.book_id" },
                "quantity":    { "stype": "numerical" },
                "rating":      { "stype": "numerical", "description": "1-5 star rating, null if not yet reviewed" },
                "ordered_at":  { "stype": "timestamp" }
            }
        }
    },
    "tasks": {
        "predict_rating": {
            "query": "SELECT order_id, rating FROM 'orders.parquet' WHERE rating IS NOT NULL",
            "anchor_table": "orders",
            "anchor_key": "order_id",
            "target_column": "rating",
            "target_stype": "numerical"
        },
        "predict_genre": {
            "query": "SELECT book_id, genre FROM 'books.parquet'",
            "anchor_table": "books",
            "anchor_key": "book_id",
            "target_column": "genre",
            "target_stype": "categorical"
        },
        "predict_is_premium": {
            "query": "SELECT customer_id, is_premium FROM 'customers.parquet'",
            "anchor_table": "customers",
            "anchor_key": "customer_id",
            "target_column": "is_premium",
            "target_stype": "boolean"
        },
        "predict_order_count_30d": {
            "query": "SELECT c.customer_id, c.signed_up_at + INTERVAL '30 days' AS obs_time, COUNT(o.order_id) AS order_count FROM 'customers.parquet' c LEFT JOIN 'orders.parquet' o ON o.customer_id = c.customer_id AND o.ordered_at <= c.signed_up_at + INTERVAL '30 days' GROUP BY c.customer_id, c.signed_up_at",
            "anchor_table": "customers",
            "anchor_key": "customer_id",
            "observation_time_column": "obs_time",
            "target_column": "order_count",
            "target_stype": "numerical"
        }
    }
}
```
