"""Generate a first-draft metadata.json from a directory of parquet files.

Uses a Pydantic AI agent with tools for introspecting parquet file schemas,
sampling data, and checking cardinalities to produce a high-quality initial
metadata annotation. The human annotator then reviews and corrects the output.

Usage:
    uv run generate_metadata.py <data_dir> [--model MODEL] [--force]

Requires the OPENROUTER_API_KEY environment variable to be set.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Literal

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits

# ============================================================================
# Pydantic output models (match the metadata.json schema)
# ============================================================================


class ColumnMeta(BaseModel):
    """Metadata for a single column."""

    stype: Literal[
        "identifier",
        "numerical",
        "timestamp",
        "boolean",
        "categorical",
        "text",
        "ignored",
    ]
    foreign_key: str | None = None
    description: str | None = None


class TableMeta(BaseModel):
    """Metadata for a single table."""

    primary_key: str | None = None
    temporal_column: str | None = None
    columns: dict[str, ColumnMeta]


class TaskDef(BaseModel):
    """A prediction task defined as a SQL query."""

    query: str
    anchor_table: str
    anchor_key: str
    target_column: str
    target_stype: Literal["numerical", "categorical", "boolean", "timestamp"]
    observation_time_column: str | None = None


class DatasetMetadata(BaseModel):
    """Top-level metadata for a dataset of parquet files."""

    name: str
    tables: dict[str, TableMeta]
    tasks: dict[str, TaskDef]


# ============================================================================
# System instructions
# ============================================================================

SYSTEM_INSTRUCTIONS = """\
You are a database schema analyst. Your task is to examine a collection of
parquet files and produce a metadata.json annotation describing the relational
schema and defining temporally-correct prediction tasks. You have tools to
introspect the data.

## Workflow

1. Call list_tables() to discover all tables.
2. For each table, call get_schema() to see column names and Arrow types.
3. Use sample_rows() to look at actual data values.
4. Use count_distinct() to check cardinality of columns (key signal for
   distinguishing identifiers, categoricals, and text).
5. Use value_counts() to see the vocabulary of low-cardinality columns.
6. Use column_stats() on numerical columns to understand their distribution.
7. Use check_foreign_key() to test suspected FK relationships between tables
   (e.g. if orders.customer_id values are a subset of customers.customer_id).
8. Identify (optionally) a temporal_column for each table (see Temporal Columns below).
9. Define prediction tasks as SQL queries (see Tasks section below). Use
   run_query() to verify your queries work before emitting them.
10. Produce the final DatasetMetadata output with every column annotated and
    tasks defined.

## Semantic Types (stype)

Every column MUST receive one of these types:

- "identifier": Distinguishes entities (e.g. customer_id, email, display_name).
  High cardinality, values are labels not meaningful content.
  Integer columns named *Id or *_id are almost always identifiers.
  String columns that are names/labels (not prose) are often identifiers.
  Other examples: phone numbers, email addresses, URLs, etc. 
  Textual fields where the *language semantics* of the value are NOT meaningful might
  be identifiers.

- "numerical": Numeric value where ordering and magnitude matter (e.g. age,
  price, score, quantity). NOT for integer columns that are IDs or codes; these should
  be marked as identifiers.

- "timestamp": An instant in time (e.g. created_at, birth_date).
  Arrow timestamp and date types are always this.

- "boolean": True/false values (e.g. is_active, has_discount).

- "categorical": Drawn from a small fixed vocabulary (e.g. country_code,
  status, product_category). Key signal: low cardinality (typically < 100
  distinct values relative to table size). Integer-flavored columns with
  a small number of distinct values are often categorical, not identifiers.

- "text": Multi-token strings with semantic meaning worth embedding
  (e.g. descriptions, reviews, bios, article bodies).

- "ignored": Column should be skipped. Use for binary blobs, audit fields,
  data artifacts, raw bytestrings, or other columns with no useful signal.

## Keys and Relationships

- Set primary_key when a table has a clear single-column primary key.
- Set foreign_key on a column using "target_table.target_column" format when
  the column references another table's primary key. Verify with
  check_foreign_key() that the values actually match.

## Temporal Columns

Each table can optionally declare a temporal_column — the column that
represents when the row came into existence. This is used by the subgraph
sampler to prevent temporal leakage: during BFS, only rows whose temporal
value is at or before the seed's observation time are included.

Guidelines:
- Most event/transaction tables have a creation timestamp (e.g. CreationDate,
  created_at, Date). Set temporal_column to this column.
- Static reference or dimension tables (e.g. a countries lookup table) have
  no meaningful creation time. Omit temporal_column for these.
- The temporal_column must be a column with stype "timestamp" in the same
  table.
- Some tables may have multiple temporal columns. For example, a table that 
  a row with multiple "checkpoint" timestamps, indicating progression through
  a process. In this case, you'll have to use judgement to determine which column
  corresponds to the "creation" time for that row.

## Tasks

Prediction tasks are defined as SQL queries (DuckDB dialect) that the
preprocessor will execute against the parquet files to materialize
ground-truth labels. Each task produces (anchor_id, target_value) tuples.
Some tasks may also produce (anchor_id, observation_time, target_value) tuples.

IMPORTANT: All tasks must be free of temporal leakage.
The model must not see data "from the future" when making a prediction.

In your SQL queries, reference tables as quoted parquet filenames:
'tablename.parquet' (e.g. FROM 'posts.parquet' p).

### Task fields

Tasks are an object mapping task name (a unique human-readable identifier,
e.g. "predict_vote_type") to its definition:

- query: SQL query returning at least anchor_key and target_column (and
  observation_time_column if the task specifies one)
- anchor_table: which table the sampler roots its subgraph sampling on
- anchor_key: column in query results that joins back to anchor_table primary key
- target_column: column in query results that the model predicts
- target_stype: one of "numerical", "categorical", "boolean", "timestamp"
- observation_time_column: (optional) column in query results that specifies
  the per-seed temporal cutoff. Required for derived tasks with temporal
  bounds. If omitted, the anchor table's temporal_column is used.

### Cell masking tasks (simple)

For each interesting non-identifier column, create a trivial task:

    "predict_<column>": {
        query: "SELECT <pk>, <column> FROM '<table>.parquet'"
        anchor_table: "<table>"
        anchor_key: "<pk>"
        target_column: "<column>"
    }

Cell masking tasks do NOT need observation_time_column — the sampler will
automatically use the anchor table's temporal_column as the observation
time. This means the model sees only data that existed at or before the
anchor row's own timestamp.

A good cell masking target satisfies:
(a) Someone would care about predicting this value (ratings, scores, outcomes, classifications, event times).
(b) It has some variance. Even near-constant columns can be interesting (rare events like fraud).
(c) It is plausibly predictable from the graph neighborhood.
(d) It is a consequence, not a precondition. Predict outcomes, not inputs.

Timestamps can be good targets when they represent events predictable from
graph context (e.g. when a vote or comment will occur, or when a post will be accepted), but not when they
are just "when did this entity get created" with no upstream signal.

### Derived tasks (aggregation, existence, etc.)

Derived tasks compute the target from joins and aggregations. These MUST:
(a) Include temporal bounds in the SQL to avoid counting future events.
(b) Output an observation_time column so the sampler knows what point in
    time the model is observing from.
(c) Set observation_time_column to the name of that column.

Use a fixed time horizon (e.g. 7 days, 30 days) after the anchor row's
creation timestamp. The SQL query bounds the aggregation to that window,
and outputs the end of the window as the observation time.

Examples:

- Aggregation with 7-day horizon:
  SELECT p.Id,
         p.CreationDate + INTERVAL '7 days' AS obs_time,
         COUNT(c.Id) AS num_comments
  FROM 'posts.parquet' p
  LEFT JOIN 'comments.parquet' c
    ON c.PostId = p.Id
    AND c.CreationDate <= p.CreationDate + INTERVAL '7 days'
  GROUP BY p.Id, p.CreationDate

- Existence with 30-day horizon:
  SELECT q.Id,
         q.CreationDate + INTERVAL '30 days' AS obs_time,
         (a.Id IS NOT NULL) AS has_accepted
  FROM 'posts.parquet' q
  LEFT JOIN 'posts.parquet' a
    ON q.AcceptedAnswerId = a.Id
    AND a.CreationDate <= q.CreationDate + INTERVAL '30 days'
  WHERE q.PostTypeId = 1

Use run_query() to test your SQL queries and make sure they execute
correctly before including them.

Be selective but not too conservative — the human annotator will trim.
Consider creating multiple time horizons for important aggregation tasks
(e.g. predict_num_comments_7d, predict_num_comments_30d).

## Other Fields

- description: Add a brief human-readable description to EVERY column based
  on what you learn from the data. This is concatenated with the column name
  before embedding, so it meaningfully improves the model's prior understanding.
  For categorical/code columns, enumerate the values (e.g. "1=Gold, 2=Silver,
  3=Bronze"). For text columns, describe what kind of content they contain.
  For identifiers, note what entity they reference. For numerical columns,
  describe what the value represents and its typical range. Keep descriptions
  concise (one sentence).

## Important Rules

- EVERY column in the parquet file must appear in the "columns" dict with an
  explicit stype. To mark a column to skip, give it stype "ignored".
- The dataset name should be the directory name.
- Use your tools thoroughly — do not guess when you can check.
- Verify ALL SQL task queries with run_query() before emitting them.
- Set temporal_column on every table that has a creation timestamp.
- All derived/aggregation tasks MUST include temporal bounds and output an
  observation_time_column.
"""

# ============================================================================
# Tools (registered on agent at build time)
# ============================================================================


def _list_tables(ctx: RunContext[Path]) -> list[str]:
    """List all parquet tables in the dataset directory. Returns table names (without .parquet extension)."""
    return sorted(p.stem for p in ctx.deps.glob("*.parquet"))


def _get_schema(ctx: RunContext[Path], table: str) -> list[dict[str, str]]:
    """Get the Arrow schema for a table. Returns a list of {name, arrow_type, nullable} for each column."""
    schema = pq.read_schema(ctx.deps / f"{table}.parquet")
    return [
        {
            "name": schema.field(i).name,
            "arrow_type": str(schema.field(i).type),
            "nullable": str(schema.field(i).nullable),
        }
        for i in range(len(schema))
    ]


def _sample_rows(ctx: RunContext[Path], table: str, n: int = 10) -> list[dict[str, Any]]:
    """Read the first N rows of a table. Returns a list of row dicts with all values stringified."""
    n = min(n, 50)
    tbl = pq.read_table(ctx.deps / f"{table}.parquet").slice(0, n)
    rows: list[dict[str, Any]] = []
    for batch in tbl.to_batches():
        for row_idx in range(batch.num_rows):
            row = {}
            for col_idx in range(batch.num_columns):
                val = batch.column(col_idx)[row_idx].as_py()
                row[batch.schema.field(col_idx).name] = str(val) if val is not None else None
            rows.append(row)
    return rows


def _count_distinct(ctx: RunContext[Path], table: str, column: str) -> dict[str, int]:
    """Count distinct and null values for a column. Returns {total_rows, distinct_count, null_count}."""
    tbl = pq.read_table(ctx.deps / f"{table}.parquet", columns=[column])
    arr = tbl.column(column)
    total = len(arr)
    null_count = arr.null_count
    distinct = pc.count_distinct(arr, mode="only_valid").as_py()
    return {"total_rows": total, "distinct_count": distinct, "null_count": null_count}


def _value_counts(ctx: RunContext[Path], table: str, column: str, top_k: int = 20) -> list[dict[str, Any]]:
    """Get the most frequent values for a column. Returns top_k [{value, count}] pairs, descending by count."""
    top_k = min(top_k, 50)
    tbl = pq.read_table(ctx.deps / f"{table}.parquet", columns=[column])
    arr = tbl.column(column)
    vc = pc.value_counts(arr)
    values_arr = vc.field("values")
    counts_arr = vc.field("counts")

    pairs = []
    for i in range(len(vc)):
        v = values_arr[i].as_py()
        c = counts_arr[i].as_py()
        pairs.append({"value": str(v) if v is not None else None, "count": c})

    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:top_k]


def _column_stats(ctx: RunContext[Path], table: str, column: str) -> dict[str, Any]:
    """Compute distribution statistics for a numerical column.

    Returns {total_rows, null_count, min, max, mean, std, median,
    dominant_value_fraction}. dominant_value_fraction is the fraction of
    non-null rows occupied by the single most common value — useful for
    detecting near-constant columns.
    """
    tbl = pq.read_table(ctx.deps / f"{table}.parquet", columns=[column])
    arr = tbl.column(column)
    total = len(arr)
    null_count = arr.null_count
    non_null = arr.drop_null()

    if len(non_null) == 0:
        return {"total_rows": total, "null_count": null_count, "non_null_count": 0}

    # Cast to float64 for stats if needed (handles int columns, booleans, etc.)
    try:
        farr = non_null.cast(pa.float64())
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return {
            "total_rows": total,
            "null_count": null_count,
            "non_null_count": len(non_null),
            "error": "Cannot cast to numeric for stats",
        }

    min_val = pc.min(farr).as_py()
    max_val = pc.max(farr).as_py()
    mean_val = pc.mean(farr).as_py()
    std_val = pc.stddev(farr, ddof=1).as_py()
    median_val = pc.approximate_median(farr).as_py()

    # Dominant value fraction: how much of the data is a single value?
    vc = pc.value_counts(non_null)
    max_count = pc.max(vc.field("counts")).as_py()
    dominant_frac = round(max_count / len(non_null), 4)

    def _safe(v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return str(v)
        return round(v, 6) if isinstance(v, float) else v

    return {
        "total_rows": total,
        "null_count": null_count,
        "non_null_count": len(non_null),
        "min": _safe(min_val),
        "max": _safe(max_val),
        "mean": _safe(mean_val),
        "std": _safe(std_val),
        "median": _safe(median_val),
        "dominant_value_fraction": dominant_frac,
    }


def _check_foreign_key(
    ctx: RunContext[Path],
    from_table: str,
    from_column: str,
    to_table: str,
    to_column: str,
) -> dict[str, Any]:
    """Check if non-null values in from_table.from_column are a subset of to_table.to_column.

    Returns {total_non_null, matched, match_rate}. A match_rate close to 1.0
    strongly suggests a foreign key relationship.
    """
    from_tbl = pq.read_table(ctx.deps / f"{from_table}.parquet", columns=[from_column])
    to_tbl = pq.read_table(ctx.deps / f"{to_table}.parquet", columns=[to_column])

    from_arr = from_tbl.column(from_column)
    to_arr = to_tbl.column(to_column)

    to_set = set(to_arr.drop_null().to_pylist())
    from_non_null = [v for v in from_arr.to_pylist() if v is not None]

    total = len(from_non_null)
    if total == 0:
        return {"total_non_null": 0, "matched": 0, "match_rate": 0.0}

    matched = sum(1 for v in from_non_null if v in to_set)
    return {
        "total_non_null": total,
        "matched": matched,
        "match_rate": round(matched / total, 4),
    }


def _run_query(ctx: RunContext[Path], query: str) -> list[dict[str, Any]]:
    """Execute a SQL query (DuckDB) against the parquet files and return up to 10 sample rows.

    Tables are referenced by their parquet filename (e.g. 'posts.parquet').
    Use this to verify that task queries work before emitting them.
    Example: SELECT p.Id, COUNT(c.Id) AS n FROM 'posts.parquet' p LEFT JOIN 'comments.parquet' c ON c.PostId = p.Id GROUP BY p.Id LIMIT 10
    """
    con = duckdb.connect()
    con.execute(f"SET file_search_path = '{ctx.deps}'")
    try:
        result = con.execute(query)
        col_names = [desc[0] for desc in result.description]
        raw_rows = result.fetchmany(10)
        rows = []
        for raw in raw_rows:
            row = {}
            for name, val in zip(col_names, raw):
                row[name] = str(val) if val is not None else None
            rows.append(row)
        return rows
    except Exception as e:
        return [{"error": str(e)}]
    finally:
        con.close()


ALL_TOOLS = [
    _list_tables,
    _get_schema,
    _sample_rows,
    _count_distinct,
    _value_counts,
    _column_stats,
    _check_foreign_key,
    _run_query,
]


def build_agent(model: str) -> Agent[Path, DatasetMetadata]:
    """Construct the metadata agent with the given model string."""
    return Agent(
        model,
        deps_type=Path,
        output_type=DatasetMetadata,
        instructions=SYSTEM_INSTRUCTIONS,
        tools=ALL_TOOLS,
    )


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a first-draft metadata.json using an LLM agent.")
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing .parquet files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="moonshotai/kimi-k2.5",
        help="OpenRouter model to use",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing metadata.json if present.",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir.resolve()
    if not data_dir.is_dir():
        print(f"Not a directory: {data_dir}", file=sys.stderr)
        sys.exit(1)

    output_path = data_dir / "metadata.json"
    if output_path.exists() and not args.force:
        print(
            f"{output_path} already exists. Use --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)

    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No .parquet files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    model_name = f"openrouter:{args.model}"
    print(f"Using model: {model_name}")
    print(f"Analyzing {len(parquet_files)} parquet files in {data_dir}...")

    agent = build_agent(model_name)
    result = agent.run_sync(
        f"Analyze the parquet dataset in directory '{data_dir.name}' and produce the metadata annotation.",
        deps=data_dir,
        usage_limits=UsageLimits(request_limit=200),
    )

    metadata_dict = result.output.model_dump(exclude_none=True)
    output_path.write_text(json.dumps(metadata_dict, indent=4) + "\n")
    print(f"Wrote {output_path}")
    print(f"Usage: {result.usage()}")


if __name__ == "__main__":
    main()
