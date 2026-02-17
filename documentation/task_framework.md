# Unified Task Framework for Tributary

> How the model gets learning signal from relational databases: a single abstraction around prediction tasks

---

## Core Abstraction

All learning signal is expressed as **tasks** against a database.
We specify at training-time the set of tasks we're interested in solving for our database.
Every task is a SQL query that materializes ground-truth labels as tuples:

```
(anchor_row, observation_time, target_value)
```

- **anchor_row**: which row to root the BFS subgraph on.
- **observation_time**: the point in time the model observes from (epoch microseconds, `i64`).
  The sampler will only include rows that existed at or before this time (temporal filtering).
  Every task always has an observation time per seed — the preprocessor resolves it at materialization time (see below).
- **target_value**: a (derived) quantity that the model predicts.
  Semantically interpreted as an instance of `target_stype` (numerical, categorical, boolean, or timestamp).
  Loss computation is determined by this value and semantic type.

Tasks live in the `tasks` map of `metadata.json`. The training loop samples a
task, draws seeds from that task's anchor rows, builds subgraphs, masks the target
on the seed row, and computes a single-type loss. The model and sampler are
task-agnostic — they just see anchor rows, subgraphs, and targets.

---

## Task Spectrum

Tasks range from trivial cell masking to complex derived predictions.
All tasks are defined with the same structure:

```json
"(task name)": {
    "query": "(defining sql query)",
    "anchor_table": "(base table)",
    "anchor_key": "(primary key for base table rows)",
    "target_column": "(name of column in query that we are predicting)",
    "target_stype": "(semantic type for target column)",
    "observation_time_column": "(name of column in query that establishes observational timestamp)"
}
```



### Cell masking (trivial)

Predict an existing column value on an anchor row for a pre-existing table.
The simplest possible task, which sorta vaguely looks like BERT-style masking:

```json
task: {
    "name": "predict_vote_type",
    "query": "SELECT Id, VoteTypeId FROM 'votes.parquet'",
    "anchor_table": "votes",
    "anchor_key": "Id",
    "target_column": "VoteTypeId",
    "target_stype": "categorical"
}
```

The target is a cell (VoteTypeId) that already exists in the table.
During training, the model will materialize this task table, sample anchor/seed rows from it,
mask out the VoteTypeId cell, and predict it from the surrounding subgraph.

No `observation_time_column` is needed — the preprocessor falls back to the
anchor table's `temporal_column` (if any), or `i64::MAX` for static tables.
See "Observation time resolution" above.

### Filtered cell tasks

You can set up tasks the same as above, but restricted to some set of rows meeting some condition:

```sql
SELECT Id, rating FROM 'orders.parquet' WHERE rating IS NOT NULL
```

### Derived tasks (aggregation, existence, etc.)

The target cell can be computed from joins and aggregations — it doesn't have to exist as a cell in any table.
These require temporal bounds in the SQL and an explicit `observation_time_column`:

```sql
-- "How many comments will this post get in 7 days?"
SELECT p.Id,
       p.CreationDate + INTERVAL '7 days' AS obs_time,
       COUNT(c.Id) AS num_comments
FROM 'posts.parquet' p
LEFT JOIN 'comments.parquet' c
  ON c.PostId = p.Id
  AND c.CreationDate <= p.CreationDate + INTERVAL '7 days'
GROUP BY p.Id, p.CreationDate
```

```sql
-- "Will this question get an accepted answer within 30 days?"
SELECT q.Id,
       q.CreationDate + INTERVAL '30 days' AS obs_time,
       (a.Id IS NOT NULL) AS has_accepted
FROM 'posts.parquet' q
LEFT JOIN 'posts.parquet' a
  ON q.AcceptedAnswerId = a.Id
  AND a.CreationDate <= q.CreationDate + INTERVAL '30 days'
WHERE q.PostTypeId = 1
```

---

## Training Loop

Each training step:

1. **Sample a task** (uniform random, or weighted by importance/difficulty).
2. **Draw a batch of seeds** from that task's anchor rows.
3. **BFS from each seed**, filtering by the seed's observation time.
4. **Mask the target column** on the seed row (for cell masking tasks) or
   provide the derived target externally (for aggregation tasks).
5. **Forward pass** through the relational transformer.
6. **Compute loss** using the type-specific decoder head matching `target_stype`.

Every sample in a batch comes from the same task, so the batch is homogeneous:
one target type, one decoder head, one loss function. This is simpler and cleaner
than BERT-style MLM where a single batch has mixed target types scattered across
positions.

---

## Temporal Correctness

Temporal leakage is prevented at two levels:

### Target leakage (SQL level)

Derived task queries must include temporal bounds so they don't count future
events. Cell masking tasks are inherently leak-free (the target is a cell on the
anchor row itself).

### Feature leakage (BFS level)

Each table can declare a `temporal_column` — the column representing when its
rows came into existence. During BFS, rows in tables with a `temporal_column` are
only included if their temporal value <= the seed's observation time. Tables
without `temporal_column` (static/dimension data) are always traversable.

### Observation time resolution

The preprocessor resolves an observation time for **every** seed row at
materialization time, so the sampler just reads a concrete `i64` per seed:

1. If the task defines `observation_time_column` → use that column's value from the query results.
2. Else if the anchor table has a `temporal_column` → use the anchor row's temporal value.
3. Else → `i64::MAX` (no temporal filtering; the entire database is visible).

This cascade is resolved once during preprocessing and stored in the task
binary file. The sampler never needs to implement fallback logic — it
unconditionally reads `observation_times[seed]` and uses it as the BFS cutoff.

---

## Train / Validation / Test Splits

### Hash-based splitting (all tasks)

All tasks use the same split mechanism: a deterministic hash of
`(task_idx, anchor_row_id, split_seed)` assigns each seed to train, val,
or test at configurable ratios (default 80/10/10).

Observation times and split assignment are **independent concerns**:
- **Observation time** governs what each seed's BFS can see (temporal
  filtering). A seed with `obs_time = June 2022` only sees data from
  before that date, regardless of whether the seed is train or val.
- **Split assignment** governs which seeds contribute to training loss
  vs. validation loss. Determined solely by the hash.

This means temporal tasks don't need a special split mechanism. Every
seed already carries its own temporal cutoff. Whether a seed is used for
training or validation doesn't change what context the model sees for
that seed.

See [system_architecture.md](system_architecture.md) §2b for the hash
mechanism, properties (cross-rank determinism, stability across runs,
per-task independence), and discussion of transductive evaluation.

### Task-level holdout

Hold out entire tasks for evaluation. Train on cell masking tasks, evaluate on
derived tasks (or vice versa). Tests whether learned representations transfer
to novel prediction objectives.

### Cross-database holdout

Pre-train on databases A, B, C. Evaluate on database D. Tests whether
representations generalize across schemas, domains, and data distributions.

---

## Three-Phase Training Recipe

### Phase 1: Pre-train (cell masking across many databases)

- Auto-generate cell masking tasks from each database's schema.
- Sample databases, sample tasks within databases, sample seeds within tasks.
- Self-supervised — no human annotation beyond schema metadata.
- Model learns general relational representations.

### Phase 2: Fine-tune (SQL tasks on target database)

- Write or auto-generate derived tasks for a specific database.
- Fine-tune the pre-trained model on these tasks.
- Or freeze the model and train only a lightweight task-specific head.

### Phase 3: Evaluate (held-out tasks and/or held-out databases)

- Hash-based split within tasks for standard evaluation.
- Task-level holdout for transfer evaluation.
- Cross-database holdout for generalization evaluation.

---