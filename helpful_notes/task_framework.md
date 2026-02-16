# Unified Task Framework for Tributary

> How the model gets learning signal from relational databases: a single abstraction around prediction tasks

---

## Core Abstraction

All learning signal is expressed as **tasks** against a database.
We specify at training-time the set of tasks we're interested in solving for our database.
Every task is a SQL query that materializes ground-truth labels as tuples:

```
(anchor_row, {observation_time}, target_value)
```

- **anchor_row**: which row to root the BFS subgraph on.
- **observation_time**: the point in time the model observes from.
  The sampler will only include rows that existed at or before this time (temporal filtering).
  Not all tasks will necessarily have an "observation" time.
- **target_value**: a (derived) quantity that the model predicts. 
  Semantically interpreted as an instance of `target_stype` (numerical, categorical, boolean, or timestamp).
  Loss computation is determined by this value and semantic type.

Tasks live in the `tasks` array of `metadata.json`. The training loop samples a
task, draws seeds from that task's anchor rows, builds subgraphs, masks the target
on the seed row, and computes a single-type loss. The model and sampler are
task-agnostic — they just see anchor rows, subgraphs, and targets.

---

## Task Spectrum

Tasks range from trivial cell masking to complex derived predictions.
All tasks are defined with the same structure:

```json
{
    "name": "(task name)",
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
mask out the VoteTypeId cell, and predicts it from the surrounding subgraph.

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

Observation time resolution:

1. If the task has `observation_time_column` -> use that column from query results.
2. Else if the anchor table has `temporal_column` -> use that column's value.
3. Else -> no temporal filtering (i64::MAX).

---

## Train / Validation / Test Splits

### With timestamps (temporal split)

Choose cutoffs T_train and T_val. Partition each task's anchor rows by
observation time:

| Split | Condition                              |
|-------|----------------------------------------|
| Train | observation_time <= T_train             |
| Val   | T_train < observation_time <= T_val    |
| Test  | observation_time > T_val               |

Mirrors deployment (predict the future from the past). Leak-proof by
construction because the BFS temporal filtering already enforces the
observation time boundary.

### Without timestamps (random split)

Randomly assign anchor rows to train/val/test (e.g. 80/10/10). The full graph
is available to all splits (transductive). Standard in GNN literature.

### Task-level holdout

Hold out entire tasks for evaluation. Train on cell masking tasks, evaluate on
derived tasks (or vice versa). Tests whether learned representations transfer
to novel prediction objectives. Works with or without timestamps.

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

- Temporal split within tasks for standard evaluation.
- Task-level holdout for transfer evaluation.
- Cross-database holdout for generalization evaluation.

---