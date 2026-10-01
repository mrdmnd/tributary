# Tributary

Tributary is a project for training relational database prediction models against a data warehouse.

## Overview

Tributary is HEAVILY based on top of Rishabh Ranjan's work in [Relational Transformer](https://arxiv.org/abs/2510.06377).

The core ideas in this project come predominantly from his work.

## Representations, Definitions, and Objects

Let's first define the objects we're interested in studying.

First, the `relational database`.

A relational database is a collection of `tables`, some of which may be connected to each other through column foreign key -> primary key relationship.

Tables have `rows` and `columns`, where the columns have associated `data types` and `semantic types`.

A column's `data type` (also sometimes spelled `dtype` here) tells us "how" the contents of that column are stored.
For example, [common data types](https://www.postgresql.org/docs/current/datatype.html) might include Postgres types like

- smallint (2 byte integer)
- decimal (variable precision numeric type)
- char(n) (fixed length string type)
- text
..
etc etc

Tables generally will have one column marked as a `primary key` column, though this is not required.
Tables may also have one or more columns marked as `foreign keys` - these indicate pointers to other rows in other tables and define legal join relationships.

Most of the concepts discussed so far are standard for relational databases. 

## Tributary-Specific Concepts

Our project also defines a few special pieces of metadata information for the purposes of building a predictive model:

### Semantic Types

First, we define a special column type, the `semantic type` (also sometimes spelled `stype`).
A column's semantic type tells us "what kind" of value the contents of that column represent.

In our model, a column must have exactly one of the following semantic types assigned in metadata:

- `Identifier`
- `Numerical`
- `Timestamp`
- `Boolean`
- `Categorical`
- `Text`
- `Ignored`

`Identifier` stype should be used for primary and foreign keys.
Example column name: "thread_id", example value: 12591259125

`Numerical` stype should be used for columns that contain values that are... numerical, where the magnitude of the number is meaningful.
Example column name: "temperature", example value: 25.1

`Timestamp` stype should be used for columns that are times or dates.
Example column: "created_at", example value: 2025-01-01

`Boolean` stype should be used for columns that are true or false.
Example column: "is_admin", example value: TRUE

`Categorical` stype is a bit tricky - it should be used for columns where the values inhabit a fixed universe (ideally, low cardinality) of values.
Example column: "color", example values: ["Red", "Blue", "Green"]
It is possible to have a `categorical` stype column whose dtype is integer - for example, imagine "order_status" with values [0, 1, 2].
In this case, the values are *integers* but the semantics of this column is that the numerics aren't important - these are effectively enumerations and should be treated as such.

`Text` stype corresponds to columns where the semantic meaning of the string contained within the column is important.
Example column: "user_name", example value: "Matt"

`Ignored` stype corresponds to columns that you want the model to simply ignore for the purposes of prediction. 

### Temporal Columns

Tables may also have columns marked as `temporal` columns in their metadata - these allow our model to learn causal behavior. 

A classic example of such a column might be "created_at" - this column is traditionally used when data in the row itself contains information about when the object entered the database.

Ideally, this information would live in metadata in the DBMS engine itself, but realistically, many real schemas place this lifecycle metadata information into the contents of the row itself.

We allow users to mark columns on tables as "temporal" to indicate that they should be used for the purposes of causal temporal filtering.
This guarantees that the model will not "leak" information from the future to predict properties of the past.


Another important object is the `cell` - this is a (table, row, column) tuple. Cells may be "null" or "not null".


## Preprocessing

Before training a prediction model on your relational database, you must annotate it with metadata.
Prepare one parquet file per table in your database, and a special metadata file with some information about the schema.

We need information on the semantic types of each column in the database, as well as some information about the "signal" columns worth masking and predicting.

This collection of parquet files + metadata is turned into a preprocessed representation:

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

These features are turned into pairs (sin(2 _pi_ x / period), cos(2 _pi_ x / period)) to normalize the periodic values.

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
