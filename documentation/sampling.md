# How the Context Window Sampling Works

> **Note:** This document describes the *intended design* for the sampler. The implementation
> in `headwater/src/sampler.rs` is not yet written.

The goal is to build a "context window" of cells from a relational database that are relevant to predicting a single masked (unknown) cell. Think of it like: given a specific row you care about (the seed row), go explore the database graph to find the most useful nearby information, up to a budget of L cells.

## The Core Idea

The database is a graph. Rows are nodes, and foreign-key relationships are edges. There are two kinds of edges:

- F->P (foreign-to-primary): "This transaction belongs to that user." These point from a child row to its parent. There are few of these per row (bounded by the number of foreign key columns in the table).
- P->F (primary-to-foreign): "This user has these transactions." These point from a parent to its children. There can be tons of these (a user might have thousands of transactions).

## The Algorithm, Step by Step

Start at the seed row from the task table. Put it in a "frontier". The collected cells set C starts empty.
Pick the next row to explore from the frontier, using this priority:

Always prefer rows reached via F->P links (i.e., parent rows).
If any are in the frontier, pick one of those first.
The intuition is that parent rows are almost always important — e.g., if you're looking at a transaction, you definitely want the user and product info.

Otherwise, pick a row that is the fewest hops away from the seed row (closest first, like normal BFS).
If there are ties, pick randomly.

Visit that row::
- Add all of its non-ignored feature cells to your context C.
- Add all of its F->P neighbors (parents) to the frontier — unconditionally, no limit.
- For its P->F neighbors (children), filter out any with a timestamp after the seed row (to prevent temporal leakage — you can't use future data to predict the present). Then randomly subsample at most w of them and add those to the frontier. This width bound w prevents any single popular entity from flooding your context with thousands of children.

Repeat until you've collected L cells or the frontier is empty.

## Why the Asymmetry Between F->P and P->F?

F->P (parents) are always followed greedily because there are only a few per row and they tend to carry critical features (e.g., a transaction's user profile, a transaction's product details).
P->F (children) are subsampled because there can be unboundedly many, and the useful signal from children tends to come from aggregation.
We get diminishing returns from including every single child row.

## Implementation Details (processes, threading, parallelism)

Sampling will be implemented as a rust library.
The intention is that the library will be callable from python code via PyO3.

Each JAX process will have its own corresponding rust sampler process.
The sampler processes will SHARE MEMORY on the underlying graph database, via memory-mapped files.