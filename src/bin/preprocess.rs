//! This is the "preprocessor" binary - it transforms raw parquet databases into a graph representation.
//!
//! ## Input Format
//!
//! Each input database is expected to be represented as a directory of parquet files, plus a "metadata.json" file.
//! We have one parquet file per table in the database.
//! The "metadata.json" file is a JSON object with roughly the following structure:
//!
//!
//! ## Output Format
//!
//! The script constructs a directed graph representation for the database.
//! Each node in the graph corresponds to a row in one of the tables.
//! The output graph has its topology stored separately from the node data.
//! Each database is mmap'able and shareable by multiple processes.
//!
//! ## Streaming Architecture
//!
//! We process one database at a time. This binary takes that database path as an argument.
//! Here are the steps we take:
//! 1. **Schema Discovery** - Read the parquet files from disk, as well as the "metadata.json" file.
//!   Discover / assign normalization statistics and column types, both data types and semantic types.
//! 2. **Vocabulary Discovery** - Stream text column values and build a shared vocabulary for all texts.
//! 3. **Embedding Generation** - Convert text values to embeddings using the embedder.
//! 4. **Cell Encoding** - Stream tables through one at a time, encoding each cell to a packed binary u32.
//!   NULL cells get encoded as the distinguished quiet NaN bit pattern - 0x7FC0_0001;
//!   
//!
//!
//! The preprocessor expec

pub fn main() {
    println!("Preprocessing data...");
}
