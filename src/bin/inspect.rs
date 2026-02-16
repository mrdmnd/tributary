//! Inspect a preprocessed database: dump schema, stats, graph structure,
//! tasks, and sample data in a human-readable format.
//!
//! ## Usage
//!
//! ```sh
//! cargo run --release --bin inspect -- --db-dir data/rel-stack.processed
//! cargo run --release --bin inspect -- --db-dir data/rel-stack.processed --sample-rows 10
//! ```

use std::path::PathBuf;

use clap::Parser;
use tributary::common::*;
use tributary::embedder::EMBEDDING_DIM;

#[derive(Parser, Debug)]
#[command(about = "Inspect a preprocessed database")]
struct Args {
    /// Path to the preprocessed database directory.
    #[arg(long)]
    db_dir: PathBuf,

    /// Number of sample rows to dump per table (0 to skip).
    #[arg(long, default_value_t = 5)]
    sample_rows: usize,

    /// Number of sample nodes to show graph neighborhoods for.
    #[arg(long, default_value_t = 5)]
    sample_nodes: usize,

    /// Number of sample seeds to dump per task (0 to skip).
    #[arg(long, default_value_t = 5)]
    sample_seeds: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();
    let db = Database::load(&args.db_dir)?;
    let meta = &db.metadata;

    let num_tables = meta.table_metadata.len();
    let num_cols = meta.column_metadata.len();
    let num_tasks = meta.task_metadata.len();
    let total_rows: usize = meta
        .table_metadata
        .iter()
        .map(|t| (t.row_range.1.0 - t.row_range.0.0) as usize)
        .sum();

    // ── Overview ──────────────────────────────────────────────────────────
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Database: {}", args.db_dir.display());
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Tables:     {num_tables:>10}");
    println!("║  Columns:    {num_cols:>10}");
    println!("║  Total rows: {total_rows:>10}");
    println!("║  Tasks:      {num_tasks:>10}");
    println!(
        "║  Graph:      {:>10} nodes, {:>10} edges",
        db.graph.num_nodes(),
        db.graph.num_edges()
    );
    let num_vocab = db.embeddings.num_embeddings() - num_cols;
    println!(
        "║  Embeddings: {:>10} total ({} column + {} vocab), {:>10} dim (f16)",
        db.embeddings.num_embeddings(),
        num_cols,
        num_vocab,
        EMBEDDING_DIM
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Tables & Columns ─────────────────────────────────────────────────
    for (ti, tmeta) in meta.table_metadata.iter().enumerate() {
        let row_count = tmeta.row_range.1.0 - tmeta.row_range.0.0;
        let col_start = tmeta.col_range.0.0 as usize;
        let col_end = tmeta.col_range.1.0 as usize;
        let col_count = col_end - col_start;

        let pkey_str = match tmeta.pkey_col {
            Some(pk) => meta.column_metadata[pk.0 as usize].name.clone(),
            None => "(none)".to_string(),
        };

        println!("┌─ Table {ti}: \"{}\"", tmeta.name);
        println!("│  Rows: {row_count}  Columns: {col_count}  PK: {pkey_str}");
        println!(
            "│  Row range: [{}, {})  Col range: [{}, {})",
            tmeta.row_range.0.0, tmeta.row_range.1.0, tmeta.col_range.0.0, tmeta.col_range.1.0,
        );
        println!("│");

        for ci in col_start..col_end {
            let cmeta = &meta.column_metadata[ci];
            let stats = &meta.column_stats[ci];
            let local_ci = ci - col_start;

            let fk_str = match cmeta.fkey_target_col {
                Some(fk) => {
                    let (fk_ti, _fk_local) = db.resolve_column(fk);
                    let fk_table = &meta.table_metadata[fk_ti].name;
                    let fk_col = &meta.column_metadata[fk.0 as usize].name;
                    format!(" → {fk_table}.{fk_col}")
                }
                None => String::new(),
            };

            println!("│  [{local_ci}] {} : {:?}{fk_str}", cmeta.name, cmeta.stype);
            print_stats(stats, "│      ");
        }

        // Sample rows
        if args.sample_rows > 0 {
            let tv = &db.tables[ti];
            let n = row_count as usize;
            let show = n.min(args.sample_rows);

            println!("│");
            println!("│  Sample rows (first {show} of {n}):");

            // Header
            print!("│  {:>6}", "row");
            for ci in col_start..col_end {
                let name = &meta.column_metadata[ci].name;
                let truncated: String = name.chars().take(18).collect();
                print!(" │ {truncated:>18}");
            }
            println!();

            // Separator
            print!("│  {:─>6}", "");
            for _ in col_start..col_end {
                print!("─┼─{:─>18}", "");
            }
            println!();

            // Data rows
            for row in 0..show {
                let global_row = tmeta.row_range.0.0 + row as u32;
                print!("│  {global_row:>6}");
                for ci in col_start..col_end {
                    let local_col = ci - col_start;
                    let stype = meta.column_metadata[ci].stype;
                    let cell = format_cell(tv, stype, local_col, row);
                    let truncated: String = cell.chars().take(18).collect();
                    print!(" │ {truncated:>18}");
                }
                println!();
            }
            if n > show {
                println!("│  ... ({} more rows)", n - show);
            }
        }

        println!("└──────────────────────────────────────────────────────────────");
        println!();
    }

    // ── Graph Structure ──────────────────────────────────────────────────
    println!("┌─ Graph");
    println!(
        "│  Nodes: {}  Edges: {}",
        db.graph.num_nodes(),
        db.graph.num_edges()
    );

    if db.graph.num_nodes() > 0 {
        // Degree distribution
        let n = db.graph.num_nodes();
        let mut out_degrees: Vec<u32> = Vec::with_capacity(n);
        let mut in_degrees: Vec<u32> = Vec::with_capacity(n);
        let mut max_out: u32 = 0;
        let mut max_in: u32 = 0;
        let mut sum_out: u64 = 0;
        let mut sum_in: u64 = 0;

        for i in 0..n {
            let row = RowIdx(i as u32);
            let od = db.graph.out_degree(row);
            let id = db.graph.in_degree(row);
            out_degrees.push(od);
            in_degrees.push(id);
            if od > max_out {
                max_out = od;
            }
            if id > max_in {
                max_in = id;
            }
            sum_out += od as u64;
            sum_in += id as u64;
        }

        out_degrees.sort_unstable();
        in_degrees.sort_unstable();

        let median_out = out_degrees[n / 2];
        let median_in = in_degrees[n / 2];
        let p99_out = out_degrees[(n as f64 * 0.99) as usize];
        let p99_in = in_degrees[(n as f64 * 0.99) as usize];

        println!("│");
        println!(
            "│  Out-degree:  mean={:.2}  median={}  max={}  p99={}",
            sum_out as f64 / n as f64,
            median_out,
            max_out,
            p99_out
        );
        println!(
            "│  In-degree:   mean={:.2}  median={}  max={}  p99={}",
            sum_in as f64 / n as f64,
            median_in,
            max_in,
            p99_in
        );

        // Sample neighborhoods
        if args.sample_nodes > 0 {
            println!("│");
            println!("│  Sample neighborhoods:");
            let step = if n > args.sample_nodes {
                n / args.sample_nodes
            } else {
                1
            };
            for s in 0..args.sample_nodes.min(n) {
                let row_idx = (s * step) as u32;
                let row = RowIdx(row_idx);
                let ti = db.row_table(row);
                let table_name = &meta.table_metadata[ti.0 as usize].name;
                let local = db.local_row(row);
                let out_nbrs = db.graph.outgoing_neighbors(row);
                let in_nbrs = db.graph.incoming_neighbors(row);

                println!(
                    "│    Row {row_idx} ({table_name}[{local}]): out={}, in={}",
                    out_nbrs.len(),
                    in_nbrs.len()
                );

                if !out_nbrs.is_empty() {
                    let show: Vec<String> = out_nbrs
                        .iter()
                        .take(8)
                        .map(|&r| {
                            let rt = db.row_table(RowIdx(r));
                            format!(
                                "{}[{}]",
                                meta.table_metadata[rt.0 as usize].name,
                                db.local_row(RowIdx(r))
                            )
                        })
                        .collect();
                    let suffix = if out_nbrs.len() > 8 {
                        format!(" ...+{}", out_nbrs.len() - 8)
                    } else {
                        String::new()
                    };
                    println!("│      → out: {}{suffix}", show.join(", "));
                }
                if !in_nbrs.is_empty() {
                    let show: Vec<String> = in_nbrs
                        .iter()
                        .take(8)
                        .map(|&r| {
                            let rt = db.row_table(RowIdx(r));
                            format!(
                                "{}[{}]",
                                meta.table_metadata[rt.0 as usize].name,
                                db.local_row(RowIdx(r))
                            )
                        })
                        .collect();
                    let suffix = if in_nbrs.len() > 8 {
                        format!(" ...+{}", in_nbrs.len() - 8)
                    } else {
                        String::new()
                    };
                    println!("│      ← in:  {}{suffix}", show.join(", "));
                }
            }
        }
    }

    println!("└──────────────────────────────────────────────────────────────");
    println!();

    // ── Tasks ────────────────────────────────────────────────────────────
    for (task_idx, tmeta) in meta.task_metadata.iter().enumerate() {
        let anchor_table = &meta.table_metadata[tmeta.anchor_table.0 as usize].name;
        let tv = &db.tasks[task_idx];

        println!("┌─ Task {task_idx}: \"{}\"", tmeta.name);
        println!("│  Anchor table: {anchor_table}");
        println!("│  Target stype: {:?}", tmeta.target_stype);
        println!(
            "│  Observation time: {}",
            if tmeta.has_observation_time {
                "yes"
            } else {
                "no"
            }
        );
        println!("│  Seeds: {}", tmeta.num_seeds);
        println!("│  Target stats:");
        print_stats(&tmeta.target_stats, "│    ");

        // Sample seeds
        if args.sample_seeds > 0 && tmeta.num_seeds > 0 {
            let n = tmeta.num_seeds as usize;
            let show = n.min(args.sample_seeds);

            println!("│");
            println!("│  Sample seeds (first {show} of {n}):");
            println!(
                "│  {:>6} {:>12} {:>16} {:>20}",
                "seed", "anchor_row", "obs_time", "target"
            );
            println!("│  {:─>6} {:─>12} {:─>16} {:─>20}", "", "", "", "");

            for seed in 0..show {
                let anchor = tv.anchor_row(seed);
                let anchor_ti = db.row_table(anchor);
                let anchor_table_name = &meta.table_metadata[anchor_ti.0 as usize].name;
                let anchor_local = db.local_row(anchor);

                let obs_str = match tv.observation_time(seed) {
                    Some(us) => {
                        if us == i64::MAX {
                            "MAX".to_string()
                        } else if let Some(dt) = chrono::DateTime::from_timestamp_micros(us) {
                            dt.format("%Y-%m-%d %H:%M").to_string()
                        } else {
                            format!("{us}")
                        }
                    }
                    None => "-".to_string(),
                };

                let target_str = if tv.target_is_null(seed) {
                    "NULL".to_string()
                } else {
                    format_target_cell(tv.target(), tmeta.target_stype, seed)
                };

                println!(
                    "│  {seed:>6} {anchor_table_name}[{anchor_local}]{:>width$} {obs_str:>16} {target_str:>20}",
                    "",
                    width = 12usize.saturating_sub(
                        anchor_table_name.len() + format!("[{anchor_local}]").len()
                    )
                );
            }
            if n > show {
                println!("│  ... ({} more seeds)", n - show);
            }
        }

        println!("└──────────────────────────────────────────────────────────────");
        println!();
    }

    // ── Embedding Table ──────────────────────────────────────────────────
    println!("┌─ Embedding Table (f16)");
    println!(
        "│  Total: {}  ({} column + {} vocab)  Dim: {}",
        db.embeddings.num_embeddings(),
        num_cols,
        num_vocab,
        EMBEDDING_DIM,
    );
    if db.embeddings.num_embeddings() > 0 {
        let show = db.embeddings.num_embeddings().min(5);
        println!("│  Sample embedding L2 norms:");
        for i in 0..show {
            let emb = db.embeddings.get(EmbeddingIdx(i as u32));
            let norm: f32 = emb
                .iter()
                .map(|v| v.to_f32())
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt();
            let kind = if i < num_cols { "col" } else { "vocab" };
            println!("│    [{i}] ({kind}) L2 norm = {norm:.4}");
        }
    }
    println!("└──────────────────────────────────────────────────────────────");

    Ok(())
}

/// Print column statistics with a given line prefix.
fn print_stats(stats: &ColumnStats, prefix: &str) {
    match stats {
        ColumnStats::Identifier { num_nulls } => {
            println!("{prefix}nulls: {num_nulls}");
        }
        ColumnStats::Numerical {
            num_nulls,
            min,
            max,
            mean,
            std,
        } => {
            println!(
                "{prefix}nulls: {num_nulls}  min: {min:.4}  max: {max:.4}  mean: {mean:.4}  std: {std:.4}"
            );
        }
        ColumnStats::Timestamp {
            num_nulls,
            min_us,
            max_us,
            mean_us,
            std_us,
        } => {
            let fmt = |us: i64| -> String {
                chrono::DateTime::from_timestamp_micros(us)
                    .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
                    .unwrap_or_else(|| format!("{us}μs"))
            };
            println!(
                "{prefix}nulls: {num_nulls}  min: {}  max: {}",
                fmt(*min_us),
                fmt(*max_us)
            );
            println!("{prefix}mean: {mean_us:.0}μs  std: {std_us:.0}μs");
        }
        ColumnStats::Boolean {
            num_nulls,
            num_true,
            num_false,
        } => {
            let total = num_true + num_false;
            let pct_true = if total > 0 {
                *num_true as f64 / total as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "{prefix}nulls: {num_nulls}  true: {num_true} ({pct_true:.1}%)  false: {num_false}"
            );
        }
        ColumnStats::Categorical {
            num_nulls,
            categories,
        } => {
            let n = categories.len();
            let preview: Vec<&str> = categories.iter().take(10).map(|s| s.as_str()).collect();
            let suffix = if n > 10 {
                format!(" ...+{}", n - 10)
            } else {
                String::new()
            };
            println!(
                "{prefix}nulls: {num_nulls}  cardinality: {n}  values: [{}]{suffix}",
                preview.join(", ")
            );
        }
        ColumnStats::Text { num_nulls } => {
            println!("{prefix}nulls: {num_nulls}");
        }
        ColumnStats::Ignored => {
            println!("{prefix}(ignored)");
        }
    }
}

/// Format a single cell value from a TableView for display.
fn format_cell(tv: &TableView, stype: SemanticType, col: usize, row: usize) -> String {
    if tv.is_null(col, row) {
        return "NULL".to_string();
    }
    match stype {
        SemanticType::Identifier => {
            if tv.identifier_present(col, row) {
                "✓".to_string()
            } else {
                "∅".to_string()
            }
        }
        SemanticType::Numerical => {
            format!("{:.4}", tv.numerical(col, row))
        }
        SemanticType::Timestamp => {
            let ts = tv.timestamp(col, row);
            // Show the z-scored epoch value (last element)
            format!("z={:.3}", ts[TIMESTAMP_DIM - 1])
        }
        SemanticType::Boolean => {
            if tv.boolean(col, row) {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        SemanticType::Categorical | SemanticType::Text => {
            let idx = tv.embedding_idx(col, row);
            format!("emb[{}]", idx.0)
        }
        SemanticType::Ignored => "(ignored)".to_string(),
    }
}

/// Format a target cell value from a TaskView's target ColumnSlice.
fn format_target_cell(target: &ColumnSlice, stype: SemanticType, seed: usize) -> String {
    match (stype, target) {
        (SemanticType::Numerical, ColumnSlice::Numerical { values, .. }) => {
            format!("{:.4}", values[seed])
        }
        (SemanticType::Boolean, ColumnSlice::Boolean { bits, .. }) => {
            if bit_is_set(bits, seed) {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        (SemanticType::Categorical, ColumnSlice::Embedded { indices, .. }) => {
            format!("emb[{}]", indices[seed])
        }
        (SemanticType::Timestamp, ColumnSlice::Timestamp { values, .. }) => {
            let ts = &values[seed * TIMESTAMP_DIM..(seed + 1) * TIMESTAMP_DIM];
            format!("z={:.3}", ts[TIMESTAMP_DIM - 1])
        }
        _ => "?".to_string(),
    }
}
