//! Generate a static HTML report visualizing a single sampled batch.
//!
//! Produces a self-contained HTML file showing all cells, their encoded
//! values, tensor summaries, and attention masks rendered as heatmaps.
//!
//! ## Usage
//!
//! ```sh
//! cargo run --release --bin inspect_batch -- --data-dir data --dataset rel-stack
//! ```

use std::fmt::Write as FmtWrite;
use std::path::PathBuf;

use clap::Parser;
use headwater::batch::RawBatch;
use headwater::common::{ColumnIdx, Database, TIMESTAMP_DIM};
use headwater::sampler::{Sampler, SamplerConfig};
use tracing::info;

const BATCH_SIZE: u32 = 1;
const SEQ_LEN: u32 = 64;

#[derive(Parser, Debug)]
#[command(about = "Generate an HTML report visualizing a single sample batch (B=1, S=64)")]
struct Args {
    /// Dataset name (e.g. "rel-stack").
    #[arg(long)]
    dataset: String,
    /// Top-level data directory (contains raw/, metadata/, processed/).
    #[arg(long)]
    data_dir: PathBuf,
    /// Output HTML file path.
    #[arg(long, default_value = "batch_inspect.html")]
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    let db_path = args.data_dir.join("processed").join(&args.dataset);
    let config = SamplerConfig {
        db_path: db_path.to_string_lossy().to_string(),
        batch_size: BATCH_SIZE,
        sequence_length: SEQ_LEN,
        ..Default::default()
    };

    info!(
        "Sampling one batch (B={BATCH_SIZE}, S={SEQ_LEN}) from {}",
        args.dataset
    );
    let mut sampler = Sampler::new(config)?;
    let batch = sampler.next_train_batch()?;
    let db = sampler.database();

    let html = build_html(&batch, db, &args.dataset);
    std::fs::write(&args.output, &html)?;
    info!("Wrote {}", args.output.display());

    sampler.shutdown();
    Ok(())
}

// ============================================================================
// HTML orchestration
// ============================================================================

fn build_html(batch: &RawBatch, db: &Database, dataset: &str) -> String {
    let s = SEQ_LEN as usize;
    let mut html = String::with_capacity(200_000);
    push_head(&mut html, dataset);
    push_overview(&mut html, batch, db, s);
    push_cells(&mut html, batch, db, s);
    push_tensors(&mut html, batch, s);
    push_masks(&mut html, batch, s);
    html.push_str("</body></html>\n");
    html
}

fn push_head(html: &mut String, dataset: &str) {
    html.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
    let _ = writeln!(html, "<title>Batch Inspector — {dataset}</title>");
    html.push_str("<style>\n");
    push_css(html);
    html.push_str("</style>\n</head>\n<body>\n");
    html.push_str("<h1>Batch Inspector</h1>\n");
    let _ = writeln!(
        html,
        "<p class=\"sub\">Dataset: <strong>{dataset}</strong> · B={BATCH_SIZE}, S={SEQ_LEN}</p>"
    );
}

// ============================================================================
// Overview cards
// ============================================================================

fn push_overview(html: &mut String, batch: &RawBatch, db: &Database, s: usize) {
    let active = count_eq(&batch.is_padding[..s], 0);
    let padding = s - active;
    let nulls = count_eq(&batch.is_null[..s], 1);
    html.push_str("<div class=\"overview\">\n");
    ov(html, "Task", resolve_task_name(batch, db));
    ov(html, "Target Type", stype_name(batch.target_stype));
    ov(html, "Active Cells", &active.to_string());
    ov(html, "Padding Cells", &padding.to_string());
    ov(html, "Null Cells", &nulls.to_string());
    ov(html, "Target Pos", "0");
    ov(html, "Unique Texts", &batch.num_unique_texts.to_string());
    html.push_str("</div>\n");
}

fn ov(html: &mut String, label: &str, value: &str) {
    let _ = writeln!(
        html,
        "<div class=\"ov\"><div class=\"lbl\">{label}</div><div class=\"val\">{value}</div></div>"
    );
}

// ============================================================================
// Cell table
// ============================================================================

fn push_cells(html: &mut String, batch: &RawBatch, db: &Database, s: usize) {
    html.push_str("<h2>Cells</h2>\n<table class=\"cells\"><thead><tr>");
    html.push_str("<th>Pos</th><th>Table</th><th>Column</th><th>Type</th>");
    html.push_str("<th>Encoded Value</th><th>Null</th><th>Target</th>");
    html.push_str("</tr></thead><tbody>\n");
    for pos in 0..s {
        push_cell_row(html, batch, db, pos);
    }
    html.push_str("</tbody></table>\n");
}

fn push_cell_row(html: &mut String, batch: &RawBatch, db: &Database, pos: usize) {
    if batch.is_padding[pos] == 1 {
        let _ = writeln!(
            html,
            "<tr class=\"pad\"><td>{pos}</td>\
             <td colspan=\"6\" class=\"pad-cell\">PADDING</td></tr>"
        );
        return;
    }
    let is_target = pos == 0;
    let cls = if is_target { " class=\"tgt\"" } else { "" };
    let stype = batch.semantic_types[pos];
    let col_id = batch.column_embed_ids[pos] as usize;
    let (table_name, col_name) = resolve_col(db, col_id);

    let _ = writeln!(
        html,
        "<tr{cls}><td>{pos}</td><td>{table_name}</td>\
         <td title=\"col_id={col_id}\">{col_name}</td>\
         <td class=\"{sc}\">{sn}</td><td>{val}</td>\
         <td>{null}</td><td>{tgt}</td></tr>",
        sc = stype_css_class(stype),
        sn = stype_name(stype),
        val = cell_value_html(batch, pos),
        null = if batch.is_null[pos] == 1 { "✓" } else { "" },
        tgt = if is_target { "🎯" } else { "" },
    );
}

fn cell_value_html(batch: &RawBatch, pos: usize) -> String {
    if batch.is_null[pos] == 1 {
        return "<em>NULL</em>".to_string();
    }
    match batch.semantic_types[pos] {
        0 => "—".to_string(),
        1 => format!("{:.4}", batch.numeric_values[pos]),
        2 => {
            let base = pos * TIMESTAMP_DIM;
            format!("z={:.3}", batch.timestamp_values[base + TIMESTAMP_DIM - 1])
        }
        3 => (if batch.bool_values[pos] == 1 {
            "true"
        } else {
            "false"
        })
        .to_string(),
        4 => format!("cat[{}]", batch.categorical_embed_ids[pos]),
        5 => format!("text[{}]", batch.text_embed_ids[pos]),
        _ => "—".to_string(),
    }
}

// ============================================================================
// Tensor summaries
// ============================================================================

fn push_tensors(html: &mut String, batch: &RawBatch, s: usize) {
    html.push_str("<h2>Tensors</h2>\n<div class=\"tgrid\">\n");
    push_tensor_u8(html, "semantic_types", s, &batch.semantic_types[..s]);
    push_tensor_f32(html, "numeric_values", s, &batch.numeric_values[..s]);
    push_tensor_ts(
        html,
        "timestamp_values",
        s,
        &batch.timestamp_values[..s * TIMESTAMP_DIM],
    );
    push_tensor_u8(html, "bool_values", s, &batch.bool_values[..s]);
    push_tensor_u32(
        html,
        "categorical_embed_ids",
        s,
        &batch.categorical_embed_ids[..s],
    );
    push_tensor_u32(html, "column_embed_ids", s, &batch.column_embed_ids[..s]);
    push_tensor_u32(html, "text_embed_ids", s, &batch.text_embed_ids[..s]);
    push_tensor_u8(html, "is_null", s, &batch.is_null[..s]);
    push_tensor_u8(html, "is_padding", s, &batch.is_padding[..s]);
    push_csr_tensor(
        html,
        "outbound_csr",
        &batch.outbound_csr_row_ptr,
        &batch.outbound_csr_col_idx,
    );
    push_csr_tensor(
        html,
        "inbound_csr",
        &batch.inbound_csr_row_ptr,
        &batch.inbound_csr_col_idx,
    );
    push_csr_tensor(
        html,
        "column_csr",
        &batch.column_csr_row_ptr,
        &batch.column_csr_col_idx,
    );
    if batch.num_unique_texts > 0 {
        let _ = writeln!(
            html,
            "<div class=\"tc\"><div class=\"tn\">text_batch_embeddings</div>\
             <div class=\"ts\">[{}, 256] f16</div></div>",
            batch.num_unique_texts,
        );
    }
    html.push_str("</div>\n");
}

fn push_tensor_u8(html: &mut String, name: &str, s: usize, data: &[u8]) {
    let vals: String = data
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(
        html,
        "<div class=\"tc\"><div class=\"tn\">{name}</div>\
         <div class=\"ts\">[1, {s}] u8</div>\
         <details><summary>values</summary><pre>[{vals}]</pre></details></div>"
    );
}

fn push_tensor_f32(html: &mut String, name: &str, s: usize, data: &[f32]) {
    let nz = data.iter().filter(|&&v| v != 0.0).count();
    let vals: String = data
        .iter()
        .map(|v| format!("{v:.4}"))
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(
        html,
        "<div class=\"tc\"><div class=\"tn\">{name}</div>\
         <div class=\"ts\">[1, {s}] f32 · non-zero: {nz}</div>\
         <details><summary>values</summary><pre>[{vals}]</pre></details></div>"
    );
}

fn push_tensor_ts(html: &mut String, name: &str, s: usize, data: &[f32]) {
    let zs: String = (0..s)
        .map(|i| format!("{:.3}", data[i * TIMESTAMP_DIM + TIMESTAMP_DIM - 1]))
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(
        html,
        "<div class=\"tc\"><div class=\"tn\">{name}</div>\
         <div class=\"ts\">[1, {s}, {TIMESTAMP_DIM}] f32 · {} total</div>\
         <details><summary>z-score epochs (dim 14)</summary><pre>[{zs}]</pre></details></div>",
        data.len(),
    );
}

fn push_tensor_u32(html: &mut String, name: &str, s: usize, data: &[u32]) {
    let vals: String = data
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(
        html,
        "<div class=\"tc\"><div class=\"tn\">{name}</div>\
         <div class=\"ts\">[1, {s}] u32</div>\
         <details><summary>values</summary><pre>[{vals}]</pre></details></div>"
    );
}

fn push_csr_tensor(html: &mut String, name: &str, row_ptr: &[u32], col_idx: &[u16]) {
    let _ = writeln!(
        html,
        "<div class=\"tc\"><div class=\"tn\">{name}</div>\
         <div class=\"ts\">row_ptr[{}], col_idx[{}] · nnz={}</div></div>",
        row_ptr.len(),
        col_idx.len(),
        col_idx.len(),
    );
}

// ============================================================================
// Attention mask heatmaps
// ============================================================================

fn push_masks(html: &mut String, batch: &RawBatch, s: usize) {
    html.push_str("<h2>Attention Masks</h2>\n<div class=\"masks\">\n");

    let outbound = csr_to_dense(
        &batch.outbound_csr_row_ptr,
        &batch.outbound_csr_seq_offsets,
        &batch.outbound_csr_col_idx,
        s,
    );
    push_mask_box(
        html,
        "Outbound (FK→)",
        "Same-row + outgoing FK neighbors",
        &outbound,
        s,
    );

    let inbound = csr_to_dense(
        &batch.inbound_csr_row_ptr,
        &batch.inbound_csr_seq_offsets,
        &batch.inbound_csr_col_idx,
        s,
    );
    push_mask_box(html, "Inbound (←FK)", "Incoming FK neighbors", &inbound, s);

    let column = csr_to_dense(
        &batch.column_csr_row_ptr,
        &batch.column_csr_seq_offsets,
        &batch.column_csr_col_idx,
        s,
    );
    push_mask_box(
        html,
        "Column (same col)",
        "Same column across different rows",
        &column,
        s,
    );

    html.push_str("</div>\n");
}

fn push_mask_box(html: &mut String, title: &str, desc: &str, mask: &[bool], s: usize) {
    let nnz: usize = mask.iter().filter(|&&v| v).count();
    let density = (nnz as f64) / (s * s) as f64 * 100.0;
    let _ = writeln!(
        html,
        "<div class=\"mbox\"><h3>{title}</h3>\
         <p class=\"mdesc\">{desc} · {nnz} filled ({density:.1}%)</p>"
    );
    push_mask_svg(html, mask, s);
    html.push_str("</div>\n");
}

fn push_mask_svg(html: &mut String, mask: &[bool], s: usize) {
    let cell = 8usize;
    let margin = 32usize;
    let dim = s * cell;
    let total = dim + margin;
    let _ = writeln!(
        html,
        "<svg width=\"{total}\" height=\"{total}\" xmlns=\"http://www.w3.org/2000/svg\">"
    );
    let _ = writeln!(
        html,
        "<rect x=\"{margin}\" y=\"{margin}\" width=\"{dim}\" height=\"{dim}\" fill=\"#f1f5f9\"/>"
    );
    push_mask_filled_cells(html, mask, s, cell, margin);
    push_mask_grid(html, s, cell, margin, dim);
    push_mask_axis_labels(html, s, cell, margin);
    html.push_str("</svg>\n");
}

fn push_mask_filled_cells(html: &mut String, mask: &[bool], s: usize, cell: usize, margin: usize) {
    for row in 0..s {
        for col in 0..s {
            if mask[row * s + col] {
                let x = margin + col * cell;
                let y = margin + row * cell;
                let _ = writeln!(
                    html,
                    "<rect x=\"{x}\" y=\"{y}\" width=\"{cell}\" height=\"{cell}\" \
                     fill=\"#2563eb\"><title>({row},{col})</title></rect>"
                );
            }
        }
    }
}

fn push_mask_grid(html: &mut String, s: usize, cell: usize, margin: usize, dim: usize) {
    let end = margin + dim;
    for i in (0..=s).step_by(8) {
        let p = margin + i * cell;
        let _ = write!(
            html,
            "<line x1=\"{p}\" y1=\"{margin}\" x2=\"{p}\" y2=\"{end}\" \
             stroke=\"#94a3b8\" stroke-width=\"0.5\"/>\n\
             <line x1=\"{margin}\" y1=\"{p}\" x2=\"{end}\" y2=\"{p}\" \
             stroke=\"#94a3b8\" stroke-width=\"0.5\"/>\n"
        );
    }
}

fn push_mask_axis_labels(html: &mut String, s: usize, cell: usize, margin: usize) {
    for i in (0..s).step_by(8) {
        let x = margin + i * cell + cell / 2;
        let y = margin + i * cell + cell / 2 + 3;
        let _ = writeln!(
            html,
            "<text x=\"{x}\" y=\"{}\" font-size=\"9\" text-anchor=\"middle\" \
             fill=\"#64748b\">{i}</text>",
            margin.saturating_sub(5),
        );
        let _ = writeln!(
            html,
            "<text x=\"{}\" y=\"{y}\" font-size=\"9\" text-anchor=\"end\" \
             fill=\"#64748b\">{i}</text>",
            margin.saturating_sub(5),
        );
    }
}

// ============================================================================
// CSS
// ============================================================================

fn push_css(css: &mut String) {
    css.push_str(concat!(
        "*{margin:0;padding:0;box-sizing:border-box}\n",
        "body{font-family:system-ui,-apple-system,sans-serif;background:#f8fafc;",
        "color:#1e293b;padding:2rem;max-width:1600px;margin:0 auto}\n",
        "h1{font-size:1.5rem;margin-bottom:.25rem}\n",
        "h2{font-size:1.2rem;margin:2rem 0 .75rem;border-bottom:2px solid #e2e8f0;",
        "padding-bottom:.5rem}\n",
        "h3{font-size:1rem;margin:0 0 .25rem;color:#334155}\n",
        ".sub{color:#64748b;margin-bottom:1.5rem;font-size:.9rem}\n",
        ".overview{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));",
        "gap:.75rem;margin-bottom:1.5rem}\n",
        ".ov{background:#fff;border-radius:8px;padding:.75rem 1rem;",
        "box-shadow:0 1px 3px rgba(0,0,0,.08)}\n",
        ".ov .lbl{font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.04em}\n",
        ".ov .val{font-size:1.15rem;font-weight:600;margin-top:.15rem}\n",
    ));
    push_css_table(css);
    push_css_tensors(css);
    push_css_masks(css);
}

fn push_css_table(css: &mut String) {
    css.push_str(concat!(
        "table.cells{width:100%;border-collapse:collapse;font-size:.82rem;background:#fff;",
        "border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.08)}\n",
        "table.cells th{background:#1e293b;color:#fff;padding:.45rem .6rem;text-align:left;",
        "font-weight:500;position:sticky;top:0}\n",
        "table.cells td{padding:.35rem .6rem;border-bottom:1px solid #f1f5f9;",
        "font-variant-numeric:tabular-nums}\n",
        "table.cells tr:hover{background:#f0f9ff}\n",
        "table.cells tr.pad{opacity:.35}\n",
        "table.cells tr.tgt{background:#fef3c7}\n",
        ".pad-cell{text-align:center;color:#94a3b8;font-style:italic}\n",
        ".stype-id{color:#6366f1;font-weight:500}\n",
        ".stype-num{color:#059669;font-weight:500}\n",
        ".stype-ts{color:#d97706;font-weight:500}\n",
        ".stype-bool{color:#dc2626;font-weight:500}\n",
        ".stype-cat{color:#7c3aed;font-weight:500}\n",
        ".stype-text{color:#0891b2;font-weight:500}\n",
        ".stype-ign{color:#94a3b8}\n",
    ));
}

fn push_css_tensors(css: &mut String) {
    css.push_str(concat!(
        ".tgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:.75rem}\n",
        ".tc{background:#fff;border-radius:8px;padding:.75rem 1rem;",
        "box-shadow:0 1px 3px rgba(0,0,0,.08)}\n",
        ".tn{font-weight:600;font-family:ui-monospace,monospace;font-size:.82rem}\n",
        ".ts{color:#64748b;font-size:.78rem;margin-top:.15rem}\n",
        "details{margin-top:.35rem}\n",
        "details summary{cursor:pointer;font-size:.78rem;color:#2563eb}\n",
        "details pre{font-size:.72rem;background:#f1f5f9;padding:.5rem;border-radius:4px;",
        "overflow-x:auto;margin-top:.25rem;white-space:pre-wrap;word-break:break-all}\n",
    ));
}

fn push_css_masks(css: &mut String) {
    css.push_str(concat!(
        ".masks{display:flex;flex-wrap:wrap;gap:2rem;align-items:flex-start}\n",
        ".mbox{text-align:center}\n",
        ".mdesc{font-size:.78rem;color:#64748b;margin-bottom:.5rem}\n",
        "svg rect:hover{opacity:.7}\n",
    ));
}

// ============================================================================
// Helpers
// ============================================================================

fn stype_name(raw: u8) -> &'static str {
    match raw {
        0 => "Identifier",
        1 => "Numerical",
        2 => "Timestamp",
        3 => "Boolean",
        4 => "Categorical",
        5 => "Text",
        6 => "Ignored",
        _ => "?",
    }
}

fn stype_css_class(raw: u8) -> &'static str {
    match raw {
        0 => "stype-id",
        1 => "stype-num",
        2 => "stype-ts",
        3 => "stype-bool",
        4 => "stype-cat",
        5 => "stype-text",
        _ => "stype-ign",
    }
}

fn resolve_task_name<'a>(batch: &RawBatch, db: &'a Database) -> &'a str {
    let idx = batch.task_idx as usize;
    if idx < db.metadata.task_metadata.len() {
        &db.metadata.task_metadata[idx].name
    } else {
        "?"
    }
}

fn resolve_col(db: &Database, global_col_id: usize) -> (&str, &str) {
    if global_col_id >= db.metadata.column_metadata.len() {
        return ("?", "?");
    }
    let (ti, _) = db.resolve_column(ColumnIdx(global_col_id as u32));
    let table_name = &db.metadata.table_metadata[ti].name;
    let col_name = &db.metadata.column_metadata[global_col_id].name;
    (table_name, col_name)
}

fn count_eq(data: &[u8], target: u8) -> usize {
    data.iter().filter(|&&v| v == target).count()
}

fn csr_to_dense(row_ptr: &[u32], seq_offsets: &[u32], col_idx: &[u16], s: usize) -> Vec<bool> {
    let base = seq_offsets[0] as usize;
    let mut mask = vec![false; s * s];
    for i in 0..s {
        let lo = row_ptr[i] as usize + base;
        let hi = row_ptr[i + 1] as usize + base;
        for &j in &col_idx[lo..hi] {
            if (j as usize) < s {
                mask[i * s + j as usize] = true;
            }
        }
    }
    mask
}
