// TODO: Implement the single sample binary, which takes a database and a task and produces a single sample.

use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(about = "Generate a single sample from a database and a task")]
struct Args {
    /// Path to the preprocessed database directory.
    #[arg(long)]
    db_dir: PathBuf,

    /// Name of the task to sample from.
    #[arg(long)]
    task_name: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    unimplemented!()
}
