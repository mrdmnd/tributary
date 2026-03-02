//! Deterministic train/val/test seed assignment and per-rank sharding.
//!
//! Each task's seeds are hash-assigned to a split, then round-robin sharded
//! across DDP ranks so no two ranks share a seed.

use tracing::info;

use crate::common::{Database, RowIdx, TaskIdx};
use crate::sampler::SamplerConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    Train,
    Val,
    Test,
}

/// Deterministic hash-based train/val/test assignment per (task, anchor_row, split_seed).
fn assign_split(
    task_idx: TaskIdx,
    anchor_row: RowIdx,
    split_seed: u64,
    train_ratio: f32,
    val_ratio: f32,
) -> Split {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    use std::hash::{Hash, Hasher};
    task_idx.0.hash(&mut hasher);
    anchor_row.0.hash(&mut hasher);
    split_seed.hash(&mut hasher);
    let bucket = (hasher.finish() % 1000) as f32;
    let train_thresh = train_ratio * 1000.0;
    let val_thresh = (train_ratio + val_ratio) * 1000.0;
    if bucket < train_thresh {
        Split::Train
    } else if bucket < val_thresh {
        Split::Val
    } else {
        Split::Test
    }
}

/// Owns the per-task, per-split seed indices visible to this DDP rank.
///
/// Each task in the database has many seeds (anchor rows that start a BFS sample).
/// At construction, every seed is deterministically assigned to train/val/test via
/// `assign_split`, then round-robin sharded across ranks so no two ranks share a seed.
/// The sampler draws from these pre-partitioned lists at sample time, picking a random
/// task then a random seed within that task for the requested split.
pub(crate) struct SeedManager {
    train_seeds: Vec<Vec<usize>>,
    val_seeds: Vec<Vec<usize>>,
    test_seeds: Vec<Vec<usize>>,
    train_tasks_with_seeds: Vec<usize>,
    val_tasks_with_seeds: Vec<usize>,
    test_tasks_with_seeds: Vec<usize>,
}

impl SeedManager {
    pub(crate) fn new(db: &Database, config: &SamplerConfig) -> Self {
        let num_tasks = db.metadata.task_metadata.len();
        let mut train_seeds = vec![Vec::new(); num_tasks];
        let mut val_seeds = vec![Vec::new(); num_tasks];
        let mut test_seeds = vec![Vec::new(); num_tasks];

        for ti in 0..num_tasks {
            let task_view = db.task(TaskIdx(ti as u32));
            let num_seeds = task_view.num_seeds();
            let mut train_for_task = Vec::new();
            let mut val_for_task = Vec::new();
            let mut test_for_task = Vec::new();

            for seed_idx in 0..num_seeds {
                let anchor_row = task_view.anchor_row(seed_idx);
                let split = assign_split(
                    TaskIdx(ti as u32),
                    anchor_row,
                    config.split_seed,
                    config.split_ratios.0,
                    config.split_ratios.1,
                );
                match split {
                    Split::Train => train_for_task.push(seed_idx),
                    Split::Val => val_for_task.push(seed_idx),
                    Split::Test => test_for_task.push(seed_idx),
                }
            }

            train_seeds[ti] = shard_by_rank(train_for_task, config);
            val_seeds[ti] = shard_by_rank(val_for_task, config);
            test_seeds[ti] = shard_by_rank(test_for_task, config);
        }

        info!(
            "SeedManager: {} tasks, train seeds/rank: [{}], val seeds/rank: [{}]",
            num_tasks,
            format_seed_counts(&train_seeds),
            format_seed_counts(&val_seeds),
        );

        let train_tasks_with_seeds = tasks_with_nonempty_seeds(&train_seeds);
        let val_tasks_with_seeds = tasks_with_nonempty_seeds(&val_seeds);
        let test_tasks_with_seeds = tasks_with_nonempty_seeds(&test_seeds);

        Self {
            train_seeds,
            val_seeds,
            test_seeds,
            train_tasks_with_seeds,
            val_tasks_with_seeds,
            test_tasks_with_seeds,
        }
    }

    pub(crate) fn seeds_for_split(&self, split: Split) -> &[Vec<usize>] {
        match split {
            Split::Train => &self.train_seeds,
            Split::Val => &self.val_seeds,
            Split::Test => &self.test_seeds,
        }
    }

    pub(crate) fn tasks_with_seeds(&self, split: Split) -> &[usize] {
        match split {
            Split::Train => &self.train_tasks_with_seeds,
            Split::Val => &self.val_tasks_with_seeds,
            Split::Test => &self.test_tasks_with_seeds,
        }
    }

    /// Total number of seeds across all tasks for a given split.
    pub(crate) fn total_seeds(&self, split: Split) -> usize {
        self.seeds_for_split(split).iter().map(|s| s.len()).sum()
    }
}

/// Round-robin shard a list of seed indices by rank.
fn shard_by_rank(seeds: Vec<usize>, config: &SamplerConfig) -> Vec<usize> {
    seeds
        .into_iter()
        .enumerate()
        .filter(|(i, _)| (*i as u32) % config.world_size == config.rank)
        .map(|(_, s)| s)
        .collect()
}

fn format_seed_counts(seeds: &[Vec<usize>]) -> String {
    seeds
        .iter()
        .map(|s| s.len().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn tasks_with_nonempty_seeds(seeds: &[Vec<usize>]) -> Vec<usize> {
    seeds
        .iter()
        .enumerate()
        .filter(|(_, s)| !s.is_empty())
        .map(|(ti, _)| ti)
        .collect()
}
