#![recursion_limit = "131"]

//! DDP training smoke test: learn XOR with a tiny MLP.
//!
//! ## Usage
//!
//! ```sh
//! cargo run --bin train                   # single GPU (default)
//! cargo run --bin train -- --num-devices 2  # 2 GPUs, data-parallel
//! ```

use std::thread;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::collective::{self, CollectiveConfig, PeerId, ReduceOperation};
use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::relu;
use clap::Parser;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(about = "DDP training smoke test (XOR MLP)")]
struct Args {
    /// Number of local devices (GPUs) for data-parallel training.
    #[arg(long, default_value_t = 1)]
    num_devices: usize,

    /// Total training steps.
    #[arg(long, default_value_t = 5000)]
    num_steps: usize,

    /// Learning rate.
    #[arg(long, default_value_t = 1e-2)]
    lr: f64,
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Two-layer MLP: 2 → 2048 (ReLU) → 1.
#[derive(Module, Debug)]
struct XorMlp<B: Backend> {
    hidden: Linear<B>,
    output: Linear<B>,
}

impl<B: Backend> XorMlp<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            hidden: LinearConfig::new(2, 2048).init(device),
            output: LinearConfig::new(2048, 1).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = relu(self.hidden.forward(x));
        self.output.forward(x)
    }
}

// ---------------------------------------------------------------------------
// Entrypoint
// ---------------------------------------------------------------------------

fn main() {
    let args = Args::parse();

    type B = Wgpu;
    type AB = Autodiff<B>;

    let num_devices = args.num_devices;
    let num_steps = args.num_steps;
    let lr = args.lr;
    let total_samples: usize = 4;

    assert!(num_devices >= 1, "need at least one device");
    assert!(
        num_devices <= total_samples,
        "more devices ({num_devices}) than XOR samples ({total_samples})"
    );

    let devices: Vec<WgpuDevice> = if num_devices == 1 {
        vec![WgpuDevice::default()]
    } else {
        (0..num_devices).map(WgpuDevice::DiscreteGpu).collect()
    };

    let collective_config =
        CollectiveConfig::default().with_num_devices(num_devices);
    let initial_model: XorMlp<AB> = XorMlp::new(&devices[0]);

    println!("Training XOR MLP for {num_steps} steps ({num_devices} device(s))...\n");

    // Spawn one worker thread per device. When num_devices == 1 this is a
    // single thread with a trivial (self-to-self) all-reduce.
    let handles: Vec<_> = (0..num_devices)
        .map(|i| {
            let peer_id = PeerId::from(i);
            let device = devices[i].clone();
            let config = collective_config.clone();
            let model = initial_model.clone();

            let step_size = total_samples / num_devices;
            let start = i * step_size;
            let end = if i == num_devices - 1 { total_samples } else { start + step_size };

            thread::spawn(move || {
                collective::register::<B>(peer_id, device.clone(), config)
                    .expect("collective registration failed");

                let mut model: XorMlp<AB> = model.fork(&device);
                let mut optim = AdamConfig::new().init();

                let full_inputs: Tensor<AB, 2> = Tensor::from_floats(
                    [[0., 0.], [0., 1.], [1., 0.], [1., 1.]],
                    &device,
                );
                let full_targets: Tensor<AB, 2> =
                    Tensor::from_floats([[0.], [1.], [1.], [0.]], &device);

                let inputs = full_inputs.slice([start..end]);
                let targets = full_targets.slice([start..end]);

                for step in 0..num_steps {
                    let pred = model.forward(inputs.clone());
                    let diff = pred - targets.clone();
                    let loss = (diff.clone() * diff).mean();

                    if peer_id == PeerId::from(0) && step % 500 == 0 {
                        let v: f32 = loss.clone().into_scalar();
                        println!("  step {step:>5}: loss = {v:.6}");
                    }

                    let grads = loss.backward();
                    let grads = GradientsParams::from_grads(grads, &model);
                    let grads = grads
                        .all_reduce::<B>(peer_id, ReduceOperation::Mean)
                        .expect("gradient all-reduce failed");

                    model = optim.step(lr, model, grads);
                }

                collective::finish_collective::<B>(peer_id)
                    .expect("collective finish failed");

                if peer_id == PeerId::from(0) { Some(model.valid()) } else { None }
            })
        })
        .collect();

    let trained_model: XorMlp<B> = handles
        .into_iter()
        .filter_map(|h| h.join().expect("worker thread panicked"))
        .next()
        .expect("main worker did not return a model");

    let device = &devices[0];
    let test: Tensor<B, 2> =
        Tensor::from_floats([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], device);
    let pred = trained_model.forward(test);

    println!("\nFinal predictions (expected: ~0, ~1, ~1, ~0):");
    println!("{pred}");
}
