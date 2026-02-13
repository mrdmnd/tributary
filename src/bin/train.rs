#![recursion_limit = "131"]

use burn::{
    backend::{Autodiff, Cuda},
    data::dataset::Dataset,
    optim::AdamConfig,
};

use tributary::{
    model::ModelConfig,
    training::{self, TrainingConfig},
};

pub fn main() {
    type MyBackend = Cuda<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let artifact_dir = "/tmp/tributary-train";

    let model_config = ModelConfig::new(10, 512);
    let optimizer_config = AdamConfig::new();

    let train_config = TrainingConfig::new(model_config, optimizer_config);
    let device = burn::backend::cuda::CudaDevice::default();

    // Run training
    training::train::<MyAutodiffBackend>(artifact_dir, train_config, device.clone())
}
