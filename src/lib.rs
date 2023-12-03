use candle_core::Device;
use candle_nn::Linear;

mod config;
mod embeddings;
mod encoder;

pub use config::{Config, HiddenAct};
use embeddings::Embeddings;
use encoder::Encoder;

struct Model {
    embeddings: Embeddings,
    encoder: Encoder,
    pub device: Device,
}

pub struct Pooler {
    dense: Linear,
    activation: HiddenActLayer,
}

struct HiddenActLayer {
    act: HiddenAct,
}
