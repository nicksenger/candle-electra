use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

mod layer;

use layer::Layer;

pub struct Encoder {
    layers: Vec<Layer>,
}

impl Encoder {
    pub fn load(vb: VarBuilder, config: &crate::Config) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| Layer::load(vb.pp(&format!("layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;

        Ok(Encoder { layers })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states)?;
        }

        Ok(hidden_states)
    }
}
