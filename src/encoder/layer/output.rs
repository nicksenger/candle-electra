use candle_core::{Module, Result, Tensor};
use candle_nn::{layer_norm, linear, Dropout, LayerNorm, Linear, VarBuilder};

pub struct Output {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl Output {
    pub fn load(vb: VarBuilder, config: &crate::Config) -> Result<Self> {
        let dense = linear(config.intermediate_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob as f32);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states, false)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}
