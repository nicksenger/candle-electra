use candle_core::{Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

pub struct Intermediate {
    dense: Linear,
    intermediate_act: crate::HiddenActLayer,
}

impl Intermediate {
    pub fn load(vb: VarBuilder, config: &crate::Config) -> Result<Self> {
        let dense = linear(config.hidden_size, config.intermediate_size, vb.pp("dense"))?;
        Ok(Self {
            dense,
            intermediate_act: crate::HiddenActLayer {
                act: config.hidden_act,
            },
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let ys = self.intermediate_act.forward(&hidden_states)?;
        Ok(ys)
    }
}
