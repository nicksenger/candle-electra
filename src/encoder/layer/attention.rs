use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

mod self_attention;
mod self_output;

use self_attention::SelfAttention;
use self_output::SelfOutput;

pub struct Attention {
    self_attention: SelfAttention,
    self_output: SelfOutput,
}

impl Attention {
    pub fn load(vb: VarBuilder, config: &crate::Config) -> Result<Self> {
        let self_attention = SelfAttention::load(vb.pp("self"), config)?;
        let self_output = SelfOutput::load(vb.pp("output"), config)?;
        Ok(Self {
            self_attention,
            self_output,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let self_outputs = self.self_attention.forward(hidden_states)?;
        let attention_output = self.self_output.forward(&self_outputs, hidden_states)?;
        Ok(attention_output)
    }
}
