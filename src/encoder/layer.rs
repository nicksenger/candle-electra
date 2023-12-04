use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

mod attention;
mod intermediate;
mod output;

use attention::Attention;
use intermediate::Intermediate;
use output::Output;

/// ELECTRA layer
pub struct Layer {
    attention: Attention,
    intermediate: Intermediate,
    output: Output,
}

impl Layer {
    pub fn load(vb: VarBuilder, config: &crate::Config) -> Result<Self> {
        let attention = Attention::load(vb.pp("attention"), config)?;
        let intermediate = Intermediate::load(vb.pp("intermediate"), config)?;
        let output = Output::load(vb.pp("output"), config)?;
        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let attention_output = self.attention.forward(hidden_states)?;

        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self
            .output
            .forward(&intermediate_output, &attention_output)?;
        Ok(layer_output)
    }
}
