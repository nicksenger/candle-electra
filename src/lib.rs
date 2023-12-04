use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

mod config;
mod embeddings;
mod encoder;

pub use config::{Config, HiddenAct, PositionEmbeddingType};
use embeddings::Embeddings;
use encoder::Encoder;

pub const DTYPE: DType = DType::F32;

pub struct Model {
    embeddings: Embeddings,
    encoder: Encoder,
    pub device: Device,
}

impl Model {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let (embeddings, encoder) = match (
            Embeddings::load(vb.pp("embeddings"), config),
            Encoder::load(vb.pp("encoder"), config),
        ) {
            (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
            (Err(err), _) | (_, Err(err)) => {
                let Some((embeddings, encoder)) =
                    config.model_type.as_ref().and_then(|model_type| {
                        Embeddings::load(vb.pp(&format!("{model_type}.embeddings")), config)
                            .into_iter()
                            .zip(Encoder::load(
                                vb.pp(&format!("{model_type}.encoder")),
                                config,
                            ))
                            .next()
                    })
                else {
                    return Err(err);
                };

                (embeddings, encoder)
            }
        };

        Ok(Self {
            embeddings,
            encoder,
            device: vb.device().clone(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let embedding_output = self.embeddings.forward(input_ids, token_type_ids)?;
        let sequence_output = self.encoder.forward(&embedding_output)?;

        Ok(sequence_output)
    }
}

pub struct Pooler {
    dense: Linear,
    activation: HiddenActLayer,
}

impl Pooler {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        Ok(Self {
            dense,
            activation: HiddenActLayer::new(HiddenAct::Tanh),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let first_token_sensor = hidden_states.i((.., 0))?;
        let pooled_output = self.dense.forward(&first_token_sensor)?;
        let pooled_output = self.activation.forward(&pooled_output)?;

        Ok(pooled_output)
    }
}

struct HiddenActLayer {
    act: HiddenAct,
}

impl HiddenActLayer {
    fn new(act: HiddenAct) -> Self {
        Self { act }
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self.act {
            HiddenAct::Gelu => xs.gelu_erf(),
            HiddenAct::Relu => xs.relu(),
            HiddenAct::Tanh => xs.tanh(),
        }
    }
}
