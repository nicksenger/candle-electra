use std::collections::HashMap;

use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
enum PositionEmbeddingType {
    #[default]
    Absolute,
}

/// Configuration class to store the configuration of a [`Model`](crate::Model)
#[derive(Deserialize)]
pub struct Config {
    /// Vocabulary size of the ELECTRA model. Defines the number of different tokens that can be
    /// represented by the`input_ids` passed when calling [`Model::forward`](crate::Model::forward)
    pub(crate) vocab_size: usize,
    /// Dimensionality of the [`Encoder`](crate::encoder::Encoder) layers and the pooler layer.
    pub(crate) hidden_size: usize,
    /// Number of hidden layers in the [`Encoder`](crate::encoder::Encoder).
    pub(crate) num_hidden_layers: usize,
    /// Number of attention heads for each attention layer in the [`Encoder`](crate::encoder::Encoder).
    pub(crate) num_attention_heads: usize,
    /// Dimensionality of the [`Intermediate`](crate::encoder::layer::intermediate::Intermediate)
    /// (i.e., feed-forward) layer in the [`Encoder`](crate::encoder::Encoder).
    pub(crate) intermediate_size: usize,
    /// The [`HiddenAct`] non-linear activation function in the [`Encoder`](crate::encoder::Encoder).
    pub(crate) hidden_act: HiddenAct,
    /// The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    pub(crate) hidden_dropout_prob: f32,
    /// The maximum sequence length that this model might ever be used with. Typically set this to something large
    /// just in case (e.g., 512 or 1024 or 2048).
    pub(crate) max_position_embeddings: usize,
    /// The vocabulary size of the `token_type_ids` passed when calling [`Model::forward`](crate::Model::forward).
    pub(crate) type_vocab_size: usize,
    /// The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    pub(crate) initializer_range: f64,
    /// The epsilon used by the layer normalization layers.
    pub(crate) layer_norm_eps: f64,
    pub(crate) pad_token_id: usize,
    pub(crate) bos_token_id: usize,
    pub(crate) eos_token_id: usize,
    /// Type of position embedding.
    #[serde(default)]
    pub(crate) position_embedding_type: PositionEmbeddingType,
    /// Whether or not the model should return the last key/values attentions.
    #[serde(default)]
    pub(crate) use_cache: bool,
    /// The dropout ratio for the classification head.
    pub(crate) classifier_dropout: Option<f64>,
    pub(crate) model_type: Option<String>,
    pub(crate) problem_type: Option<String>,
    _num_labels: Option<usize>,
    pub(crate) id2label: Option<HashMap<String, String>>,
    pub(crate) label2id: Option<HashMap<String, usize>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HiddenAct {
    Gelu,
    Relu,
    Tanh,
}
