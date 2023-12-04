use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PositionEmbeddingType {
    #[default]
    Absolute,
}

/// Configuration class to store the configuration of a [`Model`](crate::Model)
#[derive(Deserialize, Serialize)]
pub struct Config {
    /// Vocabulary size of the ELECTRA model. Defines the number of different tokens that can be
    /// represented by the`input_ids` passed when calling [`Model::forward`](crate::Model::forward)
    pub vocab_size: usize,
    /// Dimensionality of the [`Encoder`](crate::encoder::Encoder) layers and the pooler layer.
    pub hidden_size: usize,
    /// Number of hidden layers in the [`Encoder`](crate::encoder::Encoder).
    pub num_hidden_layers: usize,
    /// Number of attention heads for each attention layer in the [`Encoder`](crate::encoder::Encoder).
    pub num_attention_heads: usize,
    /// Dimensionality of the [`Intermediate`](crate::encoder::layer::intermediate::Intermediate)
    /// (i.e., feed-forward) layer in the [`Encoder`](crate::encoder::Encoder).
    pub intermediate_size: usize,
    /// The [`HiddenAct`] non-linear activation function in the [`Encoder`](crate::encoder::Encoder).
    pub hidden_act: HiddenAct,
    /// The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    pub hidden_dropout_prob: f64,
    /// The maximum sequence length that this model might ever be used with. Typically set this to something large
    /// just in case (e.g., 512 or 1024 or 2048).
    pub max_position_embeddings: usize,
    /// The vocabulary size of the `token_type_ids` passed when calling [`Model::forward`](crate::Model::forward).
    pub type_vocab_size: usize,
    // /// The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    // pub initializer_range: f64,
    /// The epsilon used by the layer normalization layers.
    pub layer_norm_eps: f64,
    pub pad_token_id: usize,
    // pub bos_token_id: usize,
    // pub eos_token_id: usize,
    /// Type of position embedding.
    #[serde(default)]
    pub position_embedding_type: PositionEmbeddingType,
    // /// Whether or not the model should return the last key/values attentions.
    // #[serde(default)]
    // pub use_cache: bool,
    /// The dropout ratio for the classification head.
    pub classifier_dropout: Option<f64>,
    pub model_type: Option<String>,
    // pub problem_type: Option<String>,
    pub num_labels: Option<usize>,
    // pub id2label: Option<HashMap<String, String>>,
    // pub label2id: Option<HashMap<String, usize>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum HiddenAct {
    Gelu,
    Relu,
    Tanh,
}
