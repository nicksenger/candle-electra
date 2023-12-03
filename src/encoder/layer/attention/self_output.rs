use candle_nn::{Dropout, LayerNorm, Linear};

pub struct SelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}
