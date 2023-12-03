use candle_nn::{Linear, LayerNorm, Dropout};

pub struct Output {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}
