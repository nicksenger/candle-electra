use candle_nn::Linear;
use serde::Deserialize;

pub struct Intermediate {
    dense: Linear,
    intermediate_act: crate::HiddenActLayer,
}
