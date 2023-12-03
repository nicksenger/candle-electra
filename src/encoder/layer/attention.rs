mod self_attention;
mod self_output;

use self_attention::SelfAttention;
use self_output::SelfOutput;

pub struct Attention {
    self_attention: SelfAttention,
    self_output: SelfOutput,
}
