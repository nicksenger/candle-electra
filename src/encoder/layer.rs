mod attention;
mod intermediate;
mod output;

use attention::Attention;
use intermediate::Intermediate;
use output::Output;

pub struct Layer {
    attention: Attention,
    intermediate: Intermediate,
    output: Output,
}
