pub mod audio_io;
pub mod gen;

use candle::Device;
use anyhow::Result;

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if candle::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if candle::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub mimi_name: String,
    pub moshi_name: String,
    pub tokenizer_name: String,
    pub model: moshi::lm::Config,
}
