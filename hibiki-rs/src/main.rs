// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use anyhow::Result;
use clap::Parser;

mod audio_io;
mod gen;

use candle::Device;

#[derive(Debug, Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,
}

#[derive(Debug, clap::Subcommand)]
enum Command {
    Gen {
        #[arg(long)]
        lm_model_file: Option<String>,

        #[arg(long)]
        mimi_model_file: Option<String>,

        #[arg(long)]
        config: Option<String>,

        #[arg(long)]
        text_tokenizer: Option<String>,

        #[arg(long, default_value = "kyutai/hibiki-1b-rs-bf16")]
        hf_repo: String,

        #[arg()]
        audio_input_file: String,

        #[arg()]
        audio_output_file: String,

        #[arg(long, default_value_t = 299_792_458)]
        seed: u64,

        #[arg(long)]
        cfg_alpha: Option<f64>,

        /// Run on cpu
        #[arg(long)]
        cpu: bool,
    },
}

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

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    match args.command {
        Command::Gen {
            Command::Gen {
                seed,
                text_tokenizer,
                lm_model_file,
                config,
                mimi_model_file,
                hf_repo,
                audio_input_file,
                audio_output_file,
                cfg_alpha,
                cpu,
            } => {
                let dev = device(cpu)?;
                tracing_subscriber::fmt::init();
                let api = hf_hub::api::sync::Api::new()?;
                let hf_repo = match hf_repo.as_str() {
                    "1b" => "kyutai/hibiki-1b-rs-bf16".to_string(),
                    "2b" => "kyutai/hibiki-2b-rs-bf16".to_string(),
                    _ => hf_repo,
                };
                let repo = api.model(hf_repo);
                let config_path = match config {
                    None => repo.get("config.toml")?,
                    Some(f) => std::path::PathBuf::from(f),
                };
                tracing::info!("loading the config");
                let config_str = std::fs::read_to_string(&config_path)?;
                let config: gen::Config = toml::from_str(&config_str)?;
            
                let lm_model_file = match lm_model_file {
                    None => repo.get(&config.moshi_name)?,
                    Some(v) => std::path::PathBuf::from(v),
                };
                let mimi_model_file = match mimi_model_file {
                    None => repo.get(&config.mimi_name)?,
                    Some(v) => std::path::PathBuf::from(v),
                };
                let text_tokenizer = match text_tokenizer {
                    None => repo.get(&config.tokenizer_name)?,
                    Some(v) => std::path::PathBuf::from(v),
                };
            
                // --- Step 2: Load and split full audio file into chunks ---
                let (pcm, sample_rate) = audio_io::pcm_decode(&audio_input_file)?;
                let chunk_size = 48_000; // 2s at 24kHz
                let chunks: Vec<&[f32]> = pcm.chunks(chunk_size).collect();
            
                for (i, chunk) in chunks.iter().enumerate() {
                    let chunk_in_path = format!("chunk_{i}.wav");
                    let chunk_out_path = format!("chunk_out_{i}.wav");
            
                    // Write the chunk as an input WAV file
                    let mut f = std::fs::File::create(&chunk_in_path)?;
                    moshi::wav::write_pcm_as_wav(&mut f, chunk, sample_rate)?;
            
                    let chunk_args = gen::Args {
                        lm_config: config.model.clone(),
                        lm_model_file: lm_model_file.clone(),
                        mimi_model_file: mimi_model_file.clone(),
                        text_tokenizer: text_tokenizer.clone(),
                        audio_input_file: chunk_in_path.into(),
                        audio_output_file: chunk_out_path.into(),
                        seed,
                        cfg_alpha,
                    };
            
                    println!("Processing chunk {i}...");
                    gen::run(&chunk_args, &dev)?;
                    println!("Saved output to {}", chunk_out_path);
                }
            }
        }
    }
    Ok(())
}
