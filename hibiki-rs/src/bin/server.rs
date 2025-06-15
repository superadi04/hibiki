use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::Extension,
    routing::get,
    Router,
    response::IntoResponse,
};
use hyper::server::Server;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

use hibiki::{gen::{Args, run}, Config, device};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    // Load models here once
    let device = device(false).unwrap(); // use CUDA
    let config: Config = toml::from_str(&std::fs::read_to_string("config.toml").unwrap()).unwrap();

    // Shared state (models, tokenizer)
    let state = Arc::new(Mutex::new(HibikiState::new(config, device).await.unwrap()));

    let app = Router::new()
        .route("/ws", get(handle_ws))
        .layer(Extension(state));

    info!("Starting server on 0.0.0.0:8080");

    Server::bind(&"0.0.0.0:8080".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}

pub struct HibikiState {
    pub config: Config,
    pub device: candle::Device,
    pub args: Args,
}

impl HibikiState {
    pub async fn new(config: Config, device: candle::Device) -> anyhow::Result<Self> {
        let repo = hf_hub::api::sync::Api::new()?.model("kyutai/hibiki-1b-rs-bf16".to_string());

        let lm_model_file = repo.get(&config.moshi_name)?;
        let mimi_model_file = repo.get(&config.mimi_name)?;
        let tokenizer_file = repo.get(&config.tokenizer_name)?;

        let args = Args {
            lm_config: config.model.clone(),
            lm_model_file,
            mimi_model_file,
            text_tokenizer: tokenizer_file,
            audio_input_file: "in.wav".into(),
            audio_output_file: "out.wav".into(),
            seed: 42,
            cfg_alpha: Some(1.0),
        };

        Ok(Self { config, device, args })
    }
}

async fn handle_ws(ws: WebSocketUpgrade, Extension(state): Extension<Arc<Mutex<HibikiState>>>) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: Arc<Mutex<HibikiState>>) {
    while let Some(Ok(msg)) = socket.recv().await {
        if let Message::Binary(audio_data) = msg {
            std::fs::write("in.wav", &audio_data).unwrap();

            // clone args + update path
            let mut state = state.lock().await;
            state.args.audio_input_file = "in.wav".into();
            state.args.audio_output_file = "out.wav".into();

            run(&state.args, &state.device).unwrap();

            let output_audio = std::fs::read("out.wav").unwrap();
            socket.send(Message::Binary(output_audio)).await.unwrap();
        }
    }
}
