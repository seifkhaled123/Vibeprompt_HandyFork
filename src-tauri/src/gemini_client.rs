use base64::Engine;
use hound::{SampleFormat, WavSpec, WavWriter};
use serde::Deserialize;
use std::io::Cursor;

use crate::settings::{AppSettings, GEMINI_DEFAULT_MODEL_ID};

const GEMINI_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

const GEMINI_SYSTEM_PROMPT: &str = "You are an expert software engineering assistant that translates Arabic speech into precise, actionable English engineering prompts.\n\nYour job is to listen to the provided voice recording and produce a final English prompt that another coding assistant can execute directly.\n\nOutput format:\n1. Goal (one line)\n2. Context (if the user mentions relevant codebase details)\n3. Requirements (numbered list)\n4. Files to Modify (only if explicitly mentioned)\n5. Constraints\n\nRules:\n- Output must be English.\n- Be specific and actionable.\n- Do not include internal reasoning.\n- Return only the final prompt body.";

#[derive(Debug, Deserialize)]
struct GeminiModelsResponse {
    #[serde(default)]
    models: Vec<GeminiModelEntry>,
}

#[derive(Debug, Deserialize)]
struct GeminiModelEntry {
    name: String,
    #[serde(default, rename = "supportedGenerationMethods")]
    supported_generation_methods: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct GenerateContentResponse {
    #[serde(default)]
    candidates: Vec<Candidate>,
}

#[derive(Debug, Deserialize)]
struct Candidate {
    #[serde(default)]
    content: Option<ResponseContent>,
}

#[derive(Debug, Deserialize)]
struct ResponseContent {
    #[serde(default)]
    parts: Vec<ResponsePart>,
}

#[derive(Debug, Deserialize)]
struct ResponsePart {
    #[serde(default)]
    text: Option<String>,
}

fn normalize_model_id(model: &str) -> String {
    let trimmed = model.trim();
    if trimmed.is_empty() {
        return GEMINI_DEFAULT_MODEL_ID.to_string();
    }

    trimmed
        .strip_prefix("models/")
        .unwrap_or(trimmed)
        .to_string()
}

fn build_wav_bytes(samples: &[f32]) -> Result<Vec<u8>, String> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut cursor, spec)
        .map_err(|e| format!("Failed to start WAV writer: {}", e))?;

    for sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let value = (clamped * i16::MAX as f32) as i16;
        writer
            .write_sample(value)
            .map_err(|e| format!("Failed to write WAV sample: {}", e))?;
    }

    writer
        .finalize()
        .map_err(|e| format!("Failed to finalize WAV: {}", e))?;

    Ok(cursor.into_inner())
}

pub async fn fetch_models(api_key: &str) -> Result<Vec<String>, String> {
    let key = api_key.trim();
    if key.is_empty() {
        return Err("Gemini API key is required.".to_string());
    }

    let url = format!("{}/models?key={}", GEMINI_BASE_URL, key);

    let response = reqwest::Client::new()
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Failed to fetch Gemini models: {}", e))?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(format!(
            "Gemini model list request failed ({}): {}",
            status, error_text
        ));
    }

    let parsed: GeminiModelsResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse Gemini models response: {}", e))?;

    let mut models: Vec<String> = parsed
        .models
        .into_iter()
        .filter(|entry| {
            entry
                .supported_generation_methods
                .iter()
                .any(|method| method == "generateContent")
        })
        .map(|entry| {
            entry
                .name
                .strip_prefix("models/")
                .unwrap_or(&entry.name)
                .to_string()
        })
        .collect();

    models.sort();
    models.dedup();

    Ok(models)
}

pub async fn transcribe_audio_to_prompt(
    settings: &AppSettings,
    samples: &[f32],
) -> Result<String, String> {
    let api_key = settings.gemini_api_key.trim();
    if api_key.is_empty() {
        return Err("Gemini API key is missing. Add it in Settings > Models.".to_string());
    }

    if samples.is_empty() {
        return Ok(String::new());
    }

    let model_id = normalize_model_id(&settings.gemini_model);
    let wav = build_wav_bytes(samples)?;
    let audio_b64 = base64::engine::general_purpose::STANDARD.encode(wav);

    let url = format!(
        "{}/models/{}:generateContent?key={}",
        GEMINI_BASE_URL, model_id, api_key
    );

    let body = serde_json::json!({
        "system_instruction": {
            "parts": [{ "text": GEMINI_SYSTEM_PROMPT }]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    { "text": "Analyze this Arabic speech recording and return the final English engineering prompt in the required format." },
                    {
                        "inline_data": {
                            "mime_type": "audio/wav",
                            "data": audio_b64
                        }
                    }
                ]
            }
        ]
    });

    let response = reqwest::Client::new()
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Failed to call Gemini: {}", e))?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(format!("Gemini request failed ({}): {}", status, error_text));
    }

    let parsed: GenerateContentResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse Gemini response: {}", e))?;

    let text = parsed
        .candidates
        .first()
        .and_then(|candidate| candidate.content.as_ref())
        .map(|content| {
            content
                .parts
                .iter()
                .filter_map(|part| part.text.clone())
                .collect::<Vec<String>>()
                .join("\n")
        })
        .unwrap_or_default()
        .trim()
        .to_string();

    if text.is_empty() {
        return Err("Gemini returned an empty response".to_string());
    }

    Ok(text)
}
