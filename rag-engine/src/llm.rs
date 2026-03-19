use serde::{Deserialize, Serialize};

use crate::error::{RagError, Result};
use crate::index::SearchResult;

pub struct LlmClient {
    api_key: String,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ApiRequest {
    model: String,
    max_tokens: u32,
    system: String,
    messages: Vec<Message>,
}

#[derive(Deserialize)]
struct ApiResponse {
    content: Vec<ContentBlock>,
}

#[derive(Deserialize)]
struct ContentBlock {
    text: Option<String>,
}

impl LlmClient {
    pub fn new() -> Result<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
            RagError::Api(
                "ANTHROPIC_API_KEY environment variable not set. \
                 Set it with: export ANTHROPIC_API_KEY=your-key"
                    .to_string(),
            )
        })?;

        Ok(Self {
            api_key,
            client: reqwest::Client::new(),
        })
    }

    pub async fn query(&self, question: &str, context: &[SearchResult]) -> Result<String> {
        let mut context_text = String::new();
        for (i, result) in context.iter().enumerate() {
            context_text.push_str(&format!(
                "--- Chunk {} (from: {}, distance: {:.4}) ---\n{}\n\n",
                i + 1,
                result.chunk_meta.doc_title,
                result.distance,
                result.chunk_meta.text
            ));
        }

        let system = "You are a helpful assistant that answers questions based on the provided \
                       context documents. Use only the information from the context to answer. \
                       If the context doesn't contain enough information to answer the question, \
                       say so clearly."
            .to_string();

        let user_message = format!(
            "Context:\n{context_text}\nQuestion: {question}\n\nAnswer based on the context above."
        );

        let request = ApiRequest {
            model: "claude-sonnet-4-20250514".to_string(),
            max_tokens: 1024,
            system,
            messages: vec![Message {
                role: "user".to_string(),
                content: user_message,
            }],
        };

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| RagError::Api(format!("Request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(RagError::Api(format!(
                "API returned {status}: {body}"
            )));
        }

        let api_response: ApiResponse = response
            .json()
            .await
            .map_err(|e| RagError::Api(format!("Failed to parse response: {e}")))?;

        let text = api_response
            .content
            .into_iter()
            .filter_map(|block| block.text)
            .collect::<Vec<_>>()
            .join("");

        if text.is_empty() {
            return Err(RagError::Api("Empty response from API".to_string()));
        }

        Ok(text)
    }
}
