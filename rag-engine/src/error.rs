use thiserror::Error;

#[derive(Debug, Error)]
pub enum RagError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Index error: {0}")]
    Index(String),

    #[error("API error: {0}")]
    Api(String),

    #[error("HTTP error: {0}")]
    Reqwest(#[from] reqwest::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, RagError>;
