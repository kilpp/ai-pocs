use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{RagError, Result};
use crate::index::{ChunkMeta, VectorIndex};

#[derive(Serialize, Deserialize)]
pub struct IndexData {
    pub items: Vec<(Vec<f32>, ChunkMeta)>,
}

impl IndexData {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn add(&mut self, embedding: Vec<f32>, meta: ChunkMeta) {
        self.items.push((embedding, meta));
    }

    pub fn build_index(&self) -> Result<VectorIndex> {
        if self.items.is_empty() {
            return Err(RagError::Index("No items in index data".to_string()));
        }
        VectorIndex::build(self.items.clone())
    }
}

pub fn default_store_dir() -> std::path::PathBuf {
    dirs_path().join("index.bin")
}

pub fn default_model_dir() -> std::path::PathBuf {
    dirs_path().join("model")
}

fn dirs_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Path::new(&home).join(".rag-engine")
}

pub fn save(data: &IndexData, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let encoded = bincode::serialize(data)
        .map_err(|e| RagError::Serialization(format!("Failed to serialize index: {e}")))?;

    fs::write(path, encoded)?;
    Ok(())
}

pub fn load(path: &Path) -> Result<IndexData> {
    if !path.exists() {
        return Ok(IndexData::new());
    }

    let bytes = fs::read(path)?;
    let data: IndexData = bincode::deserialize(&bytes)
        .map_err(|e| RagError::Serialization(format!("Failed to deserialize index: {e}")))?;

    Ok(data)
}

pub fn clear(path: &Path) -> Result<()> {
    if path.exists() {
        fs::remove_file(path)?;
    }
    Ok(())
}
