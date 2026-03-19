use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{RagError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub path: String,
    pub title: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub doc_id: String,
    pub doc_title: String,
    pub text: String,
    pub start_offset: usize,
    pub embedding: Option<Vec<f32>>,
}

pub fn read_document(path: &Path) -> Result<Document> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    if ext != "txt" && ext != "md" {
        return Err(RagError::InvalidInput(format!(
            "Unsupported file type: .{ext} (only .txt and .md supported)"
        )));
    }

    let content = fs::read_to_string(path)?;
    let title = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("untitled")
        .to_string();

    Ok(Document {
        id: Uuid::new_v4().to_string(),
        path: path.display().to_string(),
        title,
        content,
    })
}

pub fn read_documents_from_path(path: &Path) -> Result<Vec<Document>> {
    if path.is_file() {
        return Ok(vec![read_document(path)?]);
    }

    if !path.is_dir() {
        return Err(RagError::InvalidInput(format!(
            "Path does not exist: {}",
            path.display()
        )));
    }

    let mut docs = Vec::new();
    collect_documents(path, &mut docs)?;

    if docs.is_empty() {
        return Err(RagError::InvalidInput(
            "No .txt or .md files found in the directory".to_string(),
        ));
    }

    Ok(docs)
}

fn collect_documents(dir: &Path, docs: &mut Vec<Document>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_documents(&path, docs)?;
        } else if path.is_file() {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext == "txt" || ext == "md" {
                docs.push(read_document(&path)?);
            }
        }
    }
    Ok(())
}

pub fn chunk_document(doc: &Document, chunk_size: usize, overlap: usize) -> Vec<Chunk> {
    let text = doc.content.trim();
    if text.is_empty() {
        return Vec::new();
    }

    let chars: Vec<char> = text.chars().collect();
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < chars.len() {
        let end = (start + chunk_size).min(chars.len());
        let chunk_text: String = chars[start..end].iter().collect();

        if !chunk_text.trim().is_empty() {
            chunks.push(Chunk {
                id: Uuid::new_v4().to_string(),
                doc_id: doc.id.clone(),
                doc_title: doc.title.clone(),
                text: chunk_text,
                start_offset: start,
                embedding: None,
            });
        }

        if end >= chars.len() {
            break;
        }

        start += chunk_size - overlap;
    }

    chunks
}
