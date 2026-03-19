use instant_distance::{Builder, HnswMap, Search};
use serde::{Deserialize, Serialize};

use crate::error::Result;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkMeta {
    pub chunk_id: String,
    pub doc_id: String,
    pub doc_title: String,
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct Point(pub Vec<f32>);

impl instant_distance::Point for Point {
    fn distance(&self, other: &Self) -> f32 {
        // Cosine distance = 1 - cosine_similarity
        // Since vectors are L2-normalized, cosine_similarity = dot product
        let dot: f32 = self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum();
        1.0 - dot
    }
}

pub struct SearchResult {
    pub chunk_meta: ChunkMeta,
    pub distance: f32,
}

pub struct VectorIndex {
    map: HnswMap<Point, ChunkMeta>,
}

impl VectorIndex {
    pub fn build(items: Vec<(Vec<f32>, ChunkMeta)>) -> Result<Self> {
        let (points, values): (Vec<Point>, Vec<ChunkMeta>) = items
            .into_iter()
            .map(|(embedding, meta)| (Point(embedding), meta))
            .unzip();

        let map = Builder::default().build(points, values);
        Ok(Self { map })
    }

    pub fn search(&self, query_embedding: &[f32], k: usize) -> Vec<SearchResult> {
        let query = Point(query_embedding.to_vec());
        let mut search = Search::default();
        let results = self.map.search(&query, &mut search);

        results
            .take(k)
            .map(|item| SearchResult {
                chunk_meta: item.value.clone(),
                distance: item.distance,
            })
            .collect()
    }

    pub fn list_documents(&self) -> Vec<DocumentInfo> {
        let mut seen = std::collections::HashMap::new();
        for value in &self.map.values {
            seen.entry(value.doc_id.clone())
                .or_insert_with(|| DocumentInfo {
                    doc_id: value.doc_id.clone(),
                    title: value.doc_title.clone(),
                    chunk_count: 0,
                })
                .chunk_count += 1;
        }
        seen.into_values().collect()
    }

    pub fn chunk_count(&self) -> usize {
        self.map.values.len()
    }
}

#[derive(Debug)]
pub struct DocumentInfo {
    pub doc_id: String,
    pub title: String,
    pub chunk_count: usize,
}
