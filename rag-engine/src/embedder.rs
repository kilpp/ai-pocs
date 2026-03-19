use std::path::Path;

use ndarray::{Array2, Axis};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::error::{RagError, Result};

pub struct Embedder {
    session: Session,
    tokenizer: Tokenizer,
}

impl Embedder {
    pub fn new(model_dir: &Path) -> Result<Self> {
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        if !model_path.exists() {
            return Err(RagError::Embedding(format!(
                "Model file not found: {}. Download all-MiniLM-L6-v2 ONNX model first.",
                model_path.display()
            )));
        }
        if !tokenizer_path.exists() {
            return Err(RagError::Embedding(format!(
                "Tokenizer file not found: {}. Download tokenizer.json first.",
                tokenizer_path.display()
            )));
        }

        let session = Session::builder()
            .map_err(|e| RagError::Embedding(format!("Failed to create session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| RagError::Embedding(format!("Failed to set optimization level: {e}")))?
            .commit_from_file(&model_path)
            .map_err(|e| RagError::Embedding(format!("Failed to load ONNX model: {e}")))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| RagError::Embedding(format!("Failed to load tokenizer: {e}")))?;

        Ok(Self { session, tokenizer })
    }

    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let batch = self.embed_batch(&[text])?;
        Ok(batch.into_iter().next().unwrap())
    }

    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| RagError::Embedding(format!("Tokenization failed: {e}")))?;

        let batch_size = encodings.len();
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        let mut input_ids = Array2::<i64>::zeros((batch_size, max_len));
        let mut attention_mask = Array2::<i64>::zeros((batch_size, max_len));
        let mut token_type_ids = Array2::<i64>::zeros((batch_size, max_len));

        for (i, encoding) in encodings.iter().enumerate() {
            for (j, &id) in encoding.get_ids().iter().enumerate() {
                input_ids[[i, j]] = id as i64;
            }
            for (j, &mask) in encoding.get_attention_mask().iter().enumerate() {
                attention_mask[[i, j]] = mask as i64;
            }
            for (j, &type_id) in encoding.get_type_ids().iter().enumerate() {
                token_type_ids[[i, j]] = type_id as i64;
            }
        }

        let input_ids_tensor = Tensor::from_array(input_ids.clone())
            .map_err(|e| RagError::Embedding(format!("Failed to create input tensor: {e}")))?;
        let attention_mask_tensor = Tensor::from_array(attention_mask.clone())
            .map_err(|e| RagError::Embedding(format!("Failed to create mask tensor: {e}")))?;
        let token_type_ids_tensor = Tensor::from_array(token_type_ids)
            .map_err(|e| RagError::Embedding(format!("Failed to create type_ids tensor: {e}")))?;

        let outputs = self
            .session
            .run(ort::inputs![input_ids_tensor, attention_mask_tensor, token_type_ids_tensor])
            .map_err(|e| RagError::Embedding(format!("Inference failed: {e}")))?;

        // Output: use try_extract_array to get ndarray view
        let output_view = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| RagError::Embedding(format!("Failed to extract output: {e}")))?;

        // Output shape: (batch_size, seq_len, hidden_size=384)
        let shape = output_view.shape();
        let hidden_size = shape[shape.len() - 1];

        let output_3d = output_view
            .into_shape_with_order((batch_size, max_len, hidden_size))
            .map_err(|e| RagError::Embedding(format!("Failed to reshape output: {e}")))?;

        // Mean pooling with attention mask
        let mut embeddings = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let token_embeddings = output_3d.index_axis(Axis(0), i);
            let mask = attention_mask.index_axis(Axis(0), i);

            let mut pooled = vec![0.0f32; hidden_size];
            let mut count = 0.0f32;

            for (j, &m) in mask.iter().enumerate() {
                if m == 1 {
                    let token_emb = token_embeddings.index_axis(Axis(0), j);
                    for (k, &v) in token_emb.iter().enumerate() {
                        pooled[k] += v;
                    }
                    count += 1.0;
                }
            }

            if count > 0.0 {
                for v in &mut pooled {
                    *v /= count;
                }
            }

            // L2 normalize
            let norm: f32 = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut pooled {
                    *v /= norm;
                }
            }

            embeddings.push(pooled);
        }

        Ok(embeddings)
    }

    pub fn dimension(&self) -> usize {
        384 // all-MiniLM-L6-v2 output dimension
    }
}
