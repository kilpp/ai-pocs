use std::path::PathBuf;

use clap::{Parser, Subcommand};
use colored::*;

use rag_engine::document::{chunk_document, read_documents_from_path};
use rag_engine::embedder::Embedder;
use rag_engine::index::ChunkMeta;
use rag_engine::llm::LlmClient;
use rag_engine::store;

#[derive(Parser)]
#[command(name = "rag", about = "Local RAG engine - ingest documents and query with AI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Ingest documents into the vector index
    Ingest {
        /// Path to a file or directory to ingest
        #[arg(short, long)]
        path: PathBuf,

        /// Chunk size in characters
        #[arg(long, default_value = "512")]
        chunk_size: usize,

        /// Overlap between chunks in characters
        #[arg(long, default_value = "50")]
        overlap: usize,

        /// Path to the ONNX model directory
        #[arg(long)]
        model_dir: Option<PathBuf>,

        /// Path to the index file
        #[arg(long)]
        index_path: Option<PathBuf>,
    },

    /// Ask a question using RAG
    Query {
        /// The question to ask
        #[arg(short, long)]
        question: String,

        /// Number of context chunks to retrieve
        #[arg(short, long, default_value = "5")]
        top_k: usize,

        /// Path to the ONNX model directory
        #[arg(long)]
        model_dir: Option<PathBuf>,

        /// Path to the index file
        #[arg(long)]
        index_path: Option<PathBuf>,
    },

    /// List all indexed documents
    List {
        /// Path to the index file
        #[arg(long)]
        index_path: Option<PathBuf>,
    },

    /// Clear the index
    Clear {
        /// Path to the index file
        #[arg(long)]
        index_path: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    if let Err(e) = run(cli).await {
        eprintln!("{} {}", "Error:".red().bold(), e);
        std::process::exit(1);
    }
}

async fn run(cli: Cli) -> anyhow::Result<()> {
    match cli.command {
        Commands::Ingest {
            path,
            chunk_size,
            overlap,
            model_dir,
            index_path,
        } => {
            let model_dir = model_dir.unwrap_or_else(store::default_model_dir);
            let index_path = index_path.unwrap_or_else(store::default_store_dir);

            println!("{}", "Reading documents...".cyan());
            let docs = read_documents_from_path(&path)?;
            println!("  Found {} document(s)", docs.len().to_string().green());

            let mut all_chunks = Vec::new();
            for doc in &docs {
                let chunks = chunk_document(doc, chunk_size, overlap);
                println!(
                    "  {} {} chunks",
                    doc.title.yellow(),
                    chunks.len().to_string().green()
                );
                all_chunks.extend(chunks);
            }

            println!(
                "\n{} ({} total chunks)",
                "Generating embeddings...".cyan(),
                all_chunks.len()
            );
            let mut embedder = Embedder::new(&model_dir)?;

            let texts: Vec<&str> = all_chunks.iter().map(|c| c.text.as_str()).collect();
            let embeddings = embedder.embed_batch(&texts)?;

            println!("{}", "Building index...".cyan());
            let mut index_data = store::load(&index_path)?;
            for (chunk, embedding) in all_chunks.iter().zip(embeddings.into_iter()) {
                index_data.add(
                    embedding,
                    ChunkMeta {
                        chunk_id: chunk.id.clone(),
                        doc_id: chunk.doc_id.clone(),
                        doc_title: chunk.doc_title.clone(),
                        text: chunk.text.clone(),
                    },
                );
            }

            store::save(&index_data, &index_path)?;

            let index = index_data.build_index()?;
            println!(
                "\n{} {} chunks from {} document(s) indexed.",
                "Done!".green().bold(),
                index.chunk_count().to_string().green(),
                index.list_documents().len().to_string().green()
            );
        }

        Commands::Query {
            question,
            top_k,
            model_dir,
            index_path,
        } => {
            let model_dir = model_dir.unwrap_or_else(store::default_model_dir);
            let index_path = index_path.unwrap_or_else(store::default_store_dir);

            let index_data = store::load(&index_path)?;
            if index_data.items.is_empty() {
                anyhow::bail!("Index is empty. Ingest some documents first with: rag ingest --path <path>");
            }

            let index = index_data.build_index()?;

            println!("{}", "Embedding query...".cyan());
            let mut embedder = Embedder::new(&model_dir)?;
            let query_embedding = embedder.embed(&question)?;

            println!("{}", "Searching index...".cyan());
            let results = index.search(&query_embedding, top_k);

            println!("\n{}", "Retrieved context:".yellow().bold());
            for (i, result) in results.iter().enumerate() {
                println!(
                    "  {}. {} {} (distance: {:.4})",
                    i + 1,
                    "[".dimmed(),
                    result.chunk_meta.doc_title.cyan(),
                    result.distance
                );
                let preview: String = result.chunk_meta.text.chars().take(100).collect();
                println!("     {}...", preview.dimmed());
            }

            println!("\n{}", "Querying Claude...".cyan());
            let llm = LlmClient::new()?;
            let answer = llm.query(&question, &results).await?;

            println!("\n{}", "Answer:".green().bold());
            println!("{answer}");
        }

        Commands::List { index_path } => {
            let index_path = index_path.unwrap_or_else(store::default_store_dir);
            let index_data = store::load(&index_path)?;

            if index_data.items.is_empty() {
                println!("{}", "Index is empty.".yellow());
                return Ok(());
            }

            let index = index_data.build_index()?;
            let docs = index.list_documents();

            println!("{}", "Indexed documents:".green().bold());
            for doc in &docs {
                println!(
                    "  {} ({} chunks)",
                    doc.title.cyan(),
                    doc.chunk_count.to_string().green()
                );
            }
            println!(
                "\nTotal: {} document(s), {} chunk(s)",
                docs.len().to_string().green(),
                index.chunk_count().to_string().green()
            );
        }

        Commands::Clear { index_path } => {
            let index_path = index_path.unwrap_or_else(store::default_store_dir);
            store::clear(&index_path)?;
            println!("{}", "Index cleared.".green());
        }
    }

    Ok(())
}
