# Movie Recommendation System

A comprehensive movie recommendation system implemented in Rust, featuring multiple recommendation strategies including collaborative filtering, content-based filtering, and hybrid approaches.

## Features

### 1. **Collaborative Filtering**
- **User-based Collaborative Filtering**: Finds similar users based on rating patterns using Pearson correlation
- **Item-based Collaborative Filtering**: Recommends movies based on item similarity using cosine similarity
- Handles explicit feedback (ratings from 1.0 to 5.0)

### 2. **Content-Based Filtering**
- Analyzes movie features (genres, directors, actors, year)
- Builds user profiles based on rating history
- Uses Jaccard similarity for feature matching
- Can find similar movies based on content

### 3. **Hybrid Recommendation**
- **Weighted Strategy**: Combines collaborative and content-based scores with configurable weights
- **Switching Strategy**: Adapts based on data availability (uses collaborative if enough ratings exist, otherwise content-based)
- **Mixed Strategy**: Merges results from both methods and re-ranks

## Project Structure

```
src/
├── main.rs                      # Demo application with examples
├── models.rs                    # Core data models (Movie, User, Rating, Dataset)
├── collaborative_filtering.rs   # Collaborative filtering algorithms
├── content_based.rs            # Content-based filtering algorithms
├── hybrid.rs                   # Hybrid recommendation strategies
└── sample_data.rs              # Sample dataset for testing
```

## Usage

### Build and Run

```bash
cargo build --release
cargo run
```

### Example Output

The demo application shows recommendations for a sample user using all available strategies:

```
=== Movie Recommendation System ===

Loaded 10 movies and 5 users with 22 ratings

Generating recommendations for user: Alice (ID: 1)

Alice's Ratings:
  - The Godfather (1972) - Rating: 5.0
  - Pulp Fiction (1994) - Rating: 4.5
  - Goodfellas (1990) - Rating: 4.5
  - The Silence of the Lambs (1991) - Rating: 4.0

--- Collaborative Filtering (User-based) ---
  1. The Shawshank Redemption (1994) - Score: 4.750
     Genres: Drama
  2. Forrest Gump (1994) - Score: 4.625
     Genres: Drama, Romance
  ...
```

## Key Algorithms

### Collaborative Filtering
- **Pearson Correlation**: Measures similarity between users based on common ratings
- **Cosine Similarity**: Measures similarity between items based on user rating patterns

### Content-Based Filtering
- **Feature Extraction**: Creates feature vectors from movie metadata
- **User Profile Building**: Aggregates features from rated movies with weighted preferences
- **Jaccard Similarity**: Compares feature sets for similarity measurement

### Hybrid Approaches
- **Normalization**: Scores from different algorithms are normalized to [0, 1] range
- **Weighted Combination**: Allows fine-tuning the influence of each algorithm
- **Adaptive Selection**: Chooses the best algorithm based on available data

## Dependencies

```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
nalgebra = "0.33"
rayon = "1.10"
```

## API Examples

### Using Collaborative Filtering

```rust
use collaborative_filtering::CollaborativeFilter;

let dataset = create_sample_dataset();
let collab_filter = CollaborativeFilter::new(&dataset);

// User-based recommendations
let recommendations = collab_filter.recommend(user_id, 5);

// Item-based recommendations
let recommendations = collab_filter.recommend_item_based(user_id, 5);
```

### Using Content-Based Filtering

```rust
use content_based::ContentBasedFilter;

let content_filter = ContentBasedFilter::new(&dataset);

// Get recommendations
let recommendations = content_filter.recommend(user_id, 5);

// Find similar movies
let similar = content_filter.find_similar_movies(movie_id, 5);
```

### Using Hybrid Recommender

```rust
use hybrid::{HybridRecommender, HybridStrategy};

let hybrid = HybridRecommender::new(&dataset);

// Weighted strategy (70% collaborative, 30% content-based)
let recs = hybrid.recommend(
    user_id,
    5,
    HybridStrategy::Weighted {
        collaborative_weight: 0.7,
        content_weight: 0.3,
    },
);

// Switching strategy
let recs = hybrid.recommend(user_id, 5, HybridStrategy::Switching);

// Mixed strategy
let recs = hybrid.recommend(user_id, 5, HybridStrategy::Mixed);
```

## Extending the System

### Adding New Movies

```rust
dataset.add_movie(Movie {
    id: 11,
    title: "Your Movie Title".to_string(),
    genres: vec!["Action".to_string(), "Thriller".to_string()],
    year: 2024,
    director: "Director Name".to_string(),
    actors: vec!["Actor 1".to_string(), "Actor 2".to_string()],
});
```

### Adding Ratings

```rust
dataset.add_rating(Rating {
    user_id: 1,
    movie_id: 11,
    rating: 4.5,
});
```

## Performance Considerations

- User-based collaborative filtering: O(U²) where U is the number of users
- Item-based collaborative filtering: O(I²) where I is the number of items
- Content-based filtering: O(I × F) where F is the feature space size
- Hybrid methods combine complexities but provide better recommendations

## Future Enhancements

- [ ] Matrix factorization (SVD, ALS)
- [ ] Deep learning models (neural collaborative filtering)
- [ ] Implicit feedback support (views, clicks)
- [ ] Real-time recommendation updates
- [ ] Scalable storage backend (database integration)
- [ ] REST API for web integration
- [ ] A/B testing framework
- [ ] Explainable recommendations

## License

MIT

## Author

Built with Rust for high-performance recommendation systems.
