mod models;
mod collaborative_filtering;
mod content_based;
mod hybrid;
mod sample_data;

use collaborative_filtering::CollaborativeFilter;
use content_based::ContentBasedFilter;
use hybrid::{HybridRecommender, HybridStrategy};
use sample_data::create_sample_dataset;

fn main() {
    println!("=== Movie Recommendation System ===\n");

    // Create sample dataset
    let dataset = create_sample_dataset();
    println!("Loaded {} movies and {} users with {} ratings\n", 
             dataset.movies.len(), 
             dataset.users.len(), 
             dataset.ratings.len());

    // Test user
    let test_user_id = 1;
    let test_user_name = dataset.users.get(&test_user_id).unwrap().name.clone();
    println!("Generating recommendations for user: {} (ID: {})", test_user_name, test_user_id);
    
    // Display user's existing ratings
    println!("\n{}'s Ratings:", test_user_name);
    let user_ratings = dataset.get_user_ratings(test_user_id);
    for rating in user_ratings {
        let movie = dataset.movies.get(&rating.movie_id).unwrap();
        println!("  - {} ({}) - Rating: {:.1}", movie.title, movie.year, rating.rating);
    }

    // 1. Collaborative Filtering (User-based)
    println!("\n--- Collaborative Filtering (User-based) ---");
    let collab_filter = CollaborativeFilter::new(&dataset);
    let collab_recs = collab_filter.recommend(test_user_id, 5);
    display_recommendations(&dataset, &collab_recs);

    // 2. Collaborative Filtering (Item-based)
    println!("\n--- Collaborative Filtering (Item-based) ---");
    let item_recs = collab_filter.recommend_item_based(test_user_id, 5);
    display_recommendations(&dataset, &item_recs);

    // 3. Content-Based Filtering
    println!("\n--- Content-Based Filtering ---");
    let content_filter = ContentBasedFilter::new(&dataset);
    let content_recs = content_filter.recommend(test_user_id, 5);
    display_recommendations(&dataset, &content_recs);

    // 4. Hybrid Recommender - Weighted
    println!("\n--- Hybrid Recommender (Weighted: 70% Collaborative, 30% Content) ---");
    let hybrid = HybridRecommender::new(&dataset);
    let hybrid_weighted = hybrid.recommend(
        test_user_id,
        5,
        HybridStrategy::Weighted {
            collaborative_weight: 0.7,
            content_weight: 0.3,
        },
    );
    display_recommendations(&dataset, &hybrid_weighted);

    // 5. Hybrid Recommender - Switching
    println!("\n--- Hybrid Recommender (Switching Strategy) ---");
    let hybrid_switching = hybrid.recommend(test_user_id, 5, HybridStrategy::Switching);
    display_recommendations(&dataset, &hybrid_switching);

    // 6. Hybrid Recommender - Mixed
    println!("\n--- Hybrid Recommender (Mixed Strategy) ---");
    let hybrid_mixed = hybrid.recommend(test_user_id, 5, HybridStrategy::Mixed);
    display_recommendations(&dataset, &hybrid_mixed);

    // 7. Content-based similar movies
    println!("\n--- Similar Movies to 'The Dark Knight' ---");
    let similar_movies = content_filter.find_similar_movies(3, 5);
    display_recommendations(&dataset, &similar_movies);

    println!("\n=== Recommendation System Demo Complete ===");
}

fn display_recommendations(dataset: &models::Dataset, recommendations: &[(u32, f64)]) {
    if recommendations.is_empty() {
        println!("  No recommendations available.");
        return;
    }

    for (i, (movie_id, score)) in recommendations.iter().enumerate() {
        if let Some(movie) = dataset.movies.get(movie_id) {
            println!(
                "  {}. {} ({}) - Score: {:.3}",
                i + 1,
                movie.title,
                movie.year,
                score
            );
            println!("     Genres: {}", movie.genres.join(", "));
        }
    }
}
