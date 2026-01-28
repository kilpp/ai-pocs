use crate::models::Dataset;
use std::collections::{HashMap, HashSet};

pub struct CollaborativeFilter<'a> {
    dataset: &'a Dataset,
}

impl<'a> CollaborativeFilter<'a> {
    pub fn new(dataset: &'a Dataset) -> Self {
        CollaborativeFilter { dataset }
    }

    /// Calculate Pearson correlation coefficient between two users
    fn pearson_correlation(&self, user1_id: u32, user2_id: u32) -> f64 {
        let user1_ratings = self.get_user_rating_map(user1_id);
        let user2_ratings = self.get_user_rating_map(user2_id);

        // Find common movies
        let common_movies: HashSet<u32> = user1_ratings
            .keys()
            .filter(|k| user2_ratings.contains_key(k))
            .copied()
            .collect();

        if common_movies.len() < 2 {
            return 0.0;
        }

        let mut sum1 = 0.0;
        let mut sum2 = 0.0;
        let mut sum1_sq = 0.0;
        let mut sum2_sq = 0.0;
        let mut p_sum = 0.0;
        let n = common_movies.len() as f64;

        for movie_id in common_movies {
            let rating1 = user1_ratings[&movie_id];
            let rating2 = user2_ratings[&movie_id];

            sum1 += rating1;
            sum2 += rating2;
            sum1_sq += rating1 * rating1;
            sum2_sq += rating2 * rating2;
            p_sum += rating1 * rating2;
        }

        let num = p_sum - (sum1 * sum2 / n);
        let den = ((sum1_sq - sum1 * sum1 / n) * (sum2_sq - sum2 * sum2 / n)).sqrt();

        if den == 0.0 {
            0.0
        } else {
            num / den
        }
    }

    /// Get user ratings as a map
    fn get_user_rating_map(&self, user_id: u32) -> HashMap<u32, f64> {
        self.dataset
            .get_user_ratings(user_id)
            .iter()
            .map(|r| (r.movie_id, r.rating))
            .collect()
    }

    /// Find k most similar users to the given user
    fn find_similar_users(&self, user_id: u32, k: usize) -> Vec<(u32, f64)> {
        let mut similarities: Vec<(u32, f64)> = self
            .dataset
            .users
            .keys()
            .filter(|&&id| id != user_id)
            .map(|&other_id| (other_id, self.pearson_correlation(user_id, other_id)))
            .filter(|(_, sim)| *sim > 0.0)
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(k);
        similarities
    }

    /// Recommend movies using user-based collaborative filtering
    pub fn recommend(&self, user_id: u32, n: usize) -> Vec<(u32, f64)> {
        let similar_users = self.find_similar_users(user_id, 10);
        
        if similar_users.is_empty() {
            return Vec::new();
        }

        let user_rated_movies: HashSet<u32> = self
            .dataset
            .get_user_ratings(user_id)
            .iter()
            .map(|r| r.movie_id)
            .collect();

        let mut movie_scores: HashMap<u32, (f64, f64)> = HashMap::new(); // (weighted_sum, similarity_sum)

        for (similar_user_id, similarity) in similar_users {
            let ratings = self.get_user_rating_map(similar_user_id);
            for (movie_id, rating) in ratings {
                if !user_rated_movies.contains(&movie_id) {
                    let entry = movie_scores.entry(movie_id).or_insert((0.0, 0.0));
                    entry.0 += rating * similarity;
                    entry.1 += similarity;
                }
            }
        }

        let mut recommendations: Vec<(u32, f64)> = movie_scores
            .iter()
            .map(|(&movie_id, &(weighted_sum, sim_sum))| {
                (movie_id, weighted_sum / sim_sum)
            })
            .collect();

        recommendations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        recommendations.truncate(n);
        recommendations
    }

    /// Calculate item-item similarity using cosine similarity
    fn cosine_similarity(&self, movie1_id: u32, movie2_id: u32) -> f64 {
        let movie1_ratings = self.get_movie_rating_map(movie1_id);
        let movie2_ratings = self.get_movie_rating_map(movie2_id);

        let common_users: HashSet<u32> = movie1_ratings
            .keys()
            .filter(|k| movie2_ratings.contains_key(k))
            .copied()
            .collect();

        if common_users.is_empty() {
            return 0.0;
        }

        let mut dot_product = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;

        for user_id in common_users {
            let rating1 = movie1_ratings[&user_id];
            let rating2 = movie2_ratings[&user_id];
            dot_product += rating1 * rating2;
            norm1 += rating1 * rating1;
            norm2 += rating2 * rating2;
        }

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1.sqrt() * norm2.sqrt())
        }
    }

    /// Get movie ratings as a map
    fn get_movie_rating_map(&self, movie_id: u32) -> HashMap<u32, f64> {
        self.dataset
            .get_movie_ratings(movie_id)
            .iter()
            .map(|r| (r.user_id, r.rating))
            .collect()
    }

    /// Recommend movies using item-based collaborative filtering
    pub fn recommend_item_based(&self, user_id: u32, n: usize) -> Vec<(u32, f64)> {
        let user_ratings = self.get_user_rating_map(user_id);
        
        if user_ratings.is_empty() {
            return Vec::new();
        }

        let mut movie_scores: HashMap<u32, (f64, f64)> = HashMap::new(); // (weighted_sum, similarity_sum)

        for (&rated_movie_id, &user_rating) in &user_ratings {
            for &candidate_movie_id in self.dataset.movies.keys() {
                if !user_ratings.contains_key(&candidate_movie_id) {
                    let similarity = self.cosine_similarity(rated_movie_id, candidate_movie_id);
                    if similarity > 0.0 {
                        let entry = movie_scores.entry(candidate_movie_id).or_insert((0.0, 0.0));
                        entry.0 += user_rating * similarity;
                        entry.1 += similarity;
                    }
                }
            }
        }

        let mut recommendations: Vec<(u32, f64)> = movie_scores
            .iter()
            .map(|(&movie_id, &(weighted_sum, sim_sum))| {
                (movie_id, weighted_sum / sim_sum)
            })
            .collect();

        recommendations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        recommendations.truncate(n);
        recommendations
    }
}
