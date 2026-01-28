use crate::models::{Dataset, Movie, Rating, User};

pub fn create_sample_dataset() -> Dataset {
    let mut dataset = Dataset::new();

    // Add users
    dataset.add_user(User { id: 1, name: "Alice".to_string() });
    dataset.add_user(User { id: 2, name: "Bob".to_string() });
    dataset.add_user(User { id: 3, name: "Charlie".to_string() });
    dataset.add_user(User { id: 4, name: "Diana".to_string() });
    dataset.add_user(User { id: 5, name: "Eve".to_string() });

    // Add movies
    dataset.add_movie(Movie {
        id: 1,
        title: "The Shawshank Redemption".to_string(),
        genres: vec!["Drama".to_string()],
        year: 1994,
        director: "Frank Darabont".to_string(),
        actors: vec!["Tim Robbins".to_string(), "Morgan Freeman".to_string()],
    });

    dataset.add_movie(Movie {
        id: 2,
        title: "The Godfather".to_string(),
        genres: vec!["Crime".to_string(), "Drama".to_string()],
        year: 1972,
        director: "Francis Ford Coppola".to_string(),
        actors: vec!["Marlon Brando".to_string(), "Al Pacino".to_string()],
    });

    dataset.add_movie(Movie {
        id: 3,
        title: "The Dark Knight".to_string(),
        genres: vec!["Action".to_string(), "Crime".to_string(), "Drama".to_string()],
        year: 2008,
        director: "Christopher Nolan".to_string(),
        actors: vec!["Christian Bale".to_string(), "Heath Ledger".to_string()],
    });

    dataset.add_movie(Movie {
        id: 4,
        title: "Pulp Fiction".to_string(),
        genres: vec!["Crime".to_string(), "Drama".to_string()],
        year: 1994,
        director: "Quentin Tarantino".to_string(),
        actors: vec!["John Travolta".to_string(), "Samuel L. Jackson".to_string()],
    });

    dataset.add_movie(Movie {
        id: 5,
        title: "Forrest Gump".to_string(),
        genres: vec!["Drama".to_string(), "Romance".to_string()],
        year: 1994,
        director: "Robert Zemeckis".to_string(),
        actors: vec!["Tom Hanks".to_string(), "Robin Wright".to_string()],
    });

    dataset.add_movie(Movie {
        id: 6,
        title: "Inception".to_string(),
        genres: vec!["Action".to_string(), "Sci-Fi".to_string(), "Thriller".to_string()],
        year: 2010,
        director: "Christopher Nolan".to_string(),
        actors: vec!["Leonardo DiCaprio".to_string(), "Joseph Gordon-Levitt".to_string()],
    });

    dataset.add_movie(Movie {
        id: 7,
        title: "The Matrix".to_string(),
        genres: vec!["Action".to_string(), "Sci-Fi".to_string()],
        year: 1999,
        director: "Lana Wachowski".to_string(),
        actors: vec!["Keanu Reeves".to_string(), "Laurence Fishburne".to_string()],
    });

    dataset.add_movie(Movie {
        id: 8,
        title: "Goodfellas".to_string(),
        genres: vec!["Crime".to_string(), "Drama".to_string()],
        year: 1990,
        director: "Martin Scorsese".to_string(),
        actors: vec!["Robert De Niro".to_string(), "Ray Liotta".to_string()],
    });

    dataset.add_movie(Movie {
        id: 9,
        title: "Interstellar".to_string(),
        genres: vec!["Adventure".to_string(), "Drama".to_string(), "Sci-Fi".to_string()],
        year: 2014,
        director: "Christopher Nolan".to_string(),
        actors: vec!["Matthew McConaughey".to_string(), "Anne Hathaway".to_string()],
    });

    dataset.add_movie(Movie {
        id: 10,
        title: "The Silence of the Lambs".to_string(),
        genres: vec!["Crime".to_string(), "Drama".to_string(), "Thriller".to_string()],
        year: 1991,
        director: "Jonathan Demme".to_string(),
        actors: vec!["Jodie Foster".to_string(), "Anthony Hopkins".to_string()],
    });

    // Add ratings (explicit feedback: 1.0 to 5.0)
    // Alice's ratings - likes crime dramas
    dataset.add_rating(Rating { user_id: 1, movie_id: 2, rating: 5.0 });
    dataset.add_rating(Rating { user_id: 1, movie_id: 4, rating: 4.5 });
    dataset.add_rating(Rating { user_id: 1, movie_id: 8, rating: 4.5 });
    dataset.add_rating(Rating { user_id: 1, movie_id: 10, rating: 4.0 });

    // Bob's ratings - likes Nolan films and sci-fi
    dataset.add_rating(Rating { user_id: 2, movie_id: 3, rating: 5.0 });
    dataset.add_rating(Rating { user_id: 2, movie_id: 6, rating: 5.0 });
    dataset.add_rating(Rating { user_id: 2, movie_id: 9, rating: 4.5 });
    dataset.add_rating(Rating { user_id: 2, movie_id: 7, rating: 4.0 });

    // Charlie's ratings - diverse taste, prefers classics
    dataset.add_rating(Rating { user_id: 3, movie_id: 1, rating: 5.0 });
    dataset.add_rating(Rating { user_id: 3, movie_id: 2, rating: 5.0 });
    dataset.add_rating(Rating { user_id: 3, movie_id: 5, rating: 4.5 });
    dataset.add_rating(Rating { user_id: 3, movie_id: 4, rating: 4.0 });
    dataset.add_rating(Rating { user_id: 3, movie_id: 8, rating: 4.5 });

    // Diana's ratings - likes action and sci-fi
    dataset.add_rating(Rating { user_id: 4, movie_id: 3, rating: 4.5 });
    dataset.add_rating(Rating { user_id: 4, movie_id: 6, rating: 5.0 });
    dataset.add_rating(Rating { user_id: 4, movie_id: 7, rating: 4.5 });
    dataset.add_rating(Rating { user_id: 4, movie_id: 9, rating: 4.0 });

    // Eve's ratings - likes drama
    dataset.add_rating(Rating { user_id: 5, movie_id: 1, rating: 5.0 });
    dataset.add_rating(Rating { user_id: 5, movie_id: 2, rating: 4.5 });
    dataset.add_rating(Rating { user_id: 5, movie_id: 5, rating: 5.0 });
    dataset.add_rating(Rating { user_id: 5, movie_id: 10, rating: 4.0 });

    dataset
}
