use csv::ReaderBuilder;
use ndarray::{Array1, Array2};
use std::path::Path;

use crate::error::Result;
use crate::models::Property;

/// Data loader for real estate datasets
pub struct DataLoader;

impl DataLoader {
    /// Load properties from CSV file
    pub fn load_csv<P: AsRef<Path>>(path: P) -> Result<Vec<Property>> {
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)?;

        let mut properties = Vec::new();

        for result in reader.deserialize() {
            let property: Property = result?;
            properties.push(property);
        }

        Ok(properties)
    }

    /// Convert properties to feature matrix and target vector
    pub fn to_arrays(properties: &[Property]) -> (Array2<f64>, Array1<f64>) {
        let n_samples = properties.len();
        let n_features = Property::feature_count();

        let mut x = Array2::<f64>::zeros((n_samples, n_features));
        let mut y = Array1::<f64>::zeros(n_samples);

        for (i, property) in properties.iter().enumerate() {
            let features = property.to_features();
            for (j, &feature) in features.iter().enumerate() {
                x[[i, j]] = feature;
            }
            y[i] = property.price;
        }

        (x, y)
    }

    /// Generate synthetic real estate data for testing
    pub fn generate_synthetic_data(n_samples: usize, seed: u64) -> Vec<Property> {
        use rand::Rng;
        use rand::SeedableRng;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut properties = Vec::new();

        for _ in 0..n_samples {
            let area_sqft = rng.gen_range(500.0..5000.0);
            let bedrooms = rng.gen_range(1.0..6.0);
            let bathrooms = rng.gen_range(1.0..4.0);
            let stories = rng.gen_range(1.0..4.0);
            let main_road = if rng.gen_bool(0.7) { 1.0 } else { 0.0 };
            let guestroom = if rng.gen_bool(0.3) { 1.0 } else { 0.0 };
            let basement = if rng.gen_bool(0.4) { 1.0 } else { 0.0 };
            let hot_water_heating = if rng.gen_bool(0.6) { 1.0 } else { 0.0 };
            let air_conditioning = if rng.gen_bool(0.8) { 1.0 } else { 0.0 };
            let parking = rng.gen_range(0.0..4.0);
            let prefarea = if rng.gen_bool(0.5) { 1.0 } else { 0.0 };
            let furnishing_status: f64 = rng.gen_range(0.0_f64..3.0_f64).floor();

            // Generate price based on features with some noise
            let base_price = area_sqft * 100.0
                + bedrooms * 50000.0
                + bathrooms * 30000.0
                + stories * 20000.0
                + main_road * 100000.0
                + guestroom * 50000.0
                + basement * 40000.0
                + hot_water_heating * 10000.0
                + air_conditioning * 50000.0
                + parking * 25000.0
                + prefarea * 150000.0
                + furnishing_status * 75000.0;

            let noise = rng.gen_range(-50000.0..50000.0);
            let price: f64 = (base_price + noise).max(50000.0_f64);

            properties.push(Property {
                area_sqft,
                bedrooms,
                bathrooms,
                stories,
                main_road,
                guestroom,
                basement,
                hot_water_heating,
                air_conditioning,
                parking,
                prefarea,
                furnishing_status,
                price,
            });
        }

        properties
    }

    /// Save properties to CSV file
    pub fn save_csv<P: AsRef<Path>>(properties: &[Property], path: P) -> Result<()> {
        let mut writer = csv::Writer::from_path(path)?;

        for property in properties {
            writer.serialize(property)?;
        }

        writer.flush()?;
        Ok(())
    }
}
