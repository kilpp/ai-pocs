# Examples

## Example 1: Basic Usage

```rust
use price_prediction_system::{
    DataLoader, LinearRegression, Regressor, 
    FeatureEngine, Evaluator
};

fn main() -> anyhow::Result<()> {
    // Generate data
    let properties = DataLoader::generate_synthetic_data(200, 123);
    
    // Convert to arrays
    let (x, y) = DataLoader::to_arrays(&properties);
    
    // Standardize
    let (x_std, _, _) = FeatureEngine::standardize(&x);
    
    // Split
    let (x_train, y_train, x_test, y_test) = 
        FeatureEngine::train_test_split(&x_std, &y, 0.2, 42);
    
    // Train
    let mut model = LinearRegression::new();
    model.fit(&x_train, &y_train)?;
    
    // Predict and evaluate
    let predictions = model.predict(&x_test)?;
    let metrics = Evaluator::evaluate(&y_test, &predictions);
    
    println!("R² Score: {:.4}", metrics.r2_score);
    
    Ok(())
}
```

## Example 2: Comparing All Models

```rust
use price_prediction_system::{
    DataLoader, FeatureEngine, Evaluator,
    LinearRegression, RidgeRegression, 
    LassoRegression, GradientDescentRegression,
    Regressor, ModelComparison
};

fn main() -> anyhow::Result<()> {
    let properties = DataLoader::generate_synthetic_data(500, 42);
    let (x, y) = DataLoader::to_arrays(&properties);
    let (x_std, _, _) = FeatureEngine::standardize(&x);
    let (x_train, y_train, x_test, y_test) = 
        FeatureEngine::train_test_split(&x_std, &y, 0.2, 42);
    
    let mut models: Vec<Box<dyn Regressor>> = vec![
        Box::new(LinearRegression::new()),
        Box::new(RidgeRegression::new(1.0)),
        Box::new(LassoRegression::new(0.1).with_max_iter(500)),
        Box::new(GradientDescentRegression::new(0.01, 1000)),
    ];
    
    for model in models.iter_mut() {
        model.fit(&x_train, &y_train)?;
        let y_pred = model.predict(&x_test)?;
        let metrics = Evaluator::evaluate(&y_test, &y_pred);
        
        println!("{}: R² = {:.4}", model.name(), metrics.r2_score);
    }
    
    Ok(())
}
```

## Example 3: Feature Engineering

```rust
use price_prediction_system::{DataLoader, FeatureEngine};

fn main() -> anyhow::Result<()> {
    let properties = DataLoader::generate_synthetic_data(100, 42);
    let (x, y) = DataLoader::to_arrays(&properties);
    
    // Polynomial features
    let x_poly = FeatureEngine::polynomial_features(&x);
    println!("Original features: {}", x.ncols());
    println!("Polynomial features: {}", x_poly.ncols());
    
    // Custom interactions
    let interactions = vec![
        (0, 1),  // area × bedrooms
        (0, 2),  // area × bathrooms
        (1, 2),  // bedrooms × bathrooms
    ];
    let x_interact = FeatureEngine::add_interactions(&x, &interactions);
    println!("With interactions: {}", x_interact.ncols());
    
    Ok(())
}
```

## Example 4: Working with CSV Data

```rust
use price_prediction_system::{DataLoader, Property};

fn main() -> anyhow::Result<()> {
    // Generate and save
    let properties = DataLoader::generate_synthetic_data(1000, 42);
    DataLoader::save_csv(&properties, "real_estate.csv")?;
    println!("Saved {} properties", properties.len());
    
    // Load from CSV
    let loaded = DataLoader::load_csv("real_estate.csv")?;
    println!("Loaded {} properties", loaded.len());
    
    // Print first property
    if let Some(prop) = loaded.first() {
        println!("Area: {} sqft", prop.area_sqft);
        println!("Bedrooms: {}", prop.bedrooms);
        println!("Price: ${}", prop.price);
    }
    
    Ok(())
}
```

## Example 5: Hyperparameter Tuning

```rust
use price_prediction_system::{
    DataLoader, FeatureEngine, Evaluator,
    RidgeRegression, Regressor
};

fn main() -> anyhow::Result<()> {
    let properties = DataLoader::generate_synthetic_data(500, 42);
    let (x, y) = DataLoader::to_arrays(&properties);
    let (x_std, _, _) = FeatureEngine::standardize(&x);
    let (x_train, y_train, x_test, y_test) = 
        FeatureEngine::train_test_split(&x_std, &y, 0.2, 42);
    
    // Try different alpha values for Ridge
    let alphas = vec![0.01, 0.1, 1.0, 10.0, 100.0];
    
    for alpha in alphas {
        let mut model = RidgeRegression::new(alpha);
        model.fit(&x_train, &y_train)?;
        let y_pred = model.predict(&x_test)?;
        let metrics = Evaluator::evaluate(&y_test, &y_pred);
        
        println!("Alpha {}: R² = {:.4}", alpha, metrics.r2_score);
    }
    
    Ok(())
}
```

## Example 6: Making Predictions

```rust
use price_prediction_system::{
    DataLoader, LinearRegression, Regressor,
    FeatureEngine, Property
};
use ndarray::Array2;

fn main() -> anyhow::Result<()> {
    // Train model
    let properties = DataLoader::generate_synthetic_data(500, 42);
    let (x, y) = DataLoader::to_arrays(&properties);
    let (x_std, means, stds) = FeatureEngine::standardize(&x);
    
    let mut model = LinearRegression::new();
    model.fit(&x_std, &y)?;
    
    // Create new property for prediction
    let new_property = Property {
        area_sqft: 2500.0,
        bedrooms: 3.0,
        bathrooms: 2.0,
        stories: 2.0,
        main_road: 1.0,
        guestroom: 1.0,
        basement: 0.0,
        hot_water_heating: 1.0,
        air_conditioning: 1.0,
        parking: 2.0,
        prefarea: 1.0,
        furnishing_status: 2.0,
        price: 0.0, // Unknown
    };
    
    // Convert to feature vector
    let features = new_property.to_features();
    
    // Standardize using saved means and stds
    let mut std_features = features.clone();
    for (i, &feat) in features.iter().enumerate() {
        std_features[i] = (feat - means[i]) / stds[i];
    }
    
    // Create array and predict
    let x_new = Array2::from_shape_vec((1, features.len()), std_features)?;
    let prediction = model.predict(&x_new)?;
    
    println!("Predicted price: ${:.2}", prediction[0]);
    
    Ok(())
}
```
