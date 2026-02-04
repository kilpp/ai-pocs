# Real Estate Price Prediction System

A comprehensive real estate price prediction system built in Rust, featuring multiple regression algorithms, advanced feature engineering, and model comparison capabilities.

## Features

### 🎯 Multiple Regression Algorithms
- **Linear Regression** - Classic OLS using normal equation
- **Ridge Regression** - L2 regularization for better generalization
- **Lasso Regression** - L1 regularization with feature selection via coordinate descent
- **Gradient Descent Regression** - Iterative optimization approach

### 🔧 Feature Engineering
- **Normalization** - Min-max scaling
- **Standardization** - Z-score normalization
- **Polynomial Features** - Generate degree-2 polynomial features
- **Interaction Features** - Create custom feature interactions
- **Train/Test Split** - Randomized data splitting with seeding

### 📊 Model Evaluation
- **R² Score** - Coefficient of determination
- **MSE** - Mean Squared Error
- **RMSE** - Root Mean Squared Error
- **MAE** - Mean Absolute Error

### 📁 Data Handling
- **CSV Import/Export** - Load and save real estate data
- **Synthetic Data Generation** - Create realistic test datasets
- **Array Conversion** - Efficient ndarray-based processing

## Project Structure

```
price-prediction-system/
├── src/
│   ├── main.rs                 # Main application with examples
│   ├── lib.rs                  # Library exports
│   ├── models.rs               # Data models and structures
│   ├── error.rs                # Error types
│   ├── data.rs                 # Data loading and generation
│   ├── feature_engineering.rs  # Feature preprocessing
│   ├── regression.rs           # Regression algorithms
│   └── evaluation.rs           # Model evaluation metrics
├── Cargo.toml                  # Project dependencies
└── README.md                   # This file
```

## Real Estate Features

The system predicts property prices based on 12 features:

1. **area_sqft** - Property area in square feet
2. **bedrooms** - Number of bedrooms
3. **bathrooms** - Number of bathrooms
4. **stories** - Number of stories/floors
5. **main_road** - Adjacent to main road (binary)
6. **guestroom** - Has guest room (binary)
7. **basement** - Has basement (binary)
8. **hot_water_heating** - Has hot water heating (binary)
9. **air_conditioning** - Has air conditioning (binary)
10. **parking** - Number of parking spots
11. **prefarea** - In preferred area (binary)
12. **furnishing_status** - 0=unfurnished, 1=semi-furnished, 2=furnished

## Installation

### Prerequisites
- Rust 1.70+ (install from [rustup.rs](https://rustup.rs/))

### Build

```bash
# Clone or navigate to the project directory
cd price-prediction-system

# Build the project
cargo build --release

# Run the application
cargo run --release
```

## Usage

### Running the Demo

The main application demonstrates all features:

```bash
cargo run --release
```

This will:
1. Generate 500 synthetic property records
2. Save sample data to `sample_data.csv`
3. Preprocess and normalize features
4. Split data into train/test sets (80/20)
5. Train 4 different regression models
6. Evaluate and compare all models
7. Display results and best performing model

### Using as a Library

```rust
use price_prediction_system::{
    DataLoader, Evaluator, FeatureEngine, 
    LinearRegression, Regressor
};

// Load data
let properties = DataLoader::load_csv("data.csv")?;
let (x, y) = DataLoader::to_arrays(&properties);

// Preprocess
let (x_norm, _, _) = FeatureEngine::standardize(&x);

// Split data
let (x_train, y_train, x_test, y_test) = 
    FeatureEngine::train_test_split(&x_norm, &y, 0.2, 42);

// Train model
let mut model = LinearRegression::new();
model.fit(&x_train, &y_train)?;

// Predict
let predictions = model.predict(&x_test)?;

// Evaluate
let metrics = Evaluator::evaluate(&y_test, &predictions);
println!("R² Score: {:.4}", metrics.r2_score);
```

### Creating Custom Data

```rust
use price_prediction_system::{DataLoader, Property};

// Generate synthetic data
let properties = DataLoader::generate_synthetic_data(1000, 42);

// Save to CSV
DataLoader::save_csv(&properties, "my_data.csv")?;

// Load from CSV
let loaded = DataLoader::load_csv("my_data.csv")?;
```

### Advanced Feature Engineering

```rust
use price_prediction_system::FeatureEngine;

// Polynomial features (degree 2)
let x_poly = FeatureEngine::polynomial_features(&x);

// Custom interactions
let interactions = vec![(0, 1), (0, 2)]; // area×bedrooms, area×bathrooms
let x_interact = FeatureEngine::add_interactions(&x, &interactions);

// Normalization
let (x_norm, mins, maxs) = FeatureEngine::normalize(&x);

// Standardization
let (x_std, means, stds) = FeatureEngine::standardize(&x);
```

### Comparing Models

```rust
use price_prediction_system::{
    LinearRegression, RidgeRegression, LassoRegression,
    Regressor, Evaluator, ModelComparison
};

let mut models: Vec<Box<dyn Regressor>> = vec![
    Box::new(LinearRegression::new()),
    Box::new(RidgeRegression::new(1.0)),
    Box::new(LassoRegression::new(0.1)),
];

let mut results = Vec::new();

for model in models.iter_mut() {
    model.fit(&x_train, &y_train)?;
    let predictions = model.predict(&x_test)?;
    let metrics = Evaluator::evaluate(&y_test, &predictions);
    
    results.push(ModelComparison {
        model_name: model.name().to_string(),
        metrics,
    });
}

// Find best model by R²
let best = results.iter()
    .max_by(|a, b| a.metrics.r2_score.partial_cmp(&b.metrics.r2_score).unwrap())
    .unwrap();
```

## Algorithm Details

### Linear Regression
Uses the normal equation: **w = (X^T X)^(-1) X^T y**
- Pros: Exact solution, fast for small datasets
- Cons: Can be numerically unstable, no regularization

### Ridge Regression
Adds L2 penalty: **w = (X^T X + αI)^(-1) X^T y**
- Pros: Prevents overfitting, handles multicollinearity
- Cons: Doesn't perform feature selection

### Lasso Regression
Coordinate descent with L1 penalty
- Pros: Feature selection, sparse solutions
- Cons: Slower convergence, can be unstable

### Gradient Descent
Iterative optimization: **w = w - α∇J(w)**
- Pros: Scales to large datasets, flexible
- Cons: Requires tuning, can converge slowly

## Performance Tips

1. **Data Scaling**: Always normalize or standardize features before training
2. **Regularization**: Use Ridge/Lasso for datasets with many features
3. **Train/Test Split**: Use at least 20% for testing
4. **Hyperparameters**: 
   - Ridge: α ∈ [0.1, 10]
   - Lasso: α ∈ [0.01, 1]
   - Gradient Descent: learning_rate ∈ [0.001, 0.1]

## Example Output

```
=== Real Estate Price Prediction System ===

Generating synthetic real estate data...
Generated 500 properties

Feature matrix shape: 500 x 12
Train set: 400 samples
Test set: 100 samples

=== Training Models ===

Training Linear Regression...
  R² Score: 0.9845
  RMSE: 45231.23
  MAE: 32104.56

Training Ridge Regression...
  R² Score: 0.9842
  RMSE: 45567.89
  MAE: 32890.12

=== Model Comparison ===

Best Model: Linear Regression
  R² Score: 0.9845
  RMSE: 45231.23

Model                          R²           MSE          RMSE         MAE         
------------------------------------------------------------------------------
Linear Regression              0.9845       2045867234   45231.23     32104.56    
Ridge Regression               0.9842       2076234567   45567.89     32890.12    
```

## Dependencies

- **ndarray** - N-dimensional arrays
- **ndarray-stats** - Statistical functions
- **ndarray-rand** - Random array generation
- **csv** - CSV reading/writing
- **serde** - Serialization
- **rand** - Random number generation
- **anyhow** - Error handling
- **thiserror** - Error derive macros

## Testing

```bash
# Run tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_linear_regression
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## Future Enhancements

- [ ] k-Fold cross-validation
- [ ] Grid search for hyperparameter tuning
- [ ] Decision tree and random forest regressors
- [ ] Neural network regression
- [ ] Feature importance analysis
- [ ] Visualization of predictions vs actuals
- [ ] Model persistence (save/load trained models)
- [ ] Real-time prediction API
- [ ] Web interface

## Author

Built with Rust 🦀
