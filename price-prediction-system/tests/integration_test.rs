#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2};
    use price_prediction_system::*;

    #[test]
    fn test_linear_regression() {
        // Simple test case: y = 2x + 1
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0]);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let predictions = model.predict(&x).unwrap();

        // Check predictions are close to actual values
        for (pred, actual) in predictions.iter().zip(y.iter()) {
            assert_abs_diff_eq!(pred, actual, epsilon = 0.01);
        }
    }

    #[test]
    fn test_ridge_regression() {
        let x = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0]).unwrap();
        let y = Array1::from_vec(vec![5.0, 8.0, 11.0, 14.0]);

        let mut model = RidgeRegression::new(0.1);
        model.fit(&x, &y).unwrap();

        let predictions = model.predict(&x).unwrap();

        // Model should fit reasonably well
        let metrics = Evaluator::evaluate(&y, &predictions);
        assert!(metrics.r2_score > 0.9);
    }

    #[test]
    fn test_r2_score() {
        let y_true = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0]);
        let y_pred = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0]);

        let r2 = Evaluator::r2_score(&y_true, &y_pred);
        assert_abs_diff_eq!(r2, 1.0, epsilon = 0.001);
    }

    #[test]
    fn test_mse() {
        let y_true = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0]);
        let y_pred = Array1::from_vec(vec![2.0, 5.0, 8.0, 10.0]);

        let mse = Evaluator::mse(&y_true, &y_pred);
        // MSE = [(1)^2 + (0)^2 + (1)^2 + (1)^2] / 4 = 0.75
        assert_abs_diff_eq!(mse, 0.75, epsilon = 0.001);
    }

    #[test]
    fn test_mae() {
        let y_true = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0]);
        let y_pred = Array1::from_vec(vec![2.0, 5.0, 8.0, 10.0]);

        let mae = Evaluator::mae(&y_true, &y_pred);
        // MAE = [1 + 0 + 1 + 1] / 4 = 0.75
        assert_abs_diff_eq!(mae, 0.75, epsilon = 0.001);
    }

    #[test]
    fn test_normalize() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let (normalized, mins, maxs) = FeatureEngine::normalize(&data);

        // First column: min=1, max=5, range=4
        assert_abs_diff_eq!(normalized[[0, 0]], 0.0, epsilon = 0.001); // (1-1)/4
        assert_abs_diff_eq!(normalized[[1, 0]], 0.5, epsilon = 0.001); // (3-1)/4
        assert_abs_diff_eq!(normalized[[2, 0]], 1.0, epsilon = 0.001); // (5-1)/4

        assert_eq!(mins.len(), 2);
        assert_eq!(maxs.len(), 2);
    }

    #[test]
    fn test_standardize() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let (standardized, means, stds) = FeatureEngine::standardize(&data);

        // Check that mean is approximately 0 and std is approximately 1
        for col in 0..standardized.ncols() {
            let column = standardized.column(col);
            let mean = column.mean().unwrap();
            assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-10);
        }

        assert_eq!(means.len(), 2);
        assert_eq!(stds.len(), 2);
    }

    #[test]
    fn test_polynomial_features() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let poly = FeatureEngine::polynomial_features(&data);

        // Original (2) + Squared (2) + Interactions (1) = 5 features
        assert_eq!(poly.ncols(), 5);
        assert_eq!(poly.nrows(), 2);

        // Check first row: [1, 2] -> [1, 2, 1, 4, 2]
        assert_abs_diff_eq!(poly[[0, 0]], 1.0, epsilon = 0.001);
        assert_abs_diff_eq!(poly[[0, 1]], 2.0, epsilon = 0.001);
        assert_abs_diff_eq!(poly[[0, 2]], 1.0, epsilon = 0.001); // 1^2
        assert_abs_diff_eq!(poly[[0, 3]], 4.0, epsilon = 0.001); // 2^2
        assert_abs_diff_eq!(poly[[0, 4]], 2.0, epsilon = 0.001); // 1*2
    }

    #[test]
    fn test_add_interactions() {
        let data = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let interactions = vec![(0, 1), (1, 2)];

        let result = FeatureEngine::add_interactions(&data, &interactions);

        // Original 3 + 2 interactions = 5 features
        assert_eq!(result.ncols(), 5);

        // Check first row interactions
        assert_abs_diff_eq!(result[[0, 3]], 2.0, epsilon = 0.001); // 1*2
        assert_abs_diff_eq!(result[[0, 4]], 6.0, epsilon = 0.001); // 2*3
    }

    #[test]
    fn test_train_test_split() {
        let x = Array2::from_shape_vec((10, 2), vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
        ]).unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        let (x_train, y_train, x_test, y_test) =
            FeatureEngine::train_test_split(&x, &y, 0.3, 42);

        assert_eq!(x_train.nrows(), 7);
        assert_eq!(x_test.nrows(), 3);
        assert_eq!(y_train.len(), 7);
        assert_eq!(y_test.len(), 3);
    }

    #[test]
    fn test_data_generation() {
        let properties = DataLoader::generate_synthetic_data(100, 42);

        assert_eq!(properties.len(), 100);

        // Check that prices are reasonable
        for prop in &properties {
            assert!(prop.price > 0.0);
            assert!(prop.area_sqft >= 500.0 && prop.area_sqft < 5000.0);
            assert!(prop.bedrooms >= 1.0 && prop.bedrooms < 6.0);
        }
    }

    #[test]
    fn test_property_to_features() {
        let prop = Property {
            area_sqft: 2000.0,
            bedrooms: 3.0,
            bathrooms: 2.0,
            stories: 2.0,
            main_road: 1.0,
            guestroom: 0.0,
            basement: 1.0,
            hot_water_heating: 1.0,
            air_conditioning: 1.0,
            parking: 2.0,
            prefarea: 1.0,
            furnishing_status: 2.0,
            price: 500000.0,
        };

        let features = prop.to_features();

        assert_eq!(features.len(), Property::feature_count());
        assert_eq!(features[0], 2000.0);
        assert_eq!(features[1], 3.0);
        assert_eq!(features[11], 2.0);
    }

    #[test]
    fn test_to_arrays() {
        let properties = DataLoader::generate_synthetic_data(50, 123);
        let (x, y) = DataLoader::to_arrays(&properties);

        assert_eq!(x.nrows(), 50);
        assert_eq!(x.ncols(), Property::feature_count());
        assert_eq!(y.len(), 50);

        // Verify first property matches
        let first_features = properties[0].to_features();
        for (i, &feat) in first_features.iter().enumerate() {
            assert_abs_diff_eq!(x[[0, i]], feat, epsilon = 0.001);
        }
        assert_abs_diff_eq!(y[0], properties[0].price, epsilon = 0.001);
    }

    #[test]
    fn test_gradient_descent_regression() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

        let mut model = GradientDescentRegression::new(0.01, 1000);
        model.fit(&x, &y).unwrap();

        let predictions = model.predict(&x).unwrap();

        // Should fit well after many iterations
        let metrics = Evaluator::evaluate(&y, &predictions);
        assert!(metrics.r2_score > 0.99);
    }

    #[test]
    fn test_lasso_regression() {
        let x = Array2::from_shape_vec((6, 2), vec![
            1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0,
        ]).unwrap();
        let y = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0]);

        let mut model = LassoRegression::new(0.01).with_max_iter(500);
        model.fit(&x, &y).unwrap();

        let predictions = model.predict(&x).unwrap();

        // Should achieve reasonable fit
        let metrics = Evaluator::evaluate(&y, &predictions);
        assert!(metrics.r2_score > 0.85);
    }
}
