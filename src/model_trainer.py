import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib 

def train_and_evaluate_models(X_train, y_train, X_test, y_test, preprocessor):
    
    print("\nStarting model training and evaluation...")

    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42),
        'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse')
    }

    
    param_grids = {
        'LinearRegression': {}, 
        'Ridge': {'regressor__alpha': [0.1, 1.0, 10.0]},
        'Lasso': {'regressor__alpha': [0.0001, 0.001, 0.01]},
        'RandomForest': {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [10, 20, None]
        },
        'XGBoost': {
            'regressor__n_estimators': [100, 200],
            'regressor__learning_rate': [0.05, 0.1],
            'regressor__max_depth': [3, 5]
        }
    }

    results = {}
    best_trained_models = {}

    # Cross-validation strategy
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        # Create a full pipeline including preprocessing and the regressor
        full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('regressor', model)])

        # Set up GridSearchCV
        grid_search = GridSearchCV(
            full_pipeline,
            param_grids[name],
            cv=cv,
            scoring='neg_mean_squared_error', 
            n_jobs=-1, 
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        # Convert negative MSE to RMSE for CV score
        cv_rmse = np.sqrt(-grid_search.best_score_)

        # Make predictions on the test set
        y_pred = best_model.predict(X_test)

        # Calculate evaluation metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results[name] = {
            'best_params': best_params,
            'cv_rmse': cv_rmse,
            'test_r2': r2,
            'test_mae': mae,
            'test_rmse': rmse
        }
        best_trained_models[name] = best_model

        print(f"  Best parameters: {best_params}")
        print(f"  Cross-Validation RMSE: {cv_rmse:.4f}")
        print(f"  Test R²: {r2:.4f}")
        print(f"  Test MAE: {mae:.2f}")
        print(f"  Test RMSE: {rmse:.2f}")

    print("\n--- All Model Training Complete ---")
    for name, res in results.items():
        print(f"\nModel: {name}")
        for key, value in res.items():
            print(f"  {key}: {value}")

    return results, best_trained_models

def save_model(model, file_path):
    
    joblib.dump(model, file_path)
    print(f"\nModel saved to: {file_path}")

def load_model(file_path):
    
    model = joblib.load(file_path)
    print(f"\nModel loaded from: {file_path}")
    return model

if __name__ == '__main__':
    print("This script is designed to be imported. Run the notebooks or a main.py for full execution.")
    