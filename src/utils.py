
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_feature_importance(model_pipeline, n_top_features=20, title="Feature Importances"):
    
    # Extract the regressor and preprocessor from the pipeline
    regressor = model_pipeline.named_steps['regressor']
    preprocessor = model_pipeline.named_steps['preprocessor']

    # Check if the regressor has feature_importances_
    if not hasattr(regressor, 'feature_importances_'):
        print(f"Model {type(regressor).__name__} does not have 'feature_importances_'.")
        return

    importances = regressor.feature_importances_

    # Get feature names after preprocessing by ColumnTransformer
    
    try:
        feature_names_out = preprocessor.get_feature_names_out()
    except AttributeError:
        
        print("Warning: get_feature_names_out() not available. Attempting manual feature name reconstruction.")
        all_feature_names = []
        for name, transformer, original_cols in preprocessor.transformers_:
            if name == 'num':
                # Numerical features 
                all_feature_names.extend(original_cols)
            elif name == 'ord':
                # Ordinal features
                all_feature_names.extend(original_cols)
            elif name == 'nom':
                # OneHotEncoder features
                ohe_names = transformer.named_steps['onehot'].get_feature_names_out(original_cols)
                all_feature_names.extend(ohe_names)
            elif name == 'remainder' and transformer != 'drop': 
                pass 

        feature_names_out = all_feature_names

    feature_importance_df = pd.DataFrame({'feature': feature_names_out, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    plt.figure(figsize=(12, max(8, n_top_features * 0.5))) 
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(n_top_features))
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

