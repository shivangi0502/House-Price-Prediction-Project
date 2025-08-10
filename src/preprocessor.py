
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import re

def build_preprocessor(X_raw):

    print("Starting feature engineering and preprocessor building...")

    X_processed = X_raw.copy()
    print("\nDEBUG (preprocessor.py): Columns in X_processed after initial copy:", X_processed.columns.tolist())

    if 'lot_area' in X_processed.columns:
        X_processed['lot_area_bins'] = pd.qcut(X_processed['lot_area'], q=4, labels=False, duplicates='drop')
        X_processed.drop('lot_area', axis=1, inplace=True)
        print("Binned 'lot_area' into 'lot_area_bins' and dropped original 'lot_area' column.")
    elif 'lot_area_log' in X_processed.columns:
        X_processed['lot_area_bins'] = pd.qcut(X_processed['lot_area_log'], q=4, labels=False, duplicates='drop')
        X_processed.drop('lot_area_log', axis=1, inplace=True)
        print("Binned 'lot_area_log' into 'lot_area_bins' and dropped original 'lot_area_log' column.")
    else:
        print("Could not find 'lot_area' or 'lot_area_log' for binning. 'lot_area_bins' will not be created.")
    print("\nDEBUG (preprocessor.py): Columns in X_processed after immediate lot_area handling:", X_processed.columns.tolist())


    # --- Feature Engineering:

    # 1. Create 'House Age' feature and drop 'year_built'/'yr_built'
    if 'yr_sold' in X_processed.columns:
        current_year = X_processed['yr_sold'].max()
        if 'year_built' in X_processed.columns:
            X_processed['house_age'] = current_year - X_processed['year_built']
            X_processed.drop('year_built', axis=1, inplace=True)
            print("Created 'house_age' and dropped 'year_built'.")
        elif 'yr_built' in X_processed.columns:
            X_processed['house_age'] = current_year - X_processed['yr_built']
            X_processed.drop('yr_built', axis=1, inplace=True)
            print("Created 'house_age' and dropped 'yr_built'.")
        else:
            print("Warning: 'year_built' or 'yr_built' not found for 'house_age' creation.")
    else:
        print("Warning: 'yr_sold' not found for 'house_age' calculation.")


    # 2. Example Interaction Term: Overall Quality * Gr Living Area
    gr_liv_area_col = 'gr_liv_area' 
    if 'gr_liv_area_log' in X_processed.columns:
        gr_liv_area_col = 'gr_liv_area_log'

    if 'overall_qual' in X_processed.columns and gr_liv_area_col in X_processed.columns:
        X_processed['overall_qual_gr_liv_area_inter'] = X_processed['overall_qual'] * X_processed[gr_liv_area_col]
        print(f"Created interaction term: 'overall_qual_gr_liv_area_inter'.")
    else:
        print("Warning: Could not create interaction term (missing 'overall_qual' or 'gr_liv_area'/'gr_liv_area_log').")

    # 3. Example Combined Floor Area
    total_bsmt_sf_col = 'total_bsmt_sf'
    if 'total_bsmt_sf_log' in X_processed.columns:
        total_bsmt_sf_col = 'total_bsmt_sf_log'

    first_flr_sf_col = '1st_flr_sf'
    if '1st_flr_sf_log' in X_processed.columns:
        first_flr_sf_col = '1st_flr_sf_log'

    if total_bsmt_sf_col in X_processed.columns and first_flr_sf_col in X_processed.columns:
        X_processed['total_flr_sf_combined'] = X_processed[total_bsmt_sf_col] + X_processed[first_flr_sf_col]
        print("Created combined floor area: 'total_flr_sf_combined'.")
    else:
        print("Warning: Could not create 'total_flr_sf_combined' (missing basement or 1st floor SF).")


    #  Handling Specific Missing Values
    none_cols_impute_before_pipeline = [
        'alley', 'bsmt_qual', 'bsmt_cond', 'bsmt_exposure', 'bsmtfin_type_1',
        'bsmtfin_type_2', 'fireplace_qu', 'garage_type', 'garage_finish',
        'garage_qual', 'garage_cond', 'pool_qc', 'fence', 'misc_feature', 'mas_vnr_type'
    ]
    for col in none_cols_impute_before_pipeline:
        if col in X_processed.columns and X_processed[col].isnull().any():
            X_processed[col] = X_processed[col].fillna('None')

    if 'mas_vnr_area' in X_processed.columns and X_processed['mas_vnr_area'].isnull().any():
        X_processed['mas_vnr_area'] = X_processed['mas_vnr_area'].fillna(0)

    if 'garage_yr_blt' in X_processed.columns and X_processed['garage_yr_blt'].isnull().any():
        if 'house_age' in X_processed.columns and 'yr_sold' in X_processed.columns:
            X_processed['garage_yr_blt'] = X_processed['garage_yr_blt'].fillna(current_year - X_processed['house_age'])
        elif 'year_built' in X_processed.columns:
            X_processed['garage_yr_blt'] = X_processed['garage_yr_blt'].fillna(X_processed['year_built'])
        elif 'yr_built' in X_processed.columns:
            X_processed['garage_yr_blt'] = X_processed['garage_yr_blt'].fillna(X_processed['yr_built'])
        else:
            X_processed['garage_yr_blt'] = X_processed['garage_yr_blt'].fillna(X_processed['garage_yr_blt'].median())
        print("Imputed 'garage_yr_blt'.")

    print("\nDEBUG (preprocessor.py): Columns in X_processed after missing value handling:", X_processed.columns.tolist())


    #  Log Transformation  
    numerical_features_current = X_processed.select_dtypes(include=np.number).columns.tolist()

    exclude_from_log_transform = [
        'id', 'overall_qual', 'overall_cond', 'mo_sold', 'yr_sold', 'ms_subclass',
        'pool_area', 'misc_val', 'garage_cars', 'lot_area_bins', 
        'house_age' 
    ]

    skewed_candidates = [col for col in numerical_features_current if col not in exclude_from_log_transform]

    for feature in skewed_candidates:
        if feature in X_processed.columns and X_processed[feature].skew() > 0.75:
            if (X_processed[feature] >= 0).all():
                X_processed[f'{feature}_log'] = np.log1p(X_processed[feature])
                X_processed.drop(feature, axis=1, inplace=True)
                print(f"DEBUG (preprocessor.py): Log-transformed '{feature}' to '{feature}_log' and dropped original.")
            else:
                print(f"DEBUG (preprocessor.py): Skipping log transform for '{feature}' due to non-positive values.")

    print("\nDEBUG (preprocessor.py): Columns in X_processed after ALL feature engineering (pre-ColumnTransformer lists):", X_processed.columns.tolist())


    # Feature Lists for ColumnTransformer
    numerical_features_for_pipeline = X_processed.select_dtypes(include=np.number).columns.tolist()
    categorical_features_for_pipeline = X_processed.select_dtypes(include='object').columns.tolist()

    print("\nDEBUG (preprocessor.py): Final Numerical Features list FOR ColumnTransformer:", numerical_features_for_pipeline)
    print("DEBUG (preprocessor.py): Final Categorical Features list FOR ColumnTransformer:", categorical_features_for_pipeline)

    # Define Ordinal Mappings and Encoders
    ordinal_categories = {
        'lot_shape': ['ir3', 'ir2', 'ir1', 'reg'],
        'utilities': ['sev', 'no_sewr', 'no_pu', 'allpub'],
        'land_slope': ['sev', 'mod', 'gtl'],
        'exter_qual': ['po', 'fa', 'ta', 'gd', 'ex'],
        'exter_cond': ['po', 'fa', 'ta', 'gd', 'ex'],
        'bsmt_qual': ['none', 'po', 'fa', 'ta', 'gd', 'ex'],
        'bsmt_cond': ['none', 'po', 'fa', 'ta', 'gd', 'ex'],
        'bsmt_exposure': ['none', 'no', 'mn', 'av', 'gd'],
        'bsmtfin_type_1': ['none', 'unf', 'lwq', 'rec', 'blq', 'alq', 'glq'],
        'bsmtfin_type_2': ['none', 'unf', 'lwq', 'rec', 'blq', 'alq', 'glq'],
        'heating_qc': ['po', 'fa', 'ta', 'gd', 'ex'],
        'kitchen_qual': ['po', 'fa', 'ta', 'gd', 'ex'],
        'functional': ['sal', 'sev', 'maj2', 'maj1', 'mod', 'min2', 'min1', 'typ'],
        'fireplace_qu': ['none', 'po', 'fa', 'ta', 'gd', 'ex'],
        'garage_finish': ['none', 'unf', 'rfn', 'fin'],
        'garage_qual': ['none', 'po', 'fa', 'ta', 'gd', 'ex'],
        'garage_cond': ['none', 'po', 'fa', 'ta', 'gd', 'ex'],
        'paved_drive': ['n', 'p', 'y'],
        'pool_qc': ['none', 'fa', 'ta', 'gd', 'ex'],
        'fence': ['none', 'mnww', 'gdprv', 'mnprv', 'gdpry'],
        'ms_zoning': ['rh', 'rm', 'c(all)', 'fv', 'rl', 'a_agr', 'i(all)'],
        'street': ['grvl', 'pave'],
        'central_air': ['n', 'y']
    }

    final_ordinal_features = [col for col in ordinal_categories.keys() if col in categorical_features_for_pipeline]
    final_nominal_features = [col for col in categorical_features_for_pipeline if col not in final_ordinal_features]

    ordinal_encoder_categories_list = [ordinal_categories[f] for f in final_ordinal_features]


    # Define Pipelines for ColumnTransformer
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    ordinal_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('ordinal', OrdinalEncoder(categories=ordinal_encoder_categories_list, handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    nominal_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features_for_pipeline),
            ('ord', ordinal_pipeline, final_ordinal_features),
            ('nom', nominal_pipeline, final_nominal_features)
        ],
        remainder='passthrough'
    )
    print("Preprocessor (ColumnTransformer) built with numerical, ordinal, and nominal pipelines.")

    return preprocessor, X_processed


if __name__ == '__main__':
    from data_loader import load_and_initial_clean

    DATA_PATH = '../data/AmesHousing.csv'
    initial_df = load_and_initial_clean(DATA_PATH)

    if initial_df is not None:
        y = initial_df['saleprice']
        X = initial_df.drop('saleprice', axis=1)

        y_log = np.log1p(y)

        preprocessor, X_transformed_for_split = build_preprocessor(X)

        print("\nShape of X after all feature engineering and preprocessing in `build_preprocessor`:", X_transformed_for_split.shape)
        print("Columns in X_transformed_for_split:", X_transformed_for_split.columns.tolist())