import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib
import shap 
import matplotlib.pyplot as plt
import importlib 
import re 

current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = current_script_path

if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Added {project_root} to sys.path for module import.")
try:
    # Import modules as full names first
    import src.data_loader as data_loader_module
    import src.preprocessor as preprocessor_module
    import src.model_trainer as model_trainer_module
    import src.utils as utils_module

    # Force reload them to pick up any changes
    importlib.reload(data_loader_module)
    importlib.reload(preprocessor_module)
    importlib.reload(model_trainer_module)
    importlib.reload(utils_module)

    # Now import specific functions from the reloaded modules
    from src.data_loader import load_and_initial_clean
    from src.preprocessor import build_preprocessor 
    from src.model_trainer import load_model, train_and_evaluate_models, save_model
    from src.utils import plot_feature_importance

    print("Successfully reloaded src modules and imported functions.")
except ImportError as e:
    st.error(f"CRITICAL ERROR: Failed to import or reload src modules: {e}")
    st.error("Please ensure your 'src' folder exists, contains __init__.py, and verify your Streamlit working directory setup.")
    st.stop() # Stop the app if modules can't be loaded


# --- Caching functions to load model and preprocessor only once ---
@st.cache_resource
def load_model_and_template_df():
    st.write("Loading model and preparing data template...")
    DATA_PATH = os.path.join(project_root, 'data', 'AmesHousing.csv')
    MODEL_PATH = os.path.join(project_root, 'models', 'best_house_price_model_xgboost.pkl')

    # Load the trained model
    model = None
    try:
        model = load_model(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Model not found at {MODEL_PATH}. Please ensure you ran 03_Model_Training_and_Evaluation.ipynb to save the model.")
        st.stop() # Stop the app if model is not found
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop() # Stop the app on other loading errors

    # Load initial raw data to build the X_processed_template_df
    df_raw = load_and_initial_clean(DATA_PATH)
    if df_raw is None:
        st.error("Failed to load raw data for preprocessor setup.")
        st.stop()

    X_for_template_build = df_raw.drop('saleprice', axis=1)
    _, X_processed_template = build_preprocessor(X_for_template_build)

    return model, X_processed_template, df_raw # Return the loaded model, template DataFrame, and raw df


# --- Function to apply manual feature engineering steps to a single input row ---
def apply_manual_feature_engineering(input_df_single_row, X_processed_template_df):

    engineered_df = input_df_single_row.copy()

    # 1. Create 'House Age' feature and drop 'year_built'/'yr_built'
    if 'yr_sold' in engineered_df.columns:
        current_year = engineered_df['yr_sold'].iloc[0] # Use year from input row
        if 'year_built' in engineered_df.columns:
            engineered_df['house_age'] = current_year - engineered_df['year_built']
            engineered_df.drop('year_built', axis=1, inplace=True)
        elif 'yr_built' in engineered_df.columns:
            engineered_df['house_age'] = current_year - engineered_df['yr_built']
            engineered_df.drop('yr_built', axis=1, inplace=True)
    
    # 2. Example Interaction Term: Overall Quality * Gr Living Area
    gr_liv_area_val = engineered_df['gr_liv_area'].iloc[0] if 'gr_liv_area' in engineered_df.columns else 0 # Default if missing
    
    # Check if gr_liv_area was log-transformed in the template
    if 'gr_liv_area_log' in X_processed_template_df.columns:
        gr_liv_area_val_for_inter = np.log1p(gr_liv_area_val)
    else:
        gr_liv_area_val_for_inter = gr_liv_area_val

    if 'overall_qual' in engineered_df.columns:
        engineered_df['overall_qual_gr_liv_area_inter'] = engineered_df['overall_qual'] * gr_liv_area_val_for_inter
    else:
        engineered_df['overall_qual_gr_liv_area_inter'] = 0 # Sensible default


    # 3. Example Combined Floor Area
    total_bsmt_sf_val = engineered_df['total_bsmt_sf'].iloc[0] if 'total_bsmt_sf' in engineered_df.columns else 0
    first_flr_sf_val = engineered_df['1st_flr_sf'].iloc[0] if '1st_flr_sf' in engineered_df.columns else 0

    if 'total_bsmt_sf_log' in X_processed_template_df.columns:
        total_bsmt_sf_val = np.log1p(total_bsmt_sf_val)
    if '1st_flr_sf_log' in X_processed_template_df.columns:
        first_flr_sf_val = np.log1p(first_flr_sf_val)

    engineered_df['total_flr_sf_combined'] = total_bsmt_sf_val + first_flr_sf_val


    # 4. Binning (Discretization) - Example for Lot Area
    if 'lot_area' in engineered_df.columns:
        engineered_df.drop('lot_area', axis=1, inplace=True)
    if 'lot_area_log' in engineered_df.columns:
        engineered_df.drop('lot_area_log', axis=1, inplace=True)
    
    if 'lot_area_bins' in X_processed_template_df.columns and 'lot_area_bins' not in engineered_df.columns:
        engineered_df['lot_area_bins'] = X_processed_template_df['lot_area_bins'].median()


    # --- Handling Specific Missing Values (pre-pipeline for 'None' category) ---
    none_cols_impute_before_pipeline = [
        'alley', 'bsmt_qual', 'bsmt_cond', 'bsmt_exposure', 'bsmtfin_type_1',
        'bsmtfin_type_2', 'fireplace_qu', 'garage_type', 'garage_finish',
        'garage_qual', 'garage_cond', 'pool_qc', 'fence', 'misc_feature', 'mas_vnr_type'
    ]
    for col in none_cols_impute_before_pipeline:
        if col in engineered_df.columns and engineered_df[col].isnull().any():
            engineered_df[col] = engineered_df[col].fillna('None')
        elif col in engineered_df.columns and engineered_df[col].dtype == 'object': # If not null but still object, ensure 'None' is a category
            if 'None' not in engineered_df[col].unique():
                # This is a bit tricky, but ensures 'None' is a known category if it's not.
                # For Streamlit inputs, this typically isn't an issue if the selectbox provides 'None'.
                pass # Rely on preprocessor's handle_unknown='ignore' for unseen categories.


    if 'mas_vnr_area' in engineered_df.columns and engineered_df['mas_vnr_area'].isnull().any():
        engineered_df['mas_vnr_area'] = engineered_df['mas_vnr_area'].fillna(0)

    if 'garage_yr_blt' in engineered_df.columns and engineered_df['garage_yr_blt'].isnull().any():
        if 'house_age' in engineered_df.columns and 'yr_sold' in engineered_df.columns:
            current_year_input = engineered_df['yr_sold'].iloc[0]
            engineered_df['garage_yr_blt'] = engineered_df['garage_yr_blt'].fillna(current_year_input - engineered_df['house_age'].iloc[0])
        else:
            engineered_df['garage_yr_blt'] = engineered_df['garage_yr_blt'].fillna(X_processed_template_df['garage_yr_blt'].median())


    # --- Log Transformation for Skewed Numerical Features ---
    log_transformed_cols_in_template = [col for col in X_processed_template_df.columns if col.endswith('_log')]
    original_cols_that_were_log_transformed = [col.replace('_log', '') for col in log_transformed_cols_in_template]

    for original_col in original_cols_that_were_log_transformed:
        if original_col in engineered_df.columns:
            if (engineered_df[original_col] >= 0).all():
                engineered_df[f'{original_col}_log'] = np.log1p(engineered_df[original_col])
                engineered_df.drop(original_col, axis=1, inplace=True)
            # else: print warning if needed

    missing_cols = set(X_processed_template_df.columns) - set(engineered_df.columns)
    for c in missing_cols:
        engineered_df[c] = np.nan # Add missing columns with NaN

    # Ensure column order matches X_processed_template_df
    engineered_df = engineered_df[X_processed_template_df.columns]

    return engineered_df


# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="Ames House Price Predictor", layout="wide")
    st.title("Ames House Price Predictor")
    st.markdown("Enter the characteristics of a house in Ames, Iowa, to get an estimated sale price.")

    # Load the model and the template DataFrame
    model_pipeline, X_processed_template_df, df_raw_data_for_ui = load_model_and_template_df()

    # Get the fitted preprocessor from the loaded model pipeline
    fitted_preprocessor_ct = model_pipeline.named_steps['preprocessor']


    st.sidebar.header("House Features Input")

    # Create a dictionary to hold all feature values, starting with defaults from the raw data
    # This ensures we have all original columns from the raw data for input
    initial_raw_input_defaults = df_raw_data_for_ui.drop('saleprice', axis=1).iloc[[0]].copy()
    input_data_for_ui = initial_raw_input_defaults.iloc[0].to_dict() # Convert to dict for easier manipulation

    # --- UI Input Widgets ---
    with st.sidebar.form("house_features_form"):
        st.subheader("Key Features")

        # Numerical Inputs (using original names for UI)
        # Use .get() with a sensible fallback if a column somehow isn't in the defaults (e.g., if a feature was dropped early)
        input_data_for_ui['gr_liv_area'] = st.number_input("Above Grade Living Area (sq ft)", min_value=334, max_value=5642, value=int(input_data_for_ui.get('gr_liv_area', 1500)))
        input_data_for_ui['overall_qual'] = st.slider("Overall Quality (1-10)", min_value=1, max_value=10, value=int(input_data_for_ui.get('overall_qual', 5)))
        input_data_for_ui['total_bsmt_sf'] = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=6110, value=int(input_data_for_ui.get('total_bsmt_sf', 1000)))
        input_data_for_ui['garage_cars'] = st.slider("Garage Car Capacity", min_value=0, max_value=4, value=int(input_data_for_ui.get('garage_cars', 2)))
        input_data_for_ui['full_bath'] = st.slider("Full Bathrooms (Above Grade)", min_value=0, max_value=4, value=int(input_data_for_ui.get('full_bath', 2)))
        input_data_for_ui['mas_vnr_area'] = st.number_input("Masonry Veneer Area (sq ft)", min_value=0, max_value=1600, value=int(input_data_for_ui.get('mas_vnr_area', 0)))
        input_data_for_ui['year_remod_add'] = st.number_input("Year Remodeled/Added", min_value=1950, max_value=2010, value=int(input_data_for_ui.get('year_remod_add', 2000)))
        input_data_for_ui['fireplaces'] = st.slider("Number of Fireplaces", min_value=0, max_value=4, value=int(input_data_for_ui.get('fireplaces', 0)))
        input_data_for_ui['mo_sold'] = st.slider("Month Sold", min_value=1, max_value=12, value=int(input_data_for_ui.get('mo_sold', 6)))
        input_data_for_ui['yr_sold'] = st.slider("Year Sold", min_value=2006, max_value=2010, value=int(input_data_for_ui.get('yr_sold', 2008)))
        input_data_for_ui['lot_area'] = st.number_input("Lot Area (sq ft)", min_value=1000, max_value=215245, value=int(input_data_for_ui.get('lot_area', 10000)))
        input_data_for_ui['year_built'] = st.number_input("Year Built", min_value=1872, max_value=2010, value=int(input_data_for_ui.get('year_built', 1980)))


        # Categorical Inputs (using original names for UI)
        # Use df_raw_data_for_ui for unique values as it represents the raw data
        unique_neighborhoods = df_raw_data_for_ui['neighborhood'].unique().tolist() if 'neighborhood' in df_raw_data_for_ui.columns else []
        input_data_for_ui['neighborhood'] = st.selectbox("Neighborhood", unique_neighborhoods, index=unique_neighborhoods.index(df_raw_data_for_ui['neighborhood'].mode()[0]) if unique_neighborhoods else 0)

        unique_house_styles = df_raw_data_for_ui['house_style'].unique().tolist() if 'house_style' in df_raw_data_for_ui.columns else []
        input_data_for_ui['house_style'] = st.selectbox("House Style", unique_house_styles, index=unique_house_styles.index(df_raw_data_for_ui['house_style'].mode()[0]) if unique_house_styles else 0)

        unique_roof_styles = df_raw_data_for_ui['roof_style'].unique().tolist() if 'roof_style' in df_raw_data_for_ui.columns else []
        input_data_for_ui['roof_style'] = st.selectbox("Roof Style", unique_roof_styles, index=unique_roof_styles.index(df_raw_data_for_ui['roof_style'].mode()[0]) if unique_roof_styles else 0)

        unique_exter_quals = df_raw_data_for_ui['exter_qual'].unique().tolist() if 'exter_qual' in df_raw_data_for_ui.columns else []
        input_data_for_ui['exter_qual'] = st.selectbox("Exterior Quality", unique_exter_quals, index=unique_exter_quals.index(df_raw_data_for_ui['exter_qual'].mode()[0]) if unique_exter_quals else 0)

        unique_kitchen_quals = df_raw_data_for_ui['kitchen_qual'].unique().tolist() if 'kitchen_qual' in df_raw_data_for_ui.columns else []
        input_data_for_ui['kitchen_qual'] = st.selectbox("Kitchen Quality", unique_kitchen_quals, index=unique_kitchen_quals.index(df_raw_data_for_ui['kitchen_qual'].mode()[0]) if unique_kitchen_quals else 0)

        submitted = st.form_submit_button("Predict House Price")


    if submitted:
        # Start with a copy of a default row from the raw data format 
        single_input_df_raw_format = df_raw_data_for_ui.drop('saleprice', axis=1).iloc[[0]].copy()

        # Update this raw-format DataFrame with user inputs
        for feature, value in input_data_for_ui.items():
            if feature in single_input_df_raw_format.columns:
                single_input_df_raw_format[feature] = value
            # else: st.warning(f"Input feature '{feature}' not found in raw data columns. Skipping.")


        # Apply manual feature engineering steps to the single input row
        # This function must replicate the steps in build_preprocessor *before* the ColumnTransformer.
        engineered_input_df = apply_manual_feature_engineering(single_input_df_raw_format, X_processed_template_df)

        # Make prediction using the model pipeline which includes the fitted_preprocessor_ct
        try:
            log_prediction = model_pipeline.predict(engineered_input_df)[0]
            predicted_price = np.expm1(log_prediction) # Inverse transform from log scale
            st.success(f"Predicted House Price: **${predicted_price:,.2f}**")

            # --- SHAP Explanation for the Prediction ---
            st.subheader("Why this prediction? (SHAP Explanation)")
            st.info("The SHAP force plot shows how each feature contributes to pushing the prediction from the base value (average prediction) to your predicted value.")

            # Get the transformed input for SHAP using the fitted_preprocessor_ct
            transformed_input_for_shap = fitted_preprocessor_ct.transform(engineered_input_df)

            # Get feature names after preprocessing
            try:
                feature_names_out = fitted_preprocessor_ct.get_feature_names_out()
            except AttributeError:
                st.warning("Could not get feature names from preprocessor. SHAP plot might lack detailed labels.")
                # Fallback feature names if get_feature_names_out is not available
                feature_names_out = [f"feature_{i}" for i in range(transformed_input_for_shap.shape[1])]

            transformed_input_df_for_shap = pd.DataFrame(transformed_input_for_shap, columns=feature_names_out)

            # Create SHAP explainer
            explainer = shap.TreeExplainer(model_pipeline.named_steps['regressor'])
            shap_values = explainer.shap_values(transformed_input_df_for_shap)

            # Display SHAP force plot
            #st.set_option('deprecation.showPyplotGlobalUse', False) # Suppress warning for plt.show()
            #shap.initjs()

            fig_force = shap.force_plot(explainer.expected_value, shap_values[0,:], transformed_input_df_for_shap.iloc[0,:], matplotlib=True, show=False)
            st.pyplot(fig_force, bbox_inches='tight')

        except Exception as e:
            st.error(f"An error occurred during prediction or explanation: {e}")
            st.exception(e)


if __name__ == "__main__":
    main()
