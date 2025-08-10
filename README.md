House Price Prediction Project

This repository contains an end-to-end machine learning project aimed at predicting house prices in Ames, Iowa, using the Ames Housing Dataset. It covers the complete data science workflow from exploratory data analysis (EDA) and feature engineering to model training, evaluation, explainability, and deployment in an interactive dashboard.

Project Goal
The goal of this project is to create a robust regression model capable of accurately predicting house sale prices based on various property features. Alongside model development, the project focuses on building a clear and modular Streamlit dashboard for easy interaction with the trained model.

Project Structure
House_Price_Prediction/
├── data/ # Contains the raw dataset
│ └── AmesHousing.csv
├── notebooks/ # Jupyter notebooks for each stage of development
│ ├── 01_EDA.ipynb
│ ├── 02_Feature_Engineering.ipynb
│ ├── 03_Model_Training_and_Evaluation.ipynb
│ └── 04_Model_Explainability.ipynb
├── src/ # Modular Python scripts for reusable code
│ ├── init.py
│ ├── data_loader.py
│ ├── preprocessor.py
│ ├── model_trainer.py
│ └── utils.py
├── models/ # Saved trained models
│ └── best_house_price_model_xgboost.pkl
├── dashboard/ # Streamlit dashboard files
│ ├── dashboard_app.py
│ └── dashboard_utils.py
├── .gitignore # Ignored files for Git
├── README.md # Project overview and instructions
├── requirements.txt # Python dependencies

Getting Started

Prerequisites:

Python 3.8 or higher

pip (Python package installer)

git (for cloning the repository)

Setup Instructions:

Clone this repository:
git clone <repository-url>
cd House_Price_Prediction

Create and activate a virtual environment:
python -m venv venv
On Windows: .\venv\Scripts\activate
On macOS/Linux: source venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Download the AmesHousing.csv dataset from Kaggle and place it in the data/ directory.

Running the Project

Execute the notebooks in sequence (01_EDA.ipynb to 04_Model_Explainability.ipynb) to understand the workflow. Notebook 03_Model_Training_and_Evaluation.ipynb trains the final model and saves it as best_house_price_model_xgboost.pkl.

Launch the dashboard:
cd dashboard
streamlit run dashboard_app.py
This opens the interactive dashboard in your browser, allowing you to test predictions and view performance metrics.

Project Summary

Tasks Completed:

Performed advanced EDA including outlier detection, correlation analysis, and feature distribution visualization.

Engineered features using binning, interaction terms, and log transformations for skewed data.

Trained multiple regression models (Linear Regression, Ridge, Lasso, Random Forest, XGBoost) and selected XGBoost as the best performer.

Evaluated model performance using R², MAE, and RMSE.

Applied SHAP for model explainability, providing global and local feature importance insights.

Built a modular Streamlit dashboard for model testing and evaluation.

Model Performance:
XGBoost achieved strong performance on the test set:

R² Score: 0.9408

MAE: ~$12,000

RMSE: ~$19,000

These results indicate that the model explains over 94% of the variance in house prices and maintains a low average prediction error, making it a reliable tool for estimating property values.
