# Telco Customer Churn Analysis & Prediction Dashboard 📊

This repository contains the code for an interactive web application built with Streamlit to analyze customer churn for a fictional Telco company. The app provides insights into churn drivers, predicts individual customer churn risk using machine learning models, and visualizes model behavior.

## ✨ Features

*   **Interactive Dashboard:**
    *   Displays Key Performance Indicators (KPIs) like total customers, churn rate, average tenure, and average monthly charges.
    *   Visualizes overall churn distribution (Pie Chart).
    *   Presents interactive charts showing churn rates based on key factors (Contract Type, Internet Service, Payment Method, Tenure Groups).
*   **Churn Prediction Tool:**
    *   Allows users to input individual customer characteristics through an intuitive form.
    *   Uses a pre-trained XGBoost model to predict the probability of churn for the entered customer profile.
    *   Displays the prediction result (High/Low Risk) and the churn probability percentage.
    *   Provides illustrative key factors influencing the prediction.
    *   Suggests tailored retention or engagement actions based on the prediction.
*   **Detailed Churn Analysis:**
    *   **Key Drivers:** Compares churn rates across different service types, contract types, payment methods, and add-on services.
    *   **Demographics & Tenure:** Analyzes churn patterns based on gender, senior citizen status, partners, dependents, and tenure distribution. Includes scatter plots (e.g., Monthly Charges vs. Tenure).
    *   **Contract & Financials:** Explores the impact of monthly charges, paperless billing, and includes a **Contract Transition Simulation** to estimate the impact of converting Month-to-Month customers.
    *   **Segmentation:** Performs K-Means clustering on numerical features (Tenure, Charges) to identify distinct customer segments and analyzes their profiles and churn rates.
*   **Decision Tree Visualization:**
    *   Displays an interactive plot of the trained Decision Tree model, allowing users to explore the split logic.
    *   Shows Feature Importance scores derived from the Decision Tree model.

## 📸 Screenshots

*(**Important:** Replace these placeholders with actual screenshots of your running application!)*

**1. Dashboard Overview:**
![Dashboard Screenshot](placeholder_dashboard.png) _<-- Replace with link or image embed_

**2. Prediction Page (Example):**
![Prediction Screenshot](placeholder_prediction.png) _<-- Replace with link or image embed_

**3. Analysis Tab (e.g., Segmentation):**
![Analysis Screenshot](placeholder_analysis.png) _<-- Replace with link or image embed_

**4. Decision Tree Visualization:**
![Decision Tree Screenshot](placeholder_dt.png) _<-- Replace with link or image embed_

*(Optional: Consider adding a short GIF showcasing the interactivity.)*

## 💻 Technologies Used

*   **Python:** Core programming language.
*   **Streamlit:** Framework for building the interactive web application.
*   **Pandas:** Data manipulation and analysis.
*   **NumPy:** Numerical operations.
*   **Scikit-learn:**
    *   `DecisionTreeClassifier` for model training and visualization.
    *   `KMeans` for clustering.
    *   `StandardScaler` for data scaling.
    *   Metrics functions (implicitly used via model reports, potentially).
*   **XGBoost:** For the churn prediction model.
*   **Plotly & Plotly Express:** Creating interactive charts and visualizations.
*   **Matplotlib:** Used by `sklearn.tree.plot_tree` for rendering the decision tree.
*   **Joblib:** Loading the pre-trained machine learning models.

## ⚙️ Setup and Running Locally

Follow these steps to run the application on your local machine:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git # <-- Replace with your repo URL
    cd your-repo-name
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    # Using venv
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

    # Or using Conda
    # conda create -n churn_env python=3.9  # Adjust Python version if needed
    # conda activate churn_env
    ```

3.  **Install Dependencies:**
    Make sure you have a `requirements.txt` file in your repository (create one if you don't).
    ```bash
    pip install -r requirements.txt
    ```
    *(See `requirements.txt` section below if you need to create it)*

4.  **Place Data Files:**
    Ensure the following data files are present in the **root directory** of the cloned repository:
    *   `churn_raw_data.csv`: Raw customer data used for analysis and dashboard.
    *   `churn_df_su.csv`: Preprocessed data, potentially used for inferring feature names or other tasks.

5.  **Place Model Files:**
    Ensure the following pre-trained model files are present in the **root directory**:
    *   `xgboost_model.joblib`: The trained XGBoost classifier.
    *   `decision_tree_model.joblib`: The trained Decision Tree classifier.

6.  **Run the Streamlit App:**
    ```bash
    streamlit run churn_app.py
    ```

7.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## `requirements.txt`

If you don't have a `requirements.txt` file, create one in the root directory with the following content (adjust versions based on your environment if necessary, but these are common compatible versions):

```text
streamlit>=1.20.0
pandas>=1.4.0
numpy>=1.21.0
scikit-learn>=1.1.0
xgboost>=1.5.0
plotly>=5.5.0
matplotlib>=3.5.0
joblib>=1.1.0
# Pillow might be needed if you use st.image with local files
Pillow>=9.0.0
```

📁 Project Structure
churn_predictor_dashboard/
│
├── .venv/                     # Virtual environment directory (optional, if created)
├── churn_app.py               # Main Streamlit application script
├── churn_raw_data.csv         # Raw dataset
├── churn_df_su.csv            # Processed dataset
├── xgboost_model.joblib       # Trained XGBoost model
├── decision_tree_model.joblib # Trained Decision Tree model
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── (Optional: placeholder_*.png) # Placeholder images for README
└── (Optional: logo.png)       # Logo file used in sidebar

📄 License
(Optional: Choose a license, e.g., MIT)
This project is licensed under the MIT License - see the LICENSE file for details (if you add one).
