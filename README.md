🏥 Health Insurance Premium Prediction AI
Project Overview
This project is an end-to-end Machine Learning application designed to predict health insurance premiums based on individual health and demographic profiles. By leveraging advanced regression techniques, the model provides instant, data-driven cost estimates through a modern, user-friendly web interface.

🚀 Key Features
🧠 Error-Driven Model Architecture (Intelligent Routing)
During initial evaluation, a significant residual error was identified in the 18–25 age demographic. To solve this, I performed an Error Analysis and implemented a two-fold strategy:

Data Stratification: Split the dataset into two specialized subsets (Young vs. Senior) to capture distinct risk patterns.

Feature Engineering: Introduced a 'Genetic Risk' feature to provide the model with deeper context, successfully decreasing the error rate and boosting overall accuracy.

Dynamic Routing: The application logic automatically routes user data to the specific model optimized for their age bracket.

💻 Technical Highlights
High-Performance Regressors: Built using XGBoost, a powerful gradient boosting library known for its efficiency and predictive power.

Modern UI/UX: A clean, dashboard-style interface developed with Streamlit, featuring custom CSS for a professional "Health-Tech" look and feel.

Data Pipeline: Includes robust preprocessing steps like feature scaling and categorical encoding to handle real-world medical data.

🛠️ Technology Stack
Language: Python

Modeling: XGBoost, Scikit-Learn

Web Framework: Streamlit

Version Control: Git & GitHub

Deployment: Streamlit Cloud

📂 Project Structure
Plaintext

├── artifacts/               # Trained models (.joblib) and scalers
├── main.py                  # Streamlit frontend and application logic
├── prediction_helper.py     # Data processing and model loading logic
├── requirements.txt         # List of dependencies for deployment
├── .gitignore               # Files excluded from the repository
└── README.md                # Project documentation
⚙️ How It Works
Input: User enters details (Age, BMI, Smoking Status, Income, etc.) via the dashboard.

Processing: Data is cleaned and normalized using specialized scalers.

Model Selection: Based on user age, the app dynamically loads the model_young or model_rest.

Output: The model calculates the estimated premium and displays it instantly.

🚀 Future Scope & Enhancements
Expanded Medical Parameters: Integrate data points like Blood Pressure and Cholesterol for deeper risk profiling.

Explainable AI (XAI): Add SHAP value integration to show users exactly why their premium is a certain price.

Time-Series Tracking: Predict how premiums might change over 5–10 years based on aging and lifestyle trends.

Automated Retraining: Set up a pipeline to automatically update models as new insurance data becomes available.
