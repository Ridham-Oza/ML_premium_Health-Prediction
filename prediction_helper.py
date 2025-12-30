import pandas as pd
import joblib

# =========================
# Load artifacts
# =========================
model_young = joblib.load("artifacts/model_young.joblib")
model_rest = joblib.load("artifacts/model_rest.joblib")

scaler_young = joblib.load("artifacts/scaler_young.joblib")
scaler_rest = joblib.load("artifacts/scaler_rest.joblib")


# =========================
# Medical risk normalization
# =========================
def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6, "heart disease": 8, "high blood pressure": 6,
        "thyroid": 5, "no disease": 0, "none": 0
    }
    diseases = medical_history.lower().split(" & ")
    total_score = sum(risk_scores.get(d, 0) for d in diseases)
    return total_score / 14


def preprocess_input(input_dict):
    # These must match your model's training features
    expected_columns = [
        'Age', 'Number_Of_Dependants', 'Income_Lakhs',
        'Insurance_Plan', 'Genetical_Risk', 'normalized_risk_score',
        'Gender_Male', 'Region_Northwest', 'Region_Southeast', 'Region_Southwest',
        'Marital_status_Unmarried', 'BMI_Category_Obesity',
        'BMI_Category_Overweight', 'BMI_Category_Underweight',
        'Smoking_Status_Occasional', 'Smoking_Status_Regular',
        'Employment_Status_Salaried', 'Employment_Status_Self-Employed'
    ]

    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    # Assign values from input_dict
    df['Age'] = input_dict['Age']
    df['Number_Of_Dependants'] = input_dict['Number of Dependants']
    df['Income_Lakhs'] = input_dict['Income in Lakhs']
    df['Genetical_Risk'] = input_dict['Genetical Risk']

    plan_map = {'Bronze': 1, 'Silver': 2, 'Gold': 3}
    df['Insurance_Plan'] = plan_map.get(input_dict['Insurance Plan'], 1)

    if input_dict['Gender'] == 'Male': df['Gender_Male'] = 1

    region = input_dict['Region']
    if f'Region_{region}' in df.columns:
        df[f'Region_{region}'] = 1

    if input_dict['Marital Status'] == 'Unmarried':
        df['Marital_status_Unmarried'] = 1

    bmi = input_dict['BMI Category']
    if f'BMI_Category_{bmi}' in df.columns:
        df[f'BMI_Category_{bmi}'] = 1

    smoking = input_dict['Smoking Status']
    if f'Smoking_Status_{smoking}' in df.columns:
        df[f'Smoking_Status_{smoking}'] = 1

    emp = input_dict['Employment Status']
    if f'Employment_Status_{emp}' in df.columns:
        df[f'Employment_Status_{emp}'] = 1

    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])

    return df


def apply_scaling(df):
    age = df['Age'].iloc[0]

    # Select the correct scaler bundle based on age
    scaler_bundle = scaler_young if age <= 25 else scaler_rest

    scaler = scaler_bundle['scaler']
    cols_to_scale = scaler_bundle['cols_to_scale']

    # CRITICAL FIX: The scaler expects 'Income_Level' (capital L)
    # based on your error message. We create it temporarily.
    if 'Income_Level' not in df.columns:
        df['Income_Level'] = 0

        # Perform transformation
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Drop the dummy column so it doesn't interfere with the model features
    if 'Income_Level' in df.columns and 'Income_Level' not in model_rest.feature_names_in_:
        df.drop('Income_Level', axis=1, inplace=True)

    return df


def predict(input_dict):
    # 1. Convert dict to DataFrame
    df = preprocess_input(input_dict)

    # 2. Scale the numerical columns (Fixes the KeyError)
    df = apply_scaling(df)

    # 3. Select model
    model = model_young if input_dict['Age'] <= 25 else model_rest

    # 4. Ensure column order matches exactly what the model saw during training
    df = df[model.feature_names_in_]

    # 5. Predict
    prediction = model.predict(df)
    return int(prediction[0])