# Health Insurance Premium Prediction

## Overview

This project predicts health insurance premiums using Machine Learning based on demographic, lifestyle, and medical attributes. The focus of the project is improving model accuracy through error analysis, data segmentation, and feature engineering.

---

## Problem Statement

A single regression model showed inconsistent performance across age groups. Higher prediction errors were observed for individuals aged 25 and below, indicating different premium behavior compared to older individuals.

---

## Approach

### Exploratory Data Analysis (EDA)

EDA was performed using Seaborn and Matplotlib to understand:

* Feature distributions
* Age-wise premium trends
* Correlation between variables
* Presence of outliers

Insights from EDA highlighted age as a major factor influencing prediction error.

---

### Data Preprocessing

The following preprocessing steps were applied:

* Handling missing values
* Encoding categorical variables
* Feature scaling
* Train-test split

---

### Error Analysis

After training a baseline regression model, residual errors were analyzed across age groups. The analysis showed significantly higher errors for the younger age group.

---

### Data Segmentation

Based on error analysis, the dataset was split into:

* Age ≤ 25
* Age > 25

Separate models were trained for each group.

---

### Feature Engineering

A Genetic Risk feature was introduced to capture hereditary health impact and improve model learning.

---

## Model Details

* Problem Type: Regression
* Algorithm: XGBoost Regressor
* Separate models trained for each age group

---

## Prediction Logic

1. Preprocess input data
2. Check user age
3. Select the appropriate model
4. Predict insurance premium

---

## Tools & Libraries

* Python
* Pandas, NumPy
* Seaborn, Matplotlib
* Scikit-learn
* XGBoost
---

## Project Structure

```
artifacts/
main.py
prediction_helper.py
requirements.txt
README.md
```

---

## Key Learnings

* Error analysis can guide better model design
* Data segmentation improves prediction accuracy
* Feature engineering adds significant value
* XGBoost performs well for non-linear regression problems

---

## Future Work

* Add more medical parameters
* Integrate model explainability
* Extend to long-term premium forecasting

---

## Conclusion

This project demonstrates a structured Machine Learning workflow with a focus on improving model performance using data-driven insights.



