# Heart Attack Analysis Project

## Author Information
- Name: 2024mt12084@wilp.bits-pilani.ac.in
- BITS ID: 2024mt12084@wilp.bits-pilani.ac.in

## Project Overview
This project performs a comprehensive analysis of heart attack data to predict the likelihood of heart attacks based on various health parameters. The analysis includes data preprocessing, exploratory data analysis, feature selection, and predictive modeling.

## Dataset Description
The dataset `Heart_Attack_Analysis_Data.csv` contains information about patients and their health parameters:
- Age: Patient's age
- Sex: Gender (1=Male, 0=Female)
- CP_Type: Chest Pain Type (1=typical angina, 2=atypical angina, 3=non-anginal pain, 4=asymptomatic)
- BloodPressure: Blood pressure measurement
- Cholestrol: Cholesterol level
- BloodSugar: Blood sugar level
- ECG: ECG results
- MaxHeartRate: Maximum heart rate achieved
- ExerciseAngia: Exercise-induced angina (1=Yes, 0=No)
- FamilyHistory: Number of family members with heart disease
- Target: Heart attack risk (0=Less chance, 1=More chance)

## Project Structure
1. `heart_attack_analysis.py`: Main Python script containing the analysis code
2. `requirements.txt`: List of required Python packages
3. Generated visualization files:
   - `exploratory_analysis.png`: Visualizations of key features
   - `correlation_matrix.png`: Correlation analysis heatmap
   - `feature_importance.png`: Feature importance plot

## Implementation Details

### 1. Data Preprocessing
- Loading and cleaning the dataset
- Handling missing values
- Feature scaling using StandardScaler

### 2. Exploratory Data Analysis
- Statistical analysis of features
- Visualization of key distributions
- Correlation analysis between features
- Generation of informative plots

### 3. Feature Selection
- Implementation of SelectKBest with f_classif
- Analysis of feature importance scores
- Identification of most relevant features

### 4. Model Building and Evaluation
- Implementation of Random Forest Classifier
- Train-test split (80-20)
- Cross-validation
- Model evaluation metrics:
  - Accuracy
  - Classification Report
  - Confusion Matrix
  - Feature Importance

## How to Run
1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the analysis script:
   ```bash
   python heart_attack_analysis.py
   ```

## Results and Findings
The analysis provides:
1. Key factors influencing heart attack risk
2. Feature importance rankings
3. Model performance metrics
4. Visual representations of relationships between variables

## Assumptions
1. The data is representative of the general population
2. All measurements are accurate and reliable
3. The features provided are sufficient for prediction
4. The relationship between features and target is consistent

## Future Improvements
1. Try different machine learning algorithms
2. Implement hyperparameter tuning
3. Add more advanced feature engineering
4. Include additional data sources
5. Implement model deployment pipeline

## Conclusion
This project successfully implements a comprehensive analysis of heart attack risk factors and builds a predictive model. The results provide valuable insights into the key factors influencing heart attack risk and demonstrate the effectiveness of the chosen approach. 