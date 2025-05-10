"""
Heart Attack Analysis
Author: 2024mt12084@wilp.bits-pilani.ac.in
BITS ID: 2024mt12084@wilp.bits-pilani.ac.in

This script performs comprehensive analysis of heart attack data to predict the likelihood of heart attacks
based on various health parameters.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the heart attack dataset.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Display basic information
    print("\nDataset Information:")
    print(df.info())
    print("\nFirst few rows of the dataset:")
    print(df.head())
    
    # Check for missing values
    print("\nMissing values in each column:")
    print(df.isnull().sum())
    
    # Separate features and target
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    return X, y

def perform_exploratory_analysis(X, y):
    """
    Perform exploratory data analysis on the dataset.
    
    Args:
        X (DataFrame): Feature matrix
        y (Series): Target vector
    """
    # Create a copy of the data with target
    df = X.copy()
    df['Target'] = y
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # 1. Age distribution by target
    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x='Age', hue='Target', multiple="stack")
    plt.title('Age Distribution by Heart Attack Risk')
    
    # 2. Gender distribution by target
    plt.subplot(2, 2, 2)
    sns.countplot(data=df, x='Sex', hue='Target')
    plt.title('Gender Distribution by Heart Attack Risk')
    
    # 3. Chest Pain Type distribution
    plt.subplot(2, 2, 3)
    sns.countplot(data=df, x='CP_Type', hue='Target')
    plt.title('Chest Pain Type Distribution by Heart Attack Risk')
    
    # 4. Blood Pressure distribution
    plt.subplot(2, 2, 4)
    sns.boxplot(data=df, x='Target', y='BloodPressure')
    plt.title('Blood Pressure Distribution by Heart Attack Risk')
    
    plt.tight_layout()
    plt.savefig('exploratory_analysis.png')
    plt.close()
    
    # Correlation analysis
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()

def feature_selection(X, y):
    """
    Perform feature selection to identify the most important features.
    
    Args:
        X (DataFrame): Feature matrix
        y (Series): Target vector
        
    Returns:
        DataFrame: Selected features
    """
    # Use SelectKBest with f_classif
    selector = SelectKBest(f_classif, k='all')
    selector.fit(X, y)
    
    # Get feature scores
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_
    })
    feature_scores = feature_scores.sort_values('Score', ascending=False)
    
    print("\nFeature Importance Scores:")
    print(feature_scores)
    
    return X

def build_and_evaluate_model(X, y):
    """
    Build and evaluate the predictive model.
    
    Args:
        X (DataFrame): Feature matrix
        y (Series): Target vector
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model
    print("\nModel Evaluation:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print("\nCross-validation scores:", cv_scores)
    print("Average CV score:", cv_scores.mean())
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    print("\nFeature Importance from Random Forest:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='Importance', y='Feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def main():
    """
    Main function to execute the heart attack analysis.
    """
    print("Starting Heart Attack Analysis...")
    
    # Load and preprocess data
    X, y = load_and_preprocess_data('Heart_Attack_Analysis_Data.csv')
    
    # Perform exploratory analysis
    perform_exploratory_analysis(X, y)
    
    # Perform feature selection
    X_selected = feature_selection(X, y)
    
    # Build and evaluate model
    build_and_evaluate_model(X_selected, y)
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main() 