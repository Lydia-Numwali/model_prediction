import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import joblib
import warnings

warnings.filterwarnings("ignore")


def load_data(file_path):
    """
    Load the dataset from CSV file
    """
    try:
        df = pd.read_csv("./student_depression_dataset.csv")
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def clean_data(df):
    """
    Clean the dataset by handling missing values and outliers
    """
    # Create a copy of the dataframe
    df_cleaned = df.copy()

    # Handle missing values
    numeric_imputer = SimpleImputer(strategy="mean")
    categorical_imputer = SimpleImputer(strategy="most_frequent")

    numeric_columns = df_cleaned.select_dtypes(include=["int64", "float64"]).columns
    categorical_columns = df_cleaned.select_dtypes(include=["object"]).columns

    if len(numeric_columns) > 0:
        df_cleaned[numeric_columns] = numeric_imputer.fit_transform(
            df_cleaned[numeric_columns]
        )
    if len(categorical_columns) > 0:
        df_cleaned[categorical_columns] = categorical_imputer.fit_transform(
            df_cleaned[categorical_columns]
        )

    # Handle outliers using IQR method
    for column in numeric_columns:
        Q1 = df_cleaned[column].quantile(0.25)
        Q3 = df_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned[column] = df_cleaned[column].clip(lower_bound, upper_bound)

    return df_cleaned


def preprocess_data(df_cleaned):
    """
    Preprocess the data by encoding categorical variables and scaling features
    """
    # Create a copy of the dataframe
    df_processed = df_cleaned.copy()

    # Encode categorical variables
    le = LabelEncoder()
    categorical_columns = df_processed.select_dtypes(include=["object"]).columns
    for column in categorical_columns:
        df_processed[column] = le.fit_transform(df_processed[column])

    # Scale features
    scaler = StandardScaler()
    numeric_columns = df_processed.select_dtypes(include=["int64", "float64"]).columns
    df_processed[numeric_columns] = scaler.fit_transform(df_processed[numeric_columns])

    return df_processed, scaler


def train_model(X_train, y_train):
    """
    Train the model for multiple outputs
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, feature_names):
    """
    Evaluate the model performance for each target variable
    """
    y_pred = model.predict(X_test)

    print("\nModel Performance Metrics for Each Target Variable:")
    print("-" * 50)

    for i, feature in enumerate(feature_names):
        mse = mean_squared_error(y_test[feature], y_pred[:, i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test[feature], y_pred[:, i])
        r2 = r2_score(y_test[feature], y_pred[:, i])

        print(f"\n{feature}:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")

    return y_pred


def plot_results(y_test, y_pred, feature_names):
    """
    Plot actual vs predicted values for each target variable
    """
    n_features = len(feature_names)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    plt.figure(figsize=(5 * n_cols, 5 * n_rows))

    for i, feature in enumerate(feature_names):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.scatter(y_test[feature], y_pred[:, i], alpha=0.5)
        plt.plot(
            [y_test[feature].min(), y_test[feature].max()],
            [y_test[feature].min(), y_test[feature].max()],
            "r--",
            lw=2,
        )
        plt.xlabel(f"Actual {feature}")
        plt.ylabel(f"Predicted {feature}")
        plt.title(f"{feature} - Actual vs Predicted")

    plt.tight_layout()
    plt.savefig("prediction_results.png")
    plt.close()


def main():
    # Load your dataset
    print("Please provide the path to your dataset CSV file:")
    file_path = input().strip()

    # Load data
    df = load_data(file_path)
    if df is None:
        return

    # Clean data
    print("\nCleaning data...")
    df_cleaned = clean_data(df)

    # Preprocess data
    print("Preprocessing data...")
    df_processed, scaler = preprocess_data(df_cleaned)

    # Define target columns (mental health indicators)
    target_columns = ['Depression Level', 'Anxiety Level', 'Stress Level']
    
    # Define feature columns (excluding ID and target columns)
    feature_columns = [col for col in df_processed.columns 
                      if col not in target_columns + ['ID']]
    
    # Split features and targets
    X = df_processed[feature_columns]
    y = df_processed[target_columns]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    print("\nTraining model...")
    model = train_model(X_train, y_train)

    # Evaluate model
    print("\nEvaluating model...")
    y_pred = evaluate_model(model, X_test, y_test, target_columns)

    # Plot results
    print("\nGenerating plots...")
    plot_results(y_test, y_pred, target_columns)

    # Save model and scaler
    print("\nSaving model and scaler...")
    joblib.dump(model, "model.joblib")
    joblib.dump(scaler, "scaler.joblib")

    print("\nModel training completed successfully!")
    print("Files saved:")
    print("- model.joblib (trained model)")
    print("- scaler.joblib (fitted scaler)")
    print("- prediction_results.png (visualization of results)")


if __name__ == "__main__":
    main()
