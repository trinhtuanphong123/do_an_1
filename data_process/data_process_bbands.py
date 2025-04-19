import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
import os
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

def load_data(file_path):
    """
    Load the Bollinger Bands data from CSV file
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(
        file_path,
        parse_dates=['Timestamp'],  # Chỉ định cột cần parse
        index_col='Timestamp'       # Đặt làm index
    )
    # Đổi tên index thành 'Date'
    df.index.name = 'Date'
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

    

def handle_missing_values(df):
    """
    Handle missing values in the dataset
    """
    print("\nChecking for missing values...")
    missing_values = df.isnull().sum()
    print(f"Missing values:\n{missing_values}")
    
    if missing_values.sum() > 0:
        print("Filling missing values with forward fill method...")
        df = df.fillna(method='ffill')
        # If there are still missing values (at the beginning), fill with backward fill
        df = df.fillna(method='bfill')
        print("Missing values after filling:", df.isnull().sum().sum())
    else:
        print("No missing values found.")
    
    return df

def remove_outliers(df, threshold=3):
    """
    Remove outliers using the Z-score method
    """
    print("\nRemoving outliers...")
    df_clean = df.copy()
    
    for column in df.columns:
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = z_scores > threshold
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            print(f"Found {outlier_count} outliers in {column}")
            # Replace outliers with NaN
            df_clean.loc[outliers, column] = np.nan
    
    # Fill NaN values created by outlier removal
    df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
    print("Outliers removed and replaced.")
    
    return df_clean



def check_stationarity(df):
    """
    Check the stationarity of the time series using Augmented Dickey-Fuller test
    """
    print("\nChecking time series stationarity...")
    results = {}
    
    for column in df.columns:
        adf_result = adfuller(df[column].dropna())
        results[column] = {
            'ADF Statistic': adf_result[0],
            'p-value': adf_result[1],
            'Critical values': adf_result[4]
        }
        
        print(f"\nResults for {column}:")
        print(f"ADF Statistic: {adf_result[0]:.4f}")
        print(f"p-value: {adf_result[1]:.4f}")
        print("Critical values:")
        for key, value in adf_result[4].items():
            print(f"\t{key}: {value:.4f}")
        
        if adf_result[1] <= 0.05:
            print("Conclusion: The series is stationary")
        else:
            print("Conclusion: The series is non-stationary")
    
    return results

def save_processed_data(df, output_path):
    """
    Save the processed data to a CSV file
    """
    print(f"\nSaving processed data to {output_path}...")
    df.to_csv(output_path)
    print("Data saved successfully.")

def visualize_data(original_df, processed_df, output_dir):
    """
    Create visualizations of the original and processed data
    """
    print("\nCreating visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot original data
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    for column in original_df.columns:
        plt.plot(original_df.index, original_df[column], label=column)
    plt.title('Original Bollinger Bands Data')
    plt.legend()
    plt.grid(True)
    
    # Plot processed data
    plt.subplot(2, 1, 2)
    for column in processed_df.columns:
        plt.plot(processed_df.index, processed_df[column], label=column)
    plt.title('Processed Bollinger Bands Data (Normalized)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bbands_comparison.png'))
    plt.show()
    
    # Create box plots to compare distributions
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    original_df.boxplot()
    plt.title('Original Data Distribution')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    processed_df.boxplot()
    plt.title('Processed Data Distribution')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bbands_distribution.png'))
    plt.show()
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(processed_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of Processed Data')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bbands_correlation.png'))
    plt.show()

def main():
    # Define paths
    input_file = 'data/technical_data/bbands/AAPL_BBANDS.csv'
    output_file = 'data/processed_data/processed_AAPL_BBANDS.csv'
    output_dir = 'data/visualizations'
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_data(input_file)
    
    # Handle missing values
    df_no_missing = handle_missing_values(df)
    
    # Remove outliers
    df_clean = remove_outliers(df_no_missing)
    

    
    # Check stationarity
    stationarity_results = check_stationarity(df_clean)
    
    # Save processed data
    save_processed_data(df_clean, output_file)
    
    # Visualize data
    visualize_data(df, df_clean, output_dir)

    

if __name__ == "__main__":
    main()
