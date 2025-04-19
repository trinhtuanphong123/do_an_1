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
    Load the SMA data from CSV file
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

def filter_data_from_2004(df):
    """
    Filter data from 2004 onwards
    """
    print("\nFiltering data from 2004...")
    df_filtered = df[df.index >= '2004-01-01']
    print(f"Data shape after filtering: {df_filtered.shape}")
    return df_filtered

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

def normalize_data(df):
    """
    Normalize the data using StandardScaler
    """
    print("\nNormalizing data...")
    scaler = StandardScaler()
    df_normalized = pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=df.columns
    )
    print("Data normalized successfully.")
    return df_normalized

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

def visualize_data(original_df, processed_df, output_dir, sma_period):
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
    plt.title(f'Original SMA-{sma_period} Data (from 2004)')
    plt.legend()
    plt.grid(True)
    
    # Plot processed data
    plt.subplot(2, 1, 2)
    for column in processed_df.columns:
        plt.plot(processed_df.index, processed_df[column], label=column)
    plt.title(f'Processed SMA-{sma_period} Data (Normalized)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sma_{sma_period}_comparison.png'))
    plt.show()
    
    # Create box plots to compare distributions
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    original_df.boxplot()
    plt.title(f'Original SMA-{sma_period} Distribution')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    processed_df.boxplot()
    plt.title(f'Processed SMA-{sma_period} Distribution')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sma_{sma_period}_distribution.png'))
    plt.show()
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(processed_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation Heatmap of Processed SMA-{sma_period} Data')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sma_{sma_period}_correlation.png'))
    plt.show()

def process_sma_file(input_file, output_file, output_dir, sma_period):
    """
    Process a single SMA file
    """
    print(f"\nProcessing SMA-{sma_period} data...")
    
    # Load data
    df = load_data(input_file)
    
    # Filter data from 2004
    df_filtered = filter_data_from_2004(df)
    
    # Handle missing values
    df_no_missing = handle_missing_values(df_filtered)
    
    # Remove outliers
    df_clean = remove_outliers(df_no_missing)
    
    # Normalize data
    df_normalized = normalize_data(df_clean)
    
    # Check stationarity
    stationarity_results = check_stationarity(df_normalized)
    
    # Save processed data
    save_processed_data(df_normalized, output_file)
    
    # Visualize data
    visualize_data(df_filtered, df_normalized, output_dir, sma_period)
    
    print(f"\nSMA-{sma_period} data processing completed successfully!")

def main():
    # Define paths
    sma_periods = [20, 50, 200]
    base_input_path = 'data/technical_data/sma/AAPL_SMA_{}.csv'
    base_output_path = 'data/processed_data/processed_AAPL_SMA_{}.csv'
    output_dir = 'data/visualizations'
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(base_output_path.format(20)), exist_ok=True)
    
    # Process each SMA file
    for period in sma_periods:
        input_file = base_input_path.format(period)
        output_file = base_output_path.format(period)
        process_sma_file(input_file, output_file, output_dir, period)
    
    print("\nAll SMA data processing completed successfully!")

if __name__ == "__main__":
    main()
