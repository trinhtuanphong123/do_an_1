import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats
import pathlib
import os
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

def load_data(file_path):
    """
    Load the market data from CSV file
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df



def handle_duplicates(df):
    """
    Handle duplicate dates by keeping the last entry
    """
    if df.index.duplicated().any():
        print("\nRemoving duplicate dates...")
        df = df[~df.index.duplicated(keep='last')]
        print(f"Shape after removing duplicates: {df.shape}")
    return df

def handle_missing_values(df):
    """
    Handle missing values using multiple methods
    """
    print("\nHandling missing values...")
    
    # Store original missing counts
    missing_before = df.isnull().sum()
    
    # Handle price columns
    price_cols = ['open', 'high', 'low', 'close']
    
    # 1. Forward fill for missing prices (use previous day's values)
    df[price_cols] = df[price_cols].fillna(method='ffill')
    
    # 2. Backward fill for any remaining missing prices
    df[price_cols] = df[price_cols].fillna(method='bfill')
    
    # 3. For volume, use median of nearby values
    df['volume'] = df['volume'].fillna(df['volume'].rolling(window=5, center=True).median())
    
    # If any remaining missing values in volume, use forward fill
    df['volume'] = df['volume'].fillna(method='ffill').fillna(method='bfill')
    
    # Report results
    missing_after = df.isnull().sum()
    print("\nMissing values before:")
    print(missing_before)
    print("\nMissing values after:")
    print(missing_after)
    
    return df

def handle_zero_values(df):
    """
    Handle zero values in price and volume
    """
    print("\nHandling zero values...")
    
    # Store original zero counts
    zeros_before = (df == 0).sum()
    
    # Handle zero prices
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        zero_mask = df[col] == 0
        if zero_mask.any():
            # Replace zeros with previous non-zero value
            df.loc[zero_mask, col] = df[col].replace(0, method='ffill')
    
    # Handle zero volume
    zero_volume_mask = df['volume'] == 0
    if zero_volume_mask.any():
        # Replace zero volume with median of non-zero values
        non_zero_median = df.loc[~zero_volume_mask, 'volume'].median()
        df.loc[zero_volume_mask, 'volume'] = non_zero_median
    
    # Report results
    zeros_after = (df == 0).sum()
    print("\nZero values before:")
    print(zeros_before)
    print("\nZero values after:")
    print(zeros_after)
    
    return df

def remove_outliers(df, threshold=3):
    """
    Remove outliers using multiple methods
    """
    print("\nRemoving outliers...")
    df_clean = df.copy()
    
    price_cols = ['open', 'high', 'low', 'close']
    outlier_stats = {}
    
    for column in price_cols:
        # 1. Z-score method
        z_scores = np.abs(stats.zscore(df[column]))
        z_score_outliers = z_scores > threshold
        
        # 2. IQR method
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        iqr_outliers = (df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))
        
        # 3. Combine both methods (consider a point an outlier if both methods agree)
        combined_outliers = z_score_outliers & iqr_outliers
        outlier_count = combined_outliers.sum()
        
        if outlier_count > 0:
            print(f"Found {outlier_count} outliers in {column}")
            # Store outlier information
            outlier_stats[column] = {
                'count': outlier_count,
                'dates': df.index[combined_outliers].tolist(),
                'values': df.loc[combined_outliers, column].tolist()
            }
            # Replace outliers with NaN
            df_clean.loc[combined_outliers, column] = np.nan
    
    # Fill NaN values using forward fill then backward fill
    df_clean[price_cols] = df_clean[price_cols].fillna(method='ffill').fillna(method='bfill')
    
    # Handle volume outliers separately using only IQR method
    Q1 = df['volume'].quantile(0.25)
    Q3 = df['volume'].quantile(0.75)
    IQR = Q3 - Q1
    volume_outliers = (df['volume'] < (Q1 - 3 * IQR)) | (df['volume'] > (Q3 + 3 * IQR))
    
    if volume_outliers.sum() > 0:
        print(f"Found {volume_outliers.sum()} outliers in volume")
        # Replace volume outliers with median of nearby values
        df_clean.loc[volume_outliers, 'volume'] = df_clean['volume'].rolling(window=5, center=True).median()
    
    print("Outliers removed and replaced.")
    return df_clean, outlier_stats

def normalize_data(df):
    """
    Normalize the data using multiple methods
    """
    print("\nNormalizing data...")
    df_normalized = df.copy()
    
    # 1. StandardScaler for price columns
    standard_scaler = StandardScaler()
    price_cols = ['open', 'high', 'low', 'close']
    df_normalized[price_cols] = standard_scaler.fit_transform(df[price_cols])
    
    # 2. RobustScaler for volume (less sensitive to outliers)
    robust_scaler = RobustScaler()
    df_normalized['volume'] = robust_scaler.fit_transform(df[['volume']])
    
    # 3. Calculate additional normalized features
    df_normalized['price_range'] = df_normalized['high'] - df_normalized['low']
    df_normalized['price_change'] = df_normalized['close'] - df_normalized['open']
    
    print("Data normalized successfully.")
    return df_normalized



def visualize_preprocessing(original_df, processed_df, output_dir):
    """
    Create comprehensive preprocessing visualizations
    """
    print("\nCreating preprocessing visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Price Comparison
    plt.figure(figsize=(15, 8))
    plt.plot(original_df.index, original_df['close'], label='Original Close', alpha=0.5)
    plt.plot(processed_df.index, processed_df['close'], label='Processed Close', alpha=0.8)
    plt.title('Price Data - Before and After Preprocessing')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'price_comparison.png'))
    plt.show()
    
    # 2. Volume Comparison
    plt.figure(figsize=(15, 8))
    plt.plot(original_df.index, original_df['volume'], label='Original Volume', alpha=0.5)
    plt.plot(processed_df.index, processed_df['volume'], label='Processed Volume', alpha=0.8)
    plt.title('Volume Data - Before and After Preprocessing')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'volume_comparison.png'))
    plt.show()
    
    # 3. Distribution Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original price distribution
    sns.histplot(original_df['close'], bins=50, ax=axes[0,0])
    axes[0,0].set_title('Original Close Price Distribution')
    
    # Processed price distribution
    sns.histplot(processed_df['close'], bins=50, ax=axes[0,1])
    axes[0,1].set_title('Processed Close Price Distribution')
    
    # Original volume distribution
    sns.histplot(original_df['volume'], bins=50, ax=axes[1,0])
    axes[1,0].set_title('Original Volume Distribution')
    
    # Processed volume distribution
    sns.histplot(processed_df['volume'], bins=50, ax=axes[1,1])
    axes[1,1].set_title('Processed Volume Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_comparison.png'))
    plt.show()
    
    # 4. Technical Indicators
    if 'MA5' in processed_df.columns and 'MA20' in processed_df.columns:
        plt.figure(figsize=(15, 8))
        plt.plot(processed_df.index, processed_df['close'], label='Close Price', alpha=0.5)
        plt.plot(processed_df.index, processed_df['MA5'], label='5-day MA', alpha=0.8)
        plt.plot(processed_df.index, processed_df['MA20'], label='20-day MA', alpha=0.8)
        plt.title('Price with Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'technical_indicators.png'))
        plt.show()

def save_processed_data(df, output_path):
    """
    Save the processed data to a CSV file
    """
    print(f"\nSaving processed data to {output_path}...")
    df.to_csv(output_path)
    print("Data saved successfully.")

def main():
    # Define paths
    input_file = 'data/market_data/NDAQ_daily.csv'
    processed_dir ='data/processed_data'
    plots_dir = 'data/visualizations'
    output_file = os.path.join(processed_dir, 'processed_NDAQ_daily.csv')
    
    # Create output directories
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load data
    df = load_data(input_file)
    
    
    
    # Handle duplicates
    df = handle_duplicates(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Handle zero values
    df = handle_zero_values(df)
    
    # Remove outliers
    df_clean, outlier_stats = remove_outliers(df)
    
    
    
    # Visualize preprocessing results
    visualize_preprocessing(df, df_clean, plots_dir)
    
    # Save processed data
    save_processed_data(df_clean, output_file)
    
    print("\nData preprocessing completed successfully!")

if __name__ == "__main__":
    main()
