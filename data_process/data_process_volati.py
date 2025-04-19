import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from datetime import datetime
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

def load_volatility_data(data_path):
    """
    Load volatility data from CSV files
    """
    all_data = []
    base_dir = Path("data/financial_metrics/volatility")
    processed_dir = Path("data/processed_data")
    plots_dir = Path("data/visualizations")
    
    # Create directories if they don't exist
    for dir_path in [base_dir, processed_dir, plots_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    try:
        for file in base_dir.glob('*.csv'):
            df = pd.read_csv(file)
            # Convert timestamp to datetime if it exists
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
                df = df.drop('date', axis=1)
            
            # Add symbol name from filename
            symbol = file.stem.split('_')[0]
            df['symbol'] = symbol
            all_data.append(df)
        
        if len(all_data) > 0:
            combined_df = pd.concat(all_data, ignore_index=True)
            # Sort by timestamp if it exists
            if 'timestamp' in combined_df.columns:
                combined_df = combined_df.sort_values('timestamp')
            return combined_df
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def detect_outliers_iqr(df, column, k=1.5):
    """
    Detect outliers using the IQR method with dynamic thresholds
    """
    try:
        # Calculate rolling statistics
        rolling_median = df[column].rolling(window=20, center=True).median()
        q1 = df[column].rolling(window=20, center=True).quantile(0.25)
        q3 = df[column].rolling(window=20, center=True).quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        
        return (df[column] >= lower_bound) & (df[column] <= upper_bound)
    except Exception as e:
        print(f"Error detecting outliers: {str(e)}")
        return pd.Series(True, index=df.index)

def remove_outliers(df, columns):
    """
    Remove outliers using multiple methods and dynamic thresholds
    """
    try:
        df_clean = df.copy()
        
        for column in columns:
            if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                # Method 1: IQR method with rolling windows
                mask_iqr = detect_outliers_iqr(df, column)
                
                # Method 2: Z-score with rolling windows
                rolling_mean = df[column].rolling(window=20, center=True).mean()
                rolling_std = df[column].rolling(window=20, center=True).std()
                z_scores = np.abs((df[column] - rolling_mean) / rolling_std)
                mask_zscore = z_scores < 3
                
                # Combine both methods (keep data point if it passes either test)
                combined_mask = mask_iqr | mask_zscore
                
                # Replace outliers with rolling median
                rolling_median = df[column].rolling(window=20, center=True).median()
                df_clean.loc[~combined_mask, column] = rolling_median[~combined_mask]
        
        return df_clean
    except Exception as e:
        print(f"Error removing outliers: {str(e)}")
        return df

def handle_missing_data(df):
    """
    Handle missing data with sophisticated methods
    """
    try:
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Forward fill for short gaps (up to 3 days)
        df = df.ffill(limit=3)
        
        # For longer gaps, use linear interpolation
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].interpolate(method='linear')
        
        # For any remaining NaNs at the edges, use nearest value
        df = df.bfill()
        df = df.ffill()
        
        return df
    except Exception as e:
        print(f"Error handling missing data: {str(e)}")
        return df

def analyze_seasonality(df, column):
    """
    Analyze and decompose seasonality in the data
    """
    try:
        # Create a copy of the dataframe with only timestamp and target column
        df_seasonal = df[['timestamp', column]].copy()
        
        # Ensure data is daily and continuous
        df_daily = df_seasonal.set_index('timestamp').resample('D').mean()
        df_daily = df_daily.interpolate(method='linear')
        
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(df_daily[column], period=252)  # 252 trading days in a year
        
        return decomposition
    except Exception as e:
        print(f"Error in seasonality analysis: {str(e)}")
        return None

def process_volatility_data(input_path, output_path, start_year=2004):
    """
    Main function to process volatility data with enhanced methods
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Load data
        df = load_volatility_data(input_path)
        if df is None:
            print("No data found in the specified directory")
            return None
        
        # Filter data from start_year onwards
        df = df[df['timestamp'].dt.year >= start_year]
        
        # Process each symbol separately
        processed_dfs = []
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            
            # Remove outliers from numeric columns
            numeric_columns = symbol_data.select_dtypes(include=[np.number]).columns
            symbol_data = remove_outliers(symbol_data, numeric_columns)
            
            # Handle missing data
            symbol_data = handle_missing_data(symbol_data)
            
            processed_dfs.append(symbol_data)
        
        # Combine all processed data
        processed_df = pd.concat(processed_dfs, ignore_index=True)
        
        # Save processed data
        output_file = os.path.join(output_path, f'processed_volatility_data_{datetime.now().strftime("%Y%m%d")}.csv')
        processed_df.to_csv(output_file, index=False)
        
        return processed_df
    except Exception as e:
        print(f"Error processing volatility data: {str(e)}")
        return None

def plot_volatility_analysis(df, plots_dir, file_name):
    """
    Create comprehensive analysis plots
    """
    try:
        os.makedirs(plots_dir, exist_ok=True)
        
        # Set up the style
        sns.set_style("whitegrid")
        
        # Time series plot with rolling statistics
        plt.figure(figsize=(15, 8))
        if 'timestamp' in df.columns:
            x_axis = df['timestamp']
        else:
            x_axis = df.index
            
        plt.plot(x_axis, df['atr'], label='ATR', alpha=0.6)
        
        # Add rolling mean and std
        rolling_mean = df['atr'].rolling(window=20).mean()
        rolling_std = df['atr'].rolling(window=20).std()
        plt.plot(x_axis, rolling_mean, label='20-day MA', linewidth=2)
        plt.fill_between(x_axis, 
                        rolling_mean - 2*rolling_std,
                        rolling_mean + 2*rolling_std,
                        alpha=0.2, label='±2σ Band')
        
        plt.title('Volatility (ATR) Over Time')
        plt.xlabel('Date')
        plt.ylabel('ATR')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{file_name}_volatility_time_series.png")
        plt.close()
        
        # Distribution analysis
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x='atr', kde=True, bins=50)
        plt.axvline(df['atr'].mean(), color='r', linestyle='--', label='Mean')
        plt.axvline(df['atr'].median(), color='g', linestyle='--', label='Median')
        plt.title('ATR Distribution')
        plt.xlabel('ATR Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"{file_name}_atr_distribution.png")
        plt.close()
        
        # Seasonality analysis
        decomposition = analyze_seasonality(df, 'atr')
        if decomposition is not None:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
            decomposition.observed.plot(ax=ax1)
            ax1.set_title('Observed')
            decomposition.trend.plot(ax=ax2)
            ax2.set_title('Trend')
            decomposition.seasonal.plot(ax=ax3)
            ax3.set_title('Seasonal')
            decomposition.resid.plot(ax=ax4)
            ax4.set_title('Residual')
            plt.tight_layout()
            plt.savefig(plots_dir / f"{file_name}_seasonality_analysis.png")
            plt.close()
        
        # QQ Plot for normality check
        plt.figure(figsize=(10, 6))
        stats.probplot(df['atr'], dist="norm", plot=plt)
        plt.title('ATR Q-Q Plot')
        plt.tight_layout()
        plt.savefig(plots_dir / f"{file_name}_volume_boxplot.png")
        plt.close()
        
    except Exception as e:
        print(f"Error in plot_volatility_analysis: {str(e)}")

if __name__ == "__main__":
    # Define paths
    base_dir = Path("data/financial_metrics/volatility")
    processed_dir = Path("data/processed_data")
    plots_dir = Path("data/visualizations")
    
    # Create directories if they don't exist
    for dir_path in [base_dir, processed_dir, plots_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Process data
        processed_df = process_volatility_data(base_dir, processed_dir)
        
        if processed_df is not None:
            # Create visualizations for each file
            for file_path in base_dir.glob("*.csv"):
                file_name = file_path.stem
                symbol_data = processed_df[processed_df['symbol'] == file_name.split('_')[0]]
                plot_volatility_analysis(symbol_data, plots_dir, file_name)
            print("Data processing and visualization completed successfully!")
        else:
            print("Error processing data!")
    except Exception as e:
        print(f"Main execution error: {str(e)}")
