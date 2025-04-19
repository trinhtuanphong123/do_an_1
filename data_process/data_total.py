import os
import pandas as pd
from functools import reduce

def load_and_prepare_dataframe(file_path, date_col='Timestamp', value_cols=None, rename_cols=None):
    """
    Helper function to load and prepare individual DataFrames
    
    Parameters:
    - file_path: path to the CSV file
    - date_col: name of the date column
    - value_cols: list of columns to keep
    - rename_cols: dictionary to rename columns
    """
    df = pd.read_csv(file_path)
    
    # Convert date column to datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Select specified columns
    if value_cols:
        df = df[[date_col] + value_cols]
    
    # Rename columns if specified
    if rename_cols:
        df = df.rename(columns=rename_cols)
    
    return df

def merge_financial_data():
    # Set up paths
    base_dir = os.path.abspath(os.path.dirname(__file__))
    processed_dir = os.path.abspath(os.path.join(base_dir, "../data/processed_data"))
    output_dir = os.path.abspath(os.path.join(base_dir, "../data"))
    
    # First, load the reference timeline (AAPL daily data)
    reference_df = pd.read_csv(
        os.path.join(processed_dir, "processed_AAPL_daily.csv")
    )
    reference_df['Timestamp'] = pd.to_datetime(reference_df['Timestamp'])
    reference_dates = reference_df[
        (reference_df['Timestamp'] >= '2004-01-01')
    ]['Timestamp'].sort_values()
    
    # Create a template DataFrame with the reference timeline
    template_df = pd.DataFrame({'Timestamp': reference_dates})
    
    # Dictionary to define data loading configurations
    data_configs = {
        'RSI': {
            'file': 'processed_AAPL_rsi.csv',
            'value_cols': ['RSI'],
            'rename_cols': None
        },
        'OBV': {
            'file': 'processed_AAPL_OBV_daily.csv',
            'value_cols': ['obv'],
            'rename_cols': {'obv': 'OBV'}
        },
        'Volume': {
            'file': 'processed_NDAQ_daily.csv',
            'value_cols': ['volume'],
            'rename_cols': {'volume': 'Volume'}
        },
        'Volatility': {
            'file': 'processed_AAPL_volatility.csv',
            'value_cols': ['atr'],
            'rename_cols': {'timestamp': 'Timestamp', 'atr': 'Volatility'}
        },
        'BBands': {
            'file': 'processed_AAPL_BBANDS.csv',
            'value_cols': ['Real Upper Band'],
            'rename_cols': {'Real Upper Band': 'UpperBand'}
        },
        'EPS': {
            'file': 'processed_AAPL_daily_earnings.csv',
            'value_cols': ['daily_eps'],
            'rename_cols': {'daily_eps': 'DailyEPS'}
        },
        'Close': {
            'file': 'processed_AAPL_daily.csv',
            'value_cols': ['close'],
            'rename_cols': {'close': 'Close'}
        }
    }
    
    # Load and merge each dataset
    dfs = [template_df]
    for indicator, config in data_configs.items():
        file_path = os.path.join(processed_dir, config['file'])
        df = load_and_prepare_dataframe(
            file_path,
            date_col='Timestamp' if 'timestamp' not in config.get('rename_cols', {}) else 'timestamp',
            value_cols=config['value_cols'],
            rename_cols=config['rename_cols']
        )
        
        # Filter date range
        df = df[df['Timestamp'] >= '2004-01-01']
        
        # Merge with template
        dfs.append(df)
    
    # Merge all DataFrames
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='Timestamp', how='left'), dfs)
    
    # Sort by timestamp and reset index
    merged_df = merged_df.sort_values('Timestamp').reset_index(drop=True)
    
    # Handle missing values
    # First interpolate
    merged_df = merged_df.interpolate(method='time')
    
    # Then backfill any remaining NaN values at the start
    merged_df = merged_df.fillna(method='bfill')
    
    # Forward fill any remaining NaN values at the end
    merged_df = merged_df.fillna(method='ffill')
    
    # Save merged dataset
    output_file = os.path.join(output_dir, "merged_financial_data.csv")
    merged_df.to_csv(output_file, index=False)
    
    try:
        print(f"Merged dataset saved to {output_file}")
        print(f"Shape of merged dataset: {merged_df.shape}")
        print("\nColumns in merged dataset:")
        for col in merged_df.columns:
            print(f"- {col}")
    except UnicodeEncodeError:
        print("Merged dataset saved successfully")
    
    return merged_df

if __name__ == "__main__":
    merged_data = merge_financial_data()