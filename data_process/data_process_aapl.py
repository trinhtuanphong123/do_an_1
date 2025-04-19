import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

def load_data(file_path):
    """
    Load data from CSV file and filter from 2004
    """
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    # Filter data from 2004
    df = df[df.index.year >= 2004]
    return df

def handle_outliers(df, columns, n_std=3):
    """
    Handle outliers using the rolling z-score method with winsorization.
    Outliers được giới hạn bằng cách capping về ngưỡng dưới hoặc trên.
    
    Parameters:
    - df: DataFrame chứa dữ liệu
    - columns: Danh sách các cột cần xử lý
    - n_std: Số độ lệch chuẩn để xác định ngưỡng (mặc định là 3)
    
    Returns:
    - df: DataFrame sau khi xử lý outlier
    """
    for column in columns:
        # Tính rolling mean và std với window 30 (sử dụng min_periods=1 để tránh NaN ở đầu)
        rolling_mean = df[column].rolling(window=30, min_periods=1).mean()
        rolling_std = df[column].rolling(window=30, min_periods=1).std()
        
        # Tính giới hạn dưới và trên
        lower_bound = rolling_mean - n_std * rolling_std
        upper_bound = rolling_mean + n_std * rolling_std
        
        # Áp dụng winsorization: nếu giá trị nhỏ hơn lower_bound thì thay bằng lower_bound,
        # nếu giá trị lớn hơn upper_bound thì thay bằng upper_bound.
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

def backfill_missing_data(df):
    """
    Backfill missing data using forward fill then backward fill
    """
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    return df
def normalize_data(df, columns):
    """
    Chuẩn hóa dữ liệu bằng MinMaxScaler
    """
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    
    df_normalized = df.copy()
    df_normalized[columns] = scaler.fit_transform(df[columns])
    return df_normalized
def calculate_statistics(df):
    """
    Tính toán và hiển thị các chỉ số thống kê cơ bản
    """
    stats = df.describe()
    skewness = df.skew()
    kurtosis = df.kurtosis()
    
    print("\nThống kê cơ bản:")
    print(stats)
    print("\nĐộ lệch (Skewness):")
    print(skewness)
    print("\nĐộ nhọn (Kurtosis):")
    print(kurtosis)
def plot_price_comparison(original_df, processed_df):
    """
    Plot price comparison before and after processing
    """
    plt.figure(figsize=(15, 10))
    
    # Plot original data
    plt.subplot(2, 1, 1)
    plt.plot(original_df.index, original_df['close'], label='Original', color='blue')
    plt.title('Original Close Price (2004-Present)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()
    
    # Plot processed data
    plt.subplot(2, 1, 2)
    plt.plot(processed_df.index, processed_df['close'], label='Processed', color='red')
    plt.title('Processed Close Price (2004-Present)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_boxplot_comparison(original_df, processed_df):
    """
    Plot boxplot comparison for all numeric columns
    """
    numeric_columns = original_df.select_dtypes(include=[np.number]).columns
    
    plt.figure(figsize=(15, 8))
    for i, column in enumerate(numeric_columns, 1):
        plt.subplot(2, 3, i)
        data = [original_df[column], processed_df[column]]
        plt.boxplot(data, labels=['Original', 'Processed'])
        plt.title(f'{column} Distribution (2004-Present)')
        plt.ylabel('Value')
    
    plt.tight_layout()
    plt.show()

def plot_missing_values(df):
    """
    Plot missing values heatmap
    """
    plt.figure(figsize=(15, 7))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap (2004-Present)')
    plt.xlabel('Columns')
    plt.ylabel('Date')
    plt.show()

def process_aapl_data():
    # Define paths
    data_dir = Path('data/price_data')
    input_file = data_dir / 'AAPL_daily.csv'
    base_dir = Path('data/processed_data')
    output_file = base_dir / 'processed_AAPL_daily.csv'
    
    # Tạo thư mục nếu chưa tồn tại
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    original_df = load_data(input_file)
    
    # Hiển thị thông tin ban đầu
    print("\nInitial data info (2004-Present):")
    print(original_df.info())
    print("\nMissing values:")
    print(original_df.isnull().sum())
    
    # Tính toán thống kê ban đầu
    calculate_statistics(original_df)
    
    # Plot initial data visualizations
    print("\nGenerating initial data visualizations...")
    plot_missing_values(original_df)
    
    # Create a copy for processing
    processed_df = original_df.copy()
    
    
    # Handle outliers
    print("\nHandling outliers...")
    numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
    processed_df = handle_outliers(processed_df, numeric_columns)
    
    # Backfill missing data
    print("\nBackfilling missing data...")
    processed_df = backfill_missing_data(processed_df)
    
    # Chuẩn hóa dữ liệu
    print("\nNormalizing data...")
    processed_df = normalize_data(processed_df, numeric_columns)
    
    # Display final data info
    print("\nFinal data info:")
    print(processed_df.info())
    calculate_statistics(processed_df)
    
    # Plot comparison visualizations
    print("\nGenerating comparison visualizations...")
    plot_price_comparison(original_df, processed_df)
    plot_boxplot_comparison(original_df, processed_df)
    
    # Save processed data
    print("\nSaving processed data...")
    processed_df.to_csv(output_file)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    process_aapl_data()
