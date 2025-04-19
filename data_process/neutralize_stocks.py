import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')
import statsmodels.api as sm

def setup_directories():
    """Thiết lập các thư mục cần thiết"""
    price_dir = Path("data/price_data")
    processed_dir = Path("data/processed_data")
    volume_dir = Path("data/volume_data")
    processed_volume_dir = volume_dir / "processed"
    visual_dir = Path("data/visual_data")
    
    # Tạo các thư mục nếu chưa tồn tại
    for dir_path in [price_dir, processed_dir, volume_dir, processed_volume_dir, visual_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return price_dir, processed_dir, processed_volume_dir, visual_dir

def load_price_data(price_dir):
    """
    Đọc dữ liệu giá từ thư mục price_data
    Args:
        price_dir: đường dẫn đến thư mục chứa dữ liệu giá
    Returns:
        DataFrame: dữ liệu giá đã được đọc
    """
    try:
        csv_files = list(price_dir.glob("*.csv"))
        
        if not csv_files:
            print("\nKhông tìm thấy file CSV nào trong thư mục data/price_data")
            return None
        
        file_path = csv_files[0]
        df = pd.read_csv(file_path)
        
        date_columns = df.select_dtypes(include=['object']).columns
        for col in date_columns:
            if 'date' in col.lower() or 'time' in col.lower():
                df[col] = pd.to_datetime(df[col])
                df.set_index(col, inplace=True)
        
        df = df[df.index >= '2004-01-01']
        
        print(f"\nĐã đọc thành công file giá: {file_path}")
        print(f"Số lượng dòng dữ liệu: {len(df)}")
        print(f"Thời gian từ: {df.index.min()} đến {df.index.max()}")
        
        return df
    except Exception as e:
        print(f"\nLỗi khi đọc file giá: {str(e)}")
        return None

def load_volume_data(volume_dir):
    """
    Đọc dữ liệu khối lượng đã xử lý
    Args:
        volume_dir: đường dẫn đến thư mục chứa dữ liệu khối lượng đã xử lý
    Returns:
        DataFrame: dữ liệu khối lượng đã được đọc
    """
    try:
        csv_files = list(volume_dir.glob("*.csv"))
        
        if not csv_files:
            print("\nKhông tìm thấy file CSV nào trong thư mục data/volume_data/processed")
            return None
        
        file_path = csv_files[0]
        df = pd.read_csv(file_path)
        
        date_columns = df.select_dtypes(include=['object']).columns
        for col in date_columns:
            if 'date' in col.lower() or 'time' in col.lower():
                df[col] = pd.to_datetime(df[col])
                df.set_index(col, inplace=True)
        
        print(f"\nĐã đọc thành công file khối lượng: {file_path}")
        print(f"Số lượng dòng dữ liệu: {len(df)}")
        
        return df
    except Exception as e:
        print(f"\nLỗi khi đọc file khối lượng: {str(e)}")
        return None

def calculate_metrics(price_df, volume_df):
    """
    Tính toán các chỉ số cơ bản
    Args:
        price_df: DataFrame dữ liệu giá
        volume_df: DataFrame dữ liệu khối lượng
    Returns:
        DataFrame: dữ liệu đã được tính toán các chỉ số
    """
    result_df = pd.DataFrame(index=price_df.index)
    
    # Thêm các cột từ dữ liệu giá và khối lượng
    for col in price_df.columns:
        result_df[f"price_{col}"] = price_df[col]
    for col in volume_df.columns:
        result_df[f"volume_{col}"] = volume_df[col]
    
    # Tính các chỉ số
    if 'price_close' in result_df.columns:
        result_df['return'] = result_df['price_close'].pct_change()
        result_df['volatility'] = result_df['return'].rolling(window=20).std()
        result_df['momentum'] = result_df['price_close'].pct_change(periods=20)
    
    if 'volume_volume' in result_df.columns:
        result_df['volume_ma'] = result_df['volume_volume'].rolling(window=20).mean()
    
    # Xóa các hàng có giá trị NaN
    result_df = result_df.dropna()
    
    return result_df

def neutralize_data(df, window=20, neutralize_method='robust', regression_method='OLS'):
    """
    Thực hiện neutralization bằng cả hai phương pháp: time series và cross-sectional
    
    Args:
        df: DataFrame dữ liệu gốc
        window: Cửa sổ thời gian cho rolling calculations
        neutralize_method: 'robust' hoặc 'standard' cho time series neutralization
        regression_method: 'OLS' hoặc 'robust' cho cross-sectional neutralization
    
    Returns:
        DataFrame với các cột đã được neutralize
    """
    result_df = df.copy()
    metrics_to_neutralize = ['return', 'volatility', 'momentum', 'volume_ma']
    
    # 1. Time Series Neutralization
    for metric in metrics_to_neutralize:
        if metric in result_df.columns:
            # a. Time series neutralization
            if neutralize_method == 'robust':
                # Sử dụng median và IQR
                rolling_median = result_df[metric].rolling(window=window, min_periods=window//2).median()
                rolling_q1 = result_df[metric].rolling(window=window, min_periods=window//2).quantile(0.25)
                rolling_q3 = result_df[metric].rolling(window=window, min_periods=window//2).quantile(0.75)
                rolling_iqr = rolling_q3 - rolling_q1
                rolling_iqr = rolling_iqr.replace(0, np.nan)
                ts_neutral = (result_df[metric] - rolling_median) / rolling_iqr
            else:
                # Sử dụng mean và std
                rolling_mean = result_df[metric].rolling(window=window, min_periods=window//2).mean()
                rolling_std = result_df[metric].rolling(window=window, min_periods=window//2).std()
                ts_neutral = (result_df[metric] - rolling_mean) / rolling_std
            
            # Xử lý outliers và missing values
            ts_neutral = ts_neutral.clip(lower=-3, upper=3)  # Winsorization
            ts_neutral = ts_neutral.replace([np.inf, -np.inf], np.nan)
            ts_neutral = ts_neutral.ffill()  # Thay thế fillna(method='ffill')
            
            # Lưu kết quả time series neutralization
            result_df[f'{metric}_ts_neutral'] = ts_neutral
            
            try:
                # b. Cross-sectional neutralization
                if 'market_return' in result_df.columns:
                    X = sm.add_constant(result_df['market_return'])
                    y = result_df[metric]
                    
                    if regression_method == 'robust':
                        model = sm.RLM(y, X).fit()
                    else:
                        model = sm.OLS(y, X).fit()
                    
                    # Lấy residuals làm giá trị đã neutralize
                    cs_neutral = model.resid
                    
                    # Standardize residuals
                    cs_neutral = (cs_neutral - cs_neutral.mean()) / cs_neutral.std()
                    result_df[f'{metric}_cs_neutral'] = cs_neutral
                    
                    # In thống kê
                    print(f"\nNeutralization cho {metric}:")
                    print(f"Time Series - Mean: {ts_neutral.mean():.4f}, Std: {ts_neutral.std():.4f}")
                    print(f"Cross Sectional - Mean: {cs_neutral.mean():.4f}, Std: {cs_neutral.std():.4f}")
                    if regression_method == 'OLS':
                        print(f"R-squared: {model.rsquared:.4f}")
            except Exception as e:
                print(f"Lỗi trong cross-sectional neutralization cho {metric}: {str(e)}")
    
    # 2. Calculate combined neutralization
    for metric in metrics_to_neutralize:
        if f'{metric}_ts_neutral' in result_df.columns and f'{metric}_cs_neutral' in result_df.columns:
            # Kết hợp cả hai phương pháp với trọng số bằng nhau
            result_df[f'{metric}_combined_neutral'] = (
                0.5 * result_df[f'{metric}_ts_neutral'] + 
                0.5 * result_df[f'{metric}_cs_neutral']
            )
    
    return result_df


def create_visualizations(df, visual_dir):
    """
    Tạo các biểu đồ phân tích với nhiều loại neutralization
    """
    metrics = ['return', 'volatility', 'momentum', 'volume_ma']
    neutral_types = ['ts_neutral', 'cs_neutral', 'combined_neutral']
    
    # 1. Distribution plots
    plt.figure(figsize=(15, len(metrics) * 5))
    for i, metric in enumerate(metrics, 1):
        if metric in df.columns:
            plt.subplot(len(metrics), 1, i)
            
            # Plot original data
            sns.kdeplot(data=df[metric], label='Original', color='blue')
            
            # Plot neutralized data
            colors = ['red', 'green', 'purple']
            for neutral_type, color in zip(neutral_types, colors):
                col_name = f'{metric}_{neutral_type}'
                if col_name in df.columns:
                    sns.kdeplot(data=df[col_name], label=neutral_type, color=color)
            
            plt.title(f'{metric.capitalize()} Distribution')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(visual_dir / 'distributions.png')
    plt.close()
    
    # 2. Time Series plots
    for metric in metrics:
        if metric in df.columns:
            plt.figure(figsize=(15, 8))
            
            # Plot original data
            plt.plot(df.index[-100:], df[metric][-100:], 
                    label='Original', color='blue', alpha=0.7)
            
            # Plot neutralized data
            colors = ['red', 'green', 'purple']
            for neutral_type, color in zip(neutral_types, colors):
                col_name = f'{metric}_{neutral_type}'
                if col_name in df.columns:
                    plt.plot(df.index[-100:], df[col_name][-100:], 
                            label=neutral_type, color=color, alpha=0.7)
            
            plt.title(f'{metric.capitalize()} Time Series (Last 100 Days)')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plt.savefig(visual_dir / f'{metric}_timeseries.png')
            plt.close()


def save_results(df, processed_dir, visual_dir):
    """
    Lưu kết quả vào thư mục processed_data
    Args:
        df: DataFrame dữ liệu đã xử lý
        processed_dir: đường dẫn đến thư mục processed_data
    """
    try:
        # Tạo thư mục nếu chưa tồn tại
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Lưu dữ liệu đã chuẩn hóa với xử lý lỗi quyền truy cập
        try:
            neutralized_file = processed_dir / "time_neutralized_data.csv"
            df.to_csv(neutralized_file, index=True, encoding='utf-8')
            print(f"\nĐã lưu dữ liệu đã chuẩn hóa tại: {neutralized_file}")
        except PermissionError:
            alternative_file = processed_dir / "time_neutralized_data_new.csv"
            print(f"\nKhông thể ghi đè file cũ. Đang thử lưu vào file mới: {alternative_file}")
            df.to_csv(alternative_file, index=True, encoding='utf-8')
        
        # Lưu thống kê mô tả
        try:
            stats_file = processed_dir / "time_neutralization_stats.txt"
            with open(stats_file, 'w', encoding='utf-8') as f:
                # Thống kê cơ bản
                f.write("Thống kê mô tả:\n")
                f.write("---------------\n")
                metrics = ['return', 'volatility', 'momentum', 'volume_ma']
                
                for metric in metrics:
                    if metric in df.columns and f'{metric}_time_neutral' in df.columns:
                        f.write(f"\n{metric.upper()}:\n")
                        f.write("Original:\n")
                        f.write(df[metric].describe().to_string())
                        f.write("\n\nTime Neutralized:\n")
                        f.write(df[f'{metric}_time_neutral'].describe().to_string())
                        f.write("\n")
                
                # Tương quan
                f.write("\n\nMa trận tương quan:\n")
                f.write("------------------\n")
                correlation_cols = [col for col in df.columns if '_time_neutral' in col]
                correlation_matrix = df[correlation_cols].corr()
                f.write(correlation_matrix.to_string())
            
            print(f"Đã lưu thống kê mô tả tại: {stats_file}")
        except PermissionError:
            alternative_stats = processed_dir / "time_neutralization_stats_new.txt"
            print(f"\nKhông thể ghi đè file thống kê cũ. Đang thử lưu vào file mới: {alternative_stats}")
            with open(alternative_stats, 'w', encoding='utf-8') as f:
                # Thống kê cơ bản
                f.write("Thống kê mô tả:\n")
                f.write("---------------\n")
                metrics = ['return', 'volatility', 'momentum', 'volume_ma']
                
                for metric in metrics:
                    if metric in df.columns and f'{metric}_time_neutral' in df.columns:
                        f.write(f"\n{metric.upper()}:\n")
                        f.write("Original:\n")
                        f.write(df[metric].describe().to_string())
                        f.write("\n\nTime Neutralized:\n")
                        f.write(df[f'{metric}_time_neutral'].describe().to_string())
                        f.write("\n")
                
                # Tương quan
                f.write("\n\nMa trận tương quan:\n")
                f.write("------------------\n")
                correlation_cols = [col for col in df.columns if '_time_neutral' in col]
                correlation_matrix = df[correlation_cols].corr()
                f.write(correlation_matrix.to_string())
        
    except Exception as e:
        print(f"Lỗi khi lưu file: {str(e)}")
        print("Thử tạo tên file thay thế...")

def main():
    # Thiết lập thư mục
    price_dir, processed_dir, processed_volume_dir, visual_dir = setup_directories()
    
    # Đọc dữ liệu
    price_df = load_price_data(price_dir)
    if price_df is None:
        return
    
    volume_df = load_volume_data(processed_volume_dir)
    if volume_df is None:
        return
    
    # Tính toán các chỉ số cơ bản
    metrics_df = calculate_metrics(price_df, volume_df)
    
    # Thực hiện neutralization
    final_df = neutralize_data(
        metrics_df,
        window=20,
        neutralize_method='robust',
        regression_method='OLS'
    )
    
    # Tạo biểu đồ
    create_visualizations(final_df, visual_dir)
    
    # Lưu kết quả
    save_results(final_df, processed_dir, visual_dir)
    
    print("\nHoàn thành xử lý dữ liệu!")

if __name__ == "__main__":
    main()