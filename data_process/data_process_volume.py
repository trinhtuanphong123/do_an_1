
import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def setup_directories():
    """Thiết lập các thư mục cần thiết"""
    base_dir = Path("data/volume_data")
    processed_dir = Path("data/processed_data")
    plots_dir = Path("data/visualizations")
    
    # Tạo các thư mục nếu chưa tồn tại
    for dir_path in [base_dir, processed_dir, plots_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return base_dir, processed_dir, plots_dir

def load_nasdaq_data(file_path):
    """
    Đọc và xử lý dữ liệu NASDAQ
    Args:
        file_path: đường dẫn đến file dữ liệu
    Returns:
        DataFrame: dữ liệu đã được đọc và xử lý sơ bộ
    """
    try:
        # Đọc file CSV
        df = pd.read_csv(file_path)
        
        # Chuyển đổi cột ngày tháng nếu có
        date_columns = df.select_dtypes(include=['object']).columns
        for col in date_columns:
            if 'date' in col.lower() or 'time' in col.lower():
                df[col] = pd.to_datetime(df[col])
                # Đặt cột ngày làm index
                df.set_index(col, inplace=True)
        
        # Lọc dữ liệu từ năm 2004
        df = df[df.index >= '2004-01-01']
        
        print(f"\nĐã đọc thành công file: {file_path}")
        print(f"Số lượng dòng dữ liệu: {len(df)}")
        print(f"Thời gian từ: {df.index.min()} đến {df.index.max()}")
        
        return df
    except Exception as e:
        print(f"\nLỗi khi đọc file {file_path}: {str(e)}")
        return None

def clean_nasdaq_data(df):
    """
    Làm sạch dữ liệu NASDAQ
    Args:
        df: DataFrame cần xử lý
    Returns:
        DataFrame: dữ liệu đã được làm sạch
    """
    # Xử lý giá trị bị thiếu
    missing_values = df.isnull().sum()
    print("\nSố lượng giá trị bị thiếu:")
    print(missing_values[missing_values > 0])
    
    # Thay thế giá trị bị thiếu
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    # Xử lý giá trị ngoại lai cho các cột số
    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = df[column].clip(lower_bound, upper_bound)
    
    return df

def process_volume_data(df):
    """
    Xử lý dữ liệu volume
    Args:
        df: DataFrame cần xử lý
    Returns:
        DataFrame: dữ liệu đã được xử lý
    """
    # Chuẩn hóa dữ liệu volume
    volume_columns = [col for col in df.columns if 'volume' in col.lower()]
    if volume_columns:
        scaler = StandardScaler()
        df[volume_columns] = scaler.fit_transform(df[volume_columns])
    
    return df

def create_visualizations(df_original, df_processed, plots_dir, file_name):
    """
    Tạo các biểu đồ phân tích
    Args:
        df_original: DataFrame dữ liệu gốc
        df_processed: DataFrame dữ liệu đã xử lý
    """
    # Lấy các cột volume
    volume_columns = [col for col in df_original.columns if 'volume' in col.lower()]
    if not volume_columns:
        return
    
    # Vẽ biểu đồ volume theo thời gian
    plt.figure(figsize=(15, 10))
    
    # Vẽ dữ liệu gốc
    plt.subplot(2, 1, 1)
    for col in volume_columns:
        plt.plot(df_original.index, df_original[col], label=f'{col} (Gốc)')
    plt.title('Volume theo thời gian (Dữ liệu gốc)')
    plt.xlabel('Thời gian')
    plt.ylabel('Volume')
    plt.legend()
    
    # Vẽ dữ liệu đã xử lý
    plt.subplot(2, 1, 2)
    for col in volume_columns:
        plt.plot(df_processed.index, df_processed[col], label=f'{col} (Đã xử lý)')
    plt.title('Volume theo thời gian (Dữ liệu đã xử lý)')
    plt.xlabel('Thời gian')
    plt.ylabel('Volume (Chuẩn hóa)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Vẽ biểu đồ phân phối volume
    plt.figure(figsize=(15, 10))
    
    # Vẽ phân phối dữ liệu gốc
    for i, col in enumerate(volume_columns, 1):
        plt.subplot(2, len(volume_columns), i)
        sns.histplot(data=df_original, x=col)
        plt.title(f'Phân phối {col} (Gốc)')
    
    # Vẽ phân phối dữ liệu đã xử lý
    for i, col in enumerate(volume_columns, 1):
        plt.subplot(2, len(volume_columns), i + len(volume_columns))
        sns.histplot(data=df_processed, x=col)
        plt.title(f'Phân phối {col} (Đã xử lý)')
    
    plt.tight_layout()
    plt.show()
    
    # Vẽ biểu đồ box plot
    plt.figure(figsize=(15, 10))
    
    # Vẽ box plot dữ liệu gốc
    plt.subplot(2, 1, 1)
    df_original[volume_columns].boxplot()
    plt.title('Box Plot Volume (Dữ liệu gốc)')
    plt.xticks(rotation=45)
    
    # Vẽ box plot dữ liệu đã xử lý
    plt.subplot(2, 1, 2)
    df_processed[volume_columns].boxplot()
    plt.title('Box Plot Volume (Dữ liệu đã xử lý)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(plots_dir / f"{file_name}_volume_boxplot.png")
    plt.close()
    plt.show()

def save_processed_data(df, output_path):
    """
    Lưu dữ liệu đã xử lý
    Args:
        df: DataFrame cần lưu
        output_path: đường dẫn để lưu file
    """
    try:
        df.to_csv(output_path, index=True)
        print(f"\nĐã lưu dữ liệu đã xử lý tại: {output_path}")
    except Exception as e:
        print(f"Lỗi khi lưu file: {str(e)}")

def main():
    # Thiết lập thư mục
    base_dir, processed_dir,plots_dir = setup_directories()
    
    # Tìm tất cả các file CSV trong thư mục
    csv_files = list(base_dir.glob("*.csv"))
    
    if not csv_files:
        print("\nKhông tìm thấy file CSV nào trong thư mục data/volume_data")
        return
    
    # Xử lý từng file
    for file_path in csv_files:
        print(f"\nXử lý file: {file_path.name}")
        
        # Đọc dữ liệu
        df_original = load_nasdaq_data(file_path)
        if df_original is None:
            continue
        
        # Làm sạch dữ liệu
        df_processed = clean_nasdaq_data(df_original.copy())
        
        # Xử lý dữ liệu volume
        df_processed = process_volume_data(df_processed)
        
        # Tạo biểu đồ
        create_visualizations(df_original, df_processed, plots_dir, file_path.name)
        
        # Lưu dữ liệu đã xử lý
        output_file = processed_dir / f"processed_{file_path.name}"
        save_processed_data(df_processed, output_file)
        
        print(f"\nHoàn thành xử lý file: {file_path.name}")

if __name__ == "__main__":
    main()