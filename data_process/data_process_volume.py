import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class OBVDataProcessor:
    def __init__(self, start_year=2004):
        """
        Khởi tạo processor với các thư mục cần thiết
        """
        self.start_year = start_year
        self.data_dir = Path("data/volume_data")
        self.processed_dir = Path("data/processed_data")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style cho matplotlib
        plt.style.use('default')  # Sử dụng style mặc định thay vì seaborn
        sns.set_theme(style="whitegrid")  # Sử dụng seaborn theme

    def load_data(self, file_path):
        """
        Load dữ liệu OBV và chỉ giữ lại các cột cần thiết
        """
        try:
            df = pd.read_csv(file_path)
            
            # Chuyển đổi timestamp
            df['Timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Chỉ giữ lại cột Timestamp và OBV
            df = df[['Timestamp', 'obv']]
            
            # Lọc dữ liệu từ năm start_year
            df = df[df['Timestamp'].dt.year >= self.start_year]
            
            # Sắp xếp theo thời gian
            df = df.sort_values('Timestamp')
            
            return df
        except Exception as e:
            print(f"Lỗi khi đọc file: {str(e)}")
            return None

    def remove_outliers(self, df):
        """
        Loại bỏ outliers sử dụng phương pháp IQR
        """
        df_clean = df.copy()
        
        # Tính toán thống kê rolling
        rolling_q1 = df['obv'].rolling(window=20, min_periods=1).quantile(0.25)
        rolling_q3 = df['obv'].rolling(window=20, min_periods=1).quantile(0.75)
        rolling_iqr = rolling_q3 - rolling_q1
        rolling_median = df['obv'].rolling(window=20, min_periods=1).median()
        
        # Xác định outliers
        lower_bound = rolling_q1 - 1.5 * rolling_iqr
        upper_bound = rolling_q3 + 1.5 * rolling_iqr
        
        # Thay thế outliers bằng giá trị trung vị
        outlier_mask = (df['obv'] < lower_bound) | (df['obv'] > upper_bound)
        df_clean.loc[outlier_mask, 'obv'] = rolling_median[outlier_mask]
        
        return df_clean

    def normalize_data(self, df):
        """
        Chuẩn hóa dữ liệu OBV về khoảng [0,1]
        """
        scaler = MinMaxScaler()
        df['obv_normalized'] = scaler.fit_transform(df[['obv']])
        return df

    def plot_analysis(self, df, symbol):
        """
        Tạo các biểu đồ phân tích
        """
        # 1. Time Series Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        ax1.plot(df['Timestamp'], df['obv'], label='Original OBV', color='blue', alpha=0.7)
        ax1.set_title(f'OBV Time Series for {symbol}')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('OBV Value')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(df['Timestamp'], df['obv_normalized'], label='Normalized OBV', 
                color='orange')
        ax2.set_title('Normalized OBV')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Normalized Value')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 2. Distribution Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.histplot(data=df, x='obv', kde=True, ax=ax1)
        ax1.set_title('OBV Distribution')
        
        sns.histplot(data=df, x='obv_normalized', kde=True, ax=ax2)
        ax2.set_title('Normalized OBV Distribution')
        
        plt.tight_layout()
        plt.show()
        
        # 3. Rolling Statistics
        window = 20
        rolling_mean = df['obv'].rolling(window=window).mean()
        rolling_std = df['obv'].rolling(window=window).std()
        
        plt.figure(figsize=(15, 7))
        plt.plot(df['Timestamp'], df['obv'], label='OBV', alpha=0.5)
        plt.plot(df['Timestamp'], rolling_mean, label=f'{window}-day MA', 
                color='red', linewidth=2)
        plt.fill_between(df['Timestamp'], 
                        rolling_mean - 2*rolling_std,
                        rolling_mean + 2*rolling_std,
                        color='gray', alpha=0.2, label='±2σ Band')
        plt.title(f'OBV with Rolling Statistics ({window}-day window)')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()

    def process_data(self):
        """
        Xử lý tất cả các file OBV trong thư mục
        """
        try:
            for file_path in self.data_dir.glob('*_OBV_*.csv'):
                # Lấy symbol từ tên file
                symbol = file_path.stem.split('_')[0]
                print(f"\nXử lý dữ liệu OBV cho {symbol}...")
                
                # Load dữ liệu
                df = self.load_data(file_path)
                if df is None:
                    continue
                
                print(f"Đã tải dữ liệu: {len(df)} dòng")
                
                # Loại bỏ outliers
                df_clean = self.remove_outliers(df)
                print("Đã xử lý outliers")
                
                # Chuẩn hóa dữ liệu
                df_normalized = self.normalize_data(df_clean)
                print("Đã chuẩn hóa dữ liệu")
                
                # Lưu dữ liệu đã xử lý
                output_file = self.processed_dir / f"{symbol}_OBV_processed.csv"
                df_normalized.to_csv(output_file, index=False)
                print(f"Đã lưu dữ liệu vào: {output_file}")
                
                # Tạo biểu đồ phân tích
                print("Tạo biểu đồ phân tích...")
                self.plot_analysis(df_normalized, symbol)
                
        except Exception as e:
            print(f"Lỗi trong quá trình xử lý: {str(e)}")

def main():
    processor = OBVDataProcessor(start_year=2004)
    processor.process_data()

if __name__ == "__main__":
    main()