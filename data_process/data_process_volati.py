import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class ATRDataProcessor:
    def __init__(self, start_year=2004):
        """
        Khởi tạo processor với các thư mục cần thiết
        """
        self.start_year = start_year
        self.data_dir = Path("data/financial_metrics/volatility")
        self.processed_dir = Path("data/processed_data")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style cho matplotlib
        plt.style.use('default')
        sns.set_theme(style="whitegrid")

    def load_data(self, file_path):
        """
        Load và tiền xử lý dữ liệu ATR
        """
        try:
            # Đọc dữ liệu
            df = pd.read_csv(file_path)
            
            # Chuyển đổi timestamp
            df['Timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.drop('timestamp', axis=1)
            
            # Lọc dữ liệu từ năm start_year
            df = df[df['Timestamp'].dt.year >= self.start_year]
            
            # Sắp xếp theo thời gian từ cũ đến mới
            df = df.sort_values('Timestamp')
            
            return df
            
        except Exception as e:
            print(f"Lỗi khi đọc file: {str(e)}")
            return None

    def remove_outliers(self, df):
        """
        Xử lý outliers sử dụng phương pháp IQR với rolling window
        """
        df_clean = df.copy()
        
        # Tính toán thống kê rolling
        rolling_q1 = df['atr'].rolling(window=20, min_periods=1).quantile(0.25)
        rolling_q3 = df['atr'].rolling(window=20, min_periods=1).quantile(0.75)
        rolling_iqr = rolling_q3 - rolling_q1
        rolling_median = df['atr'].rolling(window=20, min_periods=1).median()
        
        # Xác định outliers
        lower_bound = rolling_q1 - 1.5 * rolling_iqr
        upper_bound = rolling_q3 + 1.5 * rolling_iqr
        
        # Thay thế outliers bằng giá trị trung vị
        outlier_mask = (df['atr'] < lower_bound) | (df['atr'] > upper_bound)
        df_clean.loc[outlier_mask, 'atr'] = rolling_median[outlier_mask]
        
        print(f"Đã xử lý {outlier_mask.sum()} outliers")
        return df_clean

    def handle_missing_data(self, df):
        """
        Xử lý dữ liệu thiếu bằng backfill và forward fill
        """
        # Sử dụng backfill trước
        df_filled = df.fillna(method='bfill')
        # Sau đó sử dụng forward fill cho các giá trị còn thiếu
        df_filled = df_filled.fillna(method='ffill')
        return df_filled

    def plot_analysis(self, df):
        """
        Tạo các biểu đồ phân tích
        """
        # 1. Time Series Plot
        plt.figure(figsize=(15, 7))
        plt.plot(df['Timestamp'], df['atr'], label='ATR')
        plt.title('Average True Range (ATR) Over Time')
        plt.xlabel('Date')
        plt.ylabel('ATR Value')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 2. Distribution Plot với KDE
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x='atr', kde=True)
        plt.title('ATR Distribution')
        plt.xlabel('ATR Value')
        plt.ylabel('Frequency')
        plt.show()

        # 3. Rolling Statistics
        window = 20
        rolling_mean = df['atr'].rolling(window=window).mean()
        rolling_std = df['atr'].rolling(window=window).std()
        
        plt.figure(figsize=(15, 7))
        plt.plot(df['Timestamp'], df['atr'], label='ATR', alpha=0.5)
        plt.plot(df['Timestamp'], rolling_mean, label=f'{window}-day MA', 
                color='red', linewidth=2)
        plt.fill_between(df['Timestamp'], 
                        rolling_mean - 2*rolling_std,
                        rolling_mean + 2*rolling_std,
                        color='gray', alpha=0.2, label='±2σ Band')
        plt.title(f'ATR with Rolling Statistics ({window}-day window)')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 4. Box Plot by Year
        plt.figure(figsize=(15, 6))
        df['Year'] = df['Timestamp'].dt.year
        sns.boxplot(data=df, x='Year', y='atr')
        plt.title('ATR Distribution by Year')
        plt.xlabel('Year')
        plt.ylabel('ATR Value')
        plt.xticks(rotation=45)
        plt.show()

    def process_data(self):
        """
        Xử lý toàn bộ dữ liệu ATR
        """
        # Load dữ liệu
        file_path = self.data_dir / 'AAPL_ATR_daily.csv'
        df = self.load_data(file_path)
        
        if df is None:
            return
        
        print(f"Đã tải dữ liệu: {len(df)} dòng")
        
        # Xử lý dữ liệu thiếu
        df = self.handle_missing_data(df)
        print("Đã xử lý dữ liệu thiếu")
        
        # Xử lý outliers
        df_clean = self.remove_outliers(df)
        print("Đã xử lý outliers")
        
        # Lưu dữ liệu đã xử lý
        output_file = self.processed_dir / 'processed_AAPL_ATR_daily.csv'
        df_clean.to_csv(output_file, index=False)
        print(f"Đã lưu dữ liệu vào: {output_file}")
        
        # Tạo biểu đồ phân tích
        print("Tạo biểu đồ phân tích...")
        self.plot_analysis(df_clean)
        
        return df_clean

def main():
    processor = ATRDataProcessor(start_year=2004)
    processed_data = processor.process_data()

if __name__ == "__main__":
    main()