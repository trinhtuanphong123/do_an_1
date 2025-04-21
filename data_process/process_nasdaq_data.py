import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class NASDAQDataProcessor:
    def __init__(self, start_year=2004):
        """
        Khởi tạo processor với các thư mục cần thiết
        """
        self.start_year = start_year
        self.data_dir = Path("data/market_data")
        self.processed_dir = Path("data/processed_data")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style cho matplotlib
        plt.style.use('default')
        sns.set_theme(style="whitegrid")

    def load_data(self, file_path):
        """
        Load dữ liệu và chọn các cột cần thiết
        """
        try:
            # Đọc dữ liệu
            df = pd.read_csv(file_path)
            
            # Chuyển đổi timestamp
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            # Chọn và đổi tên các cột cần thiết
            df_selected = df[['Timestamp', 'volume', 'close']]
            df_selected = df_selected.rename(columns={
                'volume': 'volume_nasdaq',
                'close': 'close_nasdaq'
            })
            
            # Lọc dữ liệu từ năm start_year
            df_selected = df_selected[df_selected['Timestamp'].dt.year >= self.start_year]
            
            # Sắp xếp theo thời gian
            df_selected = df_selected.sort_values('Timestamp')
            
            return df_selected
            
        except Exception as e:
            print(f"Lỗi khi đọc file: {str(e)}")
            return None

    def remove_outliers(self, df, columns):
        """
        Xử lý outliers sử dụng phương pháp IQR
        """
        df_clean = df.copy()
        
        for column in columns:
            # Tính toán thống kê rolling
            rolling_q1 = df[column].rolling(window=20, min_periods=1).quantile(0.25)
            rolling_q3 = df[column].rolling(window=20, min_periods=1).quantile(0.75)
            rolling_iqr = rolling_q3 - rolling_q1
            rolling_median = df[column].rolling(window=20, min_periods=1).median()
            
            # Xác định outliers
            lower_bound = rolling_q1 - 1.5 * rolling_iqr
            upper_bound = rolling_q3 + 1.5 * rolling_iqr
            
            # Thay thế outliers bằng giá trị trung vị
            outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            df_clean.loc[outlier_mask, column] = rolling_median[outlier_mask]
            
            print(f"Đã xử lý {outlier_mask.sum()} outliers trong cột {column}")
            
        return df_clean

    def handle_missing_data(self, df):
        """
        Xử lý dữ liệu thiếu bằng phương pháp backfill
        """
        # Sử dụng backfill trước
        df_filled = df.fillna(method='bfill')
        # Sau đó sử dụng forward fill cho các giá trị còn thiếu
        df_filled = df_filled.fillna(method='ffill')
        return df_filled

    def plot_analysis(self, df, save_dir=None):
        """
        Tạo các biểu đồ phân tích (phiên bản không chuẩn hóa)
        """
        # 1. Time Series Plot cho Volume và Close
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(df['Timestamp'], df['volume_nasdaq'], 
                label='Volume', color='blue')
        plt.title('NASDAQ Volume')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(df['Timestamp'], df['close_nasdaq'], 
                label='Close Price', color='red')
        plt.title('NASDAQ Close Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

        # 2. Distribution Plots
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(data=df, x='volume_nasdaq', kde=True)
        plt.title('Volume Distribution')
        
        plt.subplot(1, 2, 2)
        sns.histplot(data=df, x='close_nasdaq', kde=True)
        plt.title('Close Price Distribution')
        
        plt.tight_layout()
        plt.show()

        # 3. Box Plats chỉ hiển thị dữ liệu gốc
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        df.boxplot(column=['volume_nasdaq'])
        plt.title('Volume Distribution')
        
        plt.subplot(1, 2, 2)
        df.boxplot(column=['close_nasdaq'])
        plt.title('Close Price Distribution')
        
        plt.tight_layout()
        plt.show()

    def process_data(self):
        """
        Xử lý toàn bộ dữ liệu (phiên bản không chuẩn hóa)
        """
        # Load dữ liệu
        file_path = self.data_dir / 'NDAQ_daily.csv'
        df = self.load_data(file_path)
        
        if df is None:
            return
        
        print(f"Đã tải dữ liệu: {len(df)} dòng")
        
        # Xử lý dữ liệu thiếu
        df = self.handle_missing_data(df)
        print("Đã xử lý dữ liệu thiếu")
        
        # Xử lý outliers
        columns_to_process = ['volume_nasdaq', 'close_nasdaq']
        df_clean = self.remove_outliers(df, columns_to_process)
        print("Đã xử lý outliers")
        
        # Lưu dữ liệu đã xử lý
        output_file = self.processed_dir / 'processed_NASDAQ_daily.csv'
        df_clean.to_csv(output_file, index=True)
        print(f"Đã lưu dữ liệu vào: {output_file}")
        
        # Tạo biểu đồ phân tích
        print("Tạo biểu đồ phân tích...")
        self.plot_analysis(df_clean)
        
        return df_clean

def main():
    processor = NASDAQDataProcessor(start_year=2004)
    processed_data = processor.process_data()
    
if __name__ == "__main__":
    main()