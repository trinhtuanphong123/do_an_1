import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.interpolate import CubicSpline
import warnings
warnings.filterwarnings('ignore')

class EarningsDataProcessor:
    def __init__(self, start_year=2004):
        """
        Khởi tạo processor với các thư mục cần thiết
        """
        self.start_year = start_year
        self.data_dir = Path("data/financial_metrics/earnings")
        self.processed_dir = Path("data/processed_data")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style cho matplotlib
        plt.style.use('default')
        sns.set_theme(style="whitegrid")

    def load_data(self, file_path):
        """
        Load và tiền xử lý dữ liệu earnings
        """
        try:
            # Đọc dữ liệu
            df = pd.read_csv(file_path)
            
            # Chuyển đổi các cột thời gian
            df['Timestamp'] = pd.to_datetime(df['reportedDate'])
            df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            
            # Chọn các cột cần thiết
            df = df[['Timestamp', 'reportedEPS']]
            
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
        Xử lý outliers sử dụng phương pháp IQR với rolling window
        """
        df_clean = df.copy()
        
        # Tính toán thống kê rolling
        rolling_q1 = df['reportedEPS'].rolling(window=4, min_periods=1).quantile(0.25)
        rolling_q3 = df['reportedEPS'].rolling(window=4, min_periods=1).quantile(0.75)
        rolling_iqr = rolling_q3 - rolling_q1
        rolling_median = df['reportedEPS'].rolling(window=4, min_periods=1).median()
        
        # Xác định outliers
        lower_bound = rolling_q1 - 1.5 * rolling_iqr
        upper_bound = rolling_q3 + 1.5 * rolling_iqr
        
        # Thay thế outliers bằng giá trị trung vị
        outlier_mask = (df['reportedEPS'] < lower_bound) | (df['reportedEPS'] > upper_bound)
        df_clean.loc[outlier_mask, 'reportedEPS'] = rolling_median[outlier_mask]
        
        print(f"Đã xử lý {outlier_mask.sum()} outliers")
        return df_clean

    def interpolate_daily_data(self, df):
        """
        Nội suy dữ liệu theo ngày sử dụng cubic spline
        """
        # Tạo index daily mới
        date_range = pd.date_range(start=df['Timestamp'].min(), 
                                 end=df['Timestamp'].max(), 
                                 freq='D')
        
        # Tạo cubic spline interpolator
        cs = CubicSpline(df['Timestamp'].astype(np.int64) // 10**9, 
                        df['reportedEPS'])
        
        # Nội suy giá trị
        interpolated_values = cs(date_range.astype(np.int64) // 10**9)
        
        # Tạo DataFrame mới với dữ liệu đã nội suy
        df_daily = pd.DataFrame({
            'Timestamp': date_range,
            'reportedEPS': interpolated_values
        })
        
        return df_daily

    def plot_analysis(self, df_quarterly, df_daily):
        """
        Tạo các biểu đồ phân tích
        """
        # 1. Time Series Plot - Quarterly vs Daily
        plt.figure(figsize=(15, 7))
        plt.plot(df_daily['Timestamp'], df_daily['reportedEPS'], 
                label='Daily (Interpolated)', alpha=0.6)
        plt.scatter(df_quarterly['Timestamp'], df_quarterly['reportedEPS'], 
                   label='Quarterly', color='red')
        plt.title('Reported EPS Over Time')
        plt.xlabel('Date')
        plt.ylabel('EPS Value')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 2. Distribution Plot
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df_quarterly, x='reportedEPS', kde=True)
        plt.title('EPS Distribution (Quarterly)')
        plt.xlabel('EPS Value')
        plt.ylabel('Frequency')
        plt.show()

        # 3. Year-over-Year Analysis
        plt.figure(figsize=(15, 6))
        df_quarterly['Year'] = df_quarterly['Timestamp'].dt.year
        sns.boxplot(data=df_quarterly, x='Year', y='reportedEPS')
        plt.title('EPS Distribution by Year')
        plt.xlabel('Year')
        plt.ylabel('EPS Value')
        plt.xticks(rotation=45)
        plt.show()

        # 4. Quarterly Trends
        plt.figure(figsize=(15, 6))
        df_quarterly['Quarter'] = df_quarterly['Timestamp'].dt.quarter
        sns.boxplot(data=df_quarterly, x='Quarter', y='reportedEPS')
        plt.title('EPS Distribution by Quarter')
        plt.xlabel('Quarter')
        plt.ylabel('EPS Value')
        plt.show()

    def process_data(self):
        """
        Xử lý toàn bộ dữ liệu earnings
        """
        # Load dữ liệu
        file_path = self.data_dir / 'AAPL_quarterly_earnings.csv'
        df = self.load_data(file_path)
        
        if df is None:
            return
        
        print(f"Đã tải dữ liệu: {len(df)} dòng")
        
        # Xử lý outliers
        df_clean = self.remove_outliers(df)
        print("Đã xử lý outliers")
        
        # Nội suy dữ liệu theo ngày
        df_daily = self.interpolate_daily_data(df_clean)
        print("Đã nội suy dữ liệu theo ngày")
        
        # Lưu dữ liệu đã xử lý
        output_file = self.processed_dir / 'processed_AAPL_earnings_daily.csv'
        df_daily.to_csv(output_file, index=False)
        print(f"Đã lưu dữ liệu vào: {output_file}")
        
        # Tạo biểu đồ phân tích
        print("Tạo biểu đồ phân tích...")
        self.plot_analysis(df_clean, df_daily)
        
        return df_daily

def main():
    processor = EarningsDataProcessor(start_year=2004)
    processed_data = processor.process_data()

if __name__ == "__main__":
    main()