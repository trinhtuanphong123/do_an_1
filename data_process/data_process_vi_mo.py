import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class MacroDataProcessor:
    def __init__(self):
        """Khởi tạo các thư mục và tham số"""
        self.data_dir = Path("data/vi_mo")
        self.processed_dir = Path("data/processed_data")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.min_year = 2004
        
    def load_data(self):
        """Load dữ liệu CPI và GDP"""
        try:
            # Đọc dữ liệu
            cpi_df = pd.read_csv(self.data_dir / "CPI.csv")
            gdp_df = pd.read_csv(self.data_dir / "GDP.csv")
            
            # Chuyển đổi cột thời gian
            cpi_df['timestamp'] = pd.to_datetime(cpi_df['observation_date'])
            gdp_df['timestamp'] = pd.to_datetime(gdp_df['observation_date'])
            
            # Đặt timestamp làm index
            cpi_df.set_index('timestamp', inplace=True)
            gdp_df.set_index('timestamp', inplace=True)
            
            # Lọc dữ liệu từ 2004
            cpi_df = cpi_df[cpi_df.index.year >= self.min_year]
            gdp_df = gdp_df[gdp_df.index.year >= self.min_year]
            
            return cpi_df, gdp_df
        
        except Exception as e:
            print(f"Lỗi khi đọc dữ liệu: {str(e)}")
            return None, None

    def interpolate_daily_data(self, df, column_name):
        """Nội suy dữ liệu theo ngày"""
        # Tạo index daily mới
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        
        # Tạo DataFrame mới với index daily
        daily_df = pd.DataFrame(index=date_range)
        daily_df[column_name] = df[column_name]
        
        # Nội suy các giá trị thiếu
        daily_df = daily_df.interpolate(method='cubic')
        
        return daily_df

    def remove_outliers(self, df, column, threshold=3):
        """Loại bỏ outliers using IQR method"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        return df

    def normalize_data(self, df):
        """Chuẩn hóa dữ liệu về khoảng [0,1]"""
        scaler = MinMaxScaler()
        df_normalized = pd.DataFrame(
            scaler.fit_transform(df),
            columns=df.columns,
            index=df.index
        )
        return df_normalized

    def process_data(self):
        """Xử lý toàn bộ dữ liệu"""
        # Load dữ liệu
        print("Loading data...")
        cpi_df, gdp_df = self.load_data()
        
        if cpi_df is None or gdp_df is None:
            return
        
        # Nội suy dữ liệu theo ngày
        print("\nInterpolating daily data...")
        cpi_daily = self.interpolate_daily_data(cpi_df, 'CPIAUCSL')
        gdp_daily = self.interpolate_daily_data(gdp_df, 'GDP')
        
        # Loại bỏ outliers
        print("\nRemoving outliers...")
        cpi_daily = self.remove_outliers(cpi_daily, 'CPIAUCSL')
        gdp_daily = self.remove_outliers(gdp_daily, 'GDP')
        
        # Chuẩn hóa dữ liệu
        print("\nNormalizing data...")
        cpi_normalized = self.normalize_data(cpi_daily)
        gdp_normalized = self.normalize_data(gdp_daily)
        
        # Merge dữ liệu
        print("\nMerging data...")
        merged_data = pd.concat([
            cpi_normalized.rename(columns={'CPIAUCSL': 'CPI_normalized'}),
            gdp_normalized.rename(columns={'GDP': 'GDP_normalized'})
        ], axis=1)
        
        # Thêm dữ liệu gốc
        merged_data['CPI'] = cpi_daily['CPIAUCSL']
        merged_data['GDP'] = gdp_daily['GDP']
        
        # Reset index để có cột Timestamp
        merged_data = merged_data.reset_index()
        merged_data = merged_data.rename(columns={'index': 'Timestamp'})
        
        # Lưu dữ liệu đã xử lý
        output_path = self.processed_dir / 'processed_macro_daily.csv'
        merged_data.to_csv(output_path, index=False)
        print(f"\nĐã lưu dữ liệu đã xử lý vào: {output_path}")
        
        # Visualize data
        self.plot_data(merged_data)
        
        return merged_data

    def plot_data(self, df):
        """Vẽ biểu đồ dữ liệu"""
        plt.figure(figsize=(15, 10))
        
        # Plot CPI
        plt.subplot(2, 1, 1)
        plt.plot(df['Timestamp'], df['CPI_normalized'], label='CPI (normalized)')
        plt.title('Normalized CPI Data')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        # Plot GDP
        plt.subplot(2, 1, 2)
        plt.plot(df['Timestamp'], df['GDP_normalized'], label='GDP (normalized)', color='orange')
        plt.title('Normalized GDP Data')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    processor = MacroDataProcessor()
    processed_data = processor.process_data()
    print("\nData processing completed!")

if __name__ == "__main__":
    main()