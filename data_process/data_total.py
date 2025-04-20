import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataIntegrator:
    def __init__(self, start_year=2004):
        """
        Khởi tạo class với thư mục dữ liệu và năm bắt đầu
        """
        self.start_year = start_year
        self.processed_dir = Path("data/processed_data")
        
    def load_and_prepare_file(self, file_path, columns_to_use, date_column='Timestamp'):
        """
        Load file và chuẩn bị dữ liệu với các cột cần thiết
        """
        try:
            df = pd.read_csv(file_path)
            
            # Chuyển đổi cột thời gian
            df[date_column] = pd.to_datetime(df[date_column])
            
            # Chọn các cột cần thiết
            if isinstance(columns_to_use, str):
                columns_to_use = [columns_to_use]
            df = df[[date_column] + columns_to_use]
            
            # Lọc dữ liệu từ năm start_year
            df = df[df[date_column].dt.year >= self.start_year]
            
            # Sắp xếp theo thời gian và loại bỏ các dòng trùng lặp
            df = df.sort_values(date_column)
            df = df.drop_duplicates(subset=[date_column], keep='first')
            
            # Đặt cột thời gian làm index
            df.set_index(date_column, inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Lỗi khi đọc file {file_path}: {str(e)}")
            return None

    def backfill_timeseries(self, df):
        """
        Backfill dữ liệu theo phương pháp phù hợp cho dữ liệu tài chính
        """
        try:
            # Tạo index date range đầy đủ
            full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
            
            # Reindex với đầy đủ các ngày
            df = df.reindex(full_range)
            
            # Forward fill trước (vì dữ liệu tài chính thường giữ giá trị của ngày giao dịch gần nhất)
            df = df.fillna(method='ffill', limit=5)  # Giới hạn forward fill tối đa 5 ngày
            
            # Backward fill sau cho các giá trị đầu tiên nếu có
            df = df.fillna(method='bfill', limit=5)  # Giới hạn backward fill tối đa 5 ngày
            
            # Nếu còn giá trị NaN, interpolate tuyến tính
            df = df.interpolate(method='linear', limit=5)
            
            return df
            
        except Exception as e:
            print(f"Lỗi trong quá trình backfill: {str(e)}")
            return df

    def add_weekday(self, df):
        """
        Thêm cột weekday cho biết thứ trong tuần
        """
        # Tạo map cho các thứ trong tuần
        weekday_map = {
            0: 'Monday',
            1: 'Tuesday', 
            2: 'Wednesday',
            3: 'Thursday',
            4: 'Friday',
            5: 'Saturday',
            6: 'Sunday'
        }
        
        # Thêm cột weekday
        df['weekday'] = df.index.weekday.map(weekday_map)
        
        return df

    def integrate_data(self):
        """
        Tổng hợp dữ liệu từ tất cả các file
        """
        # Dictionary chứa tên file và các cột cần lấy
        file_columns = {
            'processed_AAPL_daily.csv': ['close', 'volume'],
            'processed_AAPL_rsi.csv': ['RSI_normalized'],
            'processed_AAPL_BBANDS.csv': ['Real Middle Band'],
            'processed_AAPL_earnings_daily.csv': ['reportedEPS'],
            'processed_AAPL_ATR_daily.csv': ['atr'],
            'processed_NASDAQ_daily.csv': ['volume_nasdaq_normalized', 'close_nasdaq_normalized'],
            'processed_AAPL_obv.csv': ['obv_normalized'],
            'processed_vi_mo.csv': ['CPI_normalized', 'GDP_normalized']
        }

        all_dfs = []
        
        # Load tất cả các file
        for file_name, columns in file_columns.items():
            file_path = self.processed_dir / file_name
            if file_path.exists():
                df = self.load_and_prepare_file(file_path, columns)
                if df is not None:
                    all_dfs.append(df)
                    print(f"Đã load file {file_name}: {len(df)} dòng")
            else:
                print(f"Không tìm thấy file {file_name}")

        if not all_dfs:
            print("Không có dữ liệu để tổng hợp")
            return None

        # Merge tất cả DataFrame
        print("\nĐang tổng hợp dữ liệu...")
        merged_df = all_dfs[0]
        for df in all_dfs[1:]:
            merged_df = merged_df.join(df, how='outer')

        # Backfill dữ liệu thiếu
        print("\nĐang backfill dữ liệu thiếu...")
        merged_df = self.backfill_timeseries(merged_df)

        # Thêm cột weekday
        print("\nThêm thông tin weekday...")
        merged_df = self.add_weekday(merged_df)

        # Reset index để đưa Timestamp thành cột
        merged_df.reset_index(inplace=True)
        
        # Đổi tên cột cho dễ hiểu
        merged_df = merged_df.rename(columns={
            'index': 'Timestamp',
            'Real Middle Band': 'bbands_middle',
            'close': 'price_close',
            'volume': 'price_volume'
        })

        # Lọc chỉ giữ lại các ngày giao dịch (thứ 2 đến thứ 6)
        merged_df = merged_df[~merged_df['weekday'].isin(['Saturday', 'Sunday'])]

        # Lưu kết quả
        output_file = self.processed_dir / 'integrated_data_with_weekday.csv'
        merged_df.to_csv(output_file, index=False)
        print(f"\nĐã lưu dữ liệu tổng hợp vào: {output_file}")
        
        # In thông tin về dữ liệu tổng hợp
        print("\nThông tin dữ liệu tổng hợp:")
        print(f"Số lượng dòng: {len(merged_df)}")
        print(f"Số lượng cột: {len(merged_df.columns)}")
        print(f"Khoảng thời gian: {merged_df['Timestamp'].min()} đến {merged_df['Timestamp'].max()}")
        print("\nCác cột trong dữ liệu tổng hợp:")
        for col in merged_df.columns:
            print(f"- {col}")
        
        # In thông tin về phân bố các ngày trong tuần
        print("\nPhân bố các ngày trong tuần:")
        print(merged_df['weekday'].value_counts())

        return merged_df

def main():
    integrator = DataIntegrator(start_year=2004)
    integrated_data = integrator.integrate_data()
    
    # In thêm một số thống kê về dữ liệu
    if integrated_data is not None:
        print("\nThống kê mô tả cho các cột số:")
        numeric_cols = integrated_data.select_dtypes(include=[np.number]).columns
        print(integrated_data[numeric_cols].describe())

if __name__ == "__main__":
    main()