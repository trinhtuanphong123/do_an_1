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
            
            # Đặt cột thời gian làm index
            df.set_index(date_column, inplace=True)
            
            # Chọn các cột cần thiết
            if isinstance(columns_to_use, str):
                columns_to_use = [columns_to_use]
            df = df[columns_to_use]
            
            # Lọc dữ liệu từ năm start_year
            df = df[df.index.year >= self.start_year]
            
            return df
            
        except Exception as e:
            print(f"Lỗi khi đọc file {file_path}: {str(e)}")
            return None

    def integrate_data(self):
        """
        Tổng hợp dữ liệu từ tất cả các file
        """
        # Dictionary chứa tên file và các cột cần lấy
        file_columns = {
            'processed_AAPL_daily.csv': ['close', 'volume'],
            'processed_AAPL_RSI.csv': ['RSI_normalized'],
            'processed_AAPL_BBANDS.csv': ['Real Middle Band '],
            'processed_AAPL_earnings_daily.csv': ['reportedEPS'],
            'processed_AAPL_ATR_daily.csv': ['atr'],
            'processed_NASDAQ_daily.csv': ['volume_nasdaq_normalized', 'close_nasdaq_normalized'],
            'processed_AAPL_obv.csv': ['obv_normalized'],
            'processed_AAPL_vi_mo.csv': ['CPI_normalized', 'GDP_normalized']
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
            merged_df = merged_df.join(df, how='inner')

        # Kiểm tra và xử lý dữ liệu thiếu
        missing_count = merged_df.isnull().sum()
        if missing_count.any():
            print("\nSố lượng giá trị thiếu trong mỗi cột:")
            print(missing_count)
            
            # Loại bỏ các dòng có giá trị thiếu
            merged_df = merged_df.dropna()
            print(f"\nSố dòng sau khi loại bỏ giá trị thiếu: {len(merged_df)}")

        # Sắp xếp theo thời gian
        merged_df.sort_index(inplace=True)

        # Reset index để đưa Timestamp thành cột
        merged_df.reset_index(inplace=True)
        
        # Đổi tên cột cho dễ hiểu
        merged_df = merged_df.rename(columns={
            'index': 'Timestamp',
            'Middle Band': 'bbands_middle',
            'close': 'price_close',
            'volume': 'price_volume'
        })

        # Lưu kết quả
        output_file = self.processed_dir / 'integrated_data.csv'
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

        return merged_df

def main():
    integrator = DataIntegrator(start_year=2004)
    integrated_data = integrator.integrate_data()

if __name__ == "__main__":
    main()