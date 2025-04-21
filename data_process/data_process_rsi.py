import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import matplotlib
matplotlib.use('TkAgg')  # Sử dụng backend TkAgg để hiển thị biểu đồ

class RSIDataProcessor:
    def __init__(self):
        """Khởi tạo các thư mục cần thiết"""
        self.rsi_dir = Path("data/technical_data/rsi")
        self.rsi_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir = Path("data/processed_data")
        self.processed_dir.mkdir(exist_ok=True)
        self.min_year = 2004  # Chỉ lấy dữ liệu từ năm 2004 trở đi
        self.plots_dir = Path("data/visualizations")
        
    def load_rsi_data(self, file_path):
        """
        Đọc dữ liệu RSI từ file CSV
        Args:
            file_path: đường dẫn đến file dữ liệu
        Returns:
            DataFrame: dữ liệu RSI đã được đọc
        """
        try:
            df = pd.read_csv(file_path)
            
            # Kiểm tra và chuyển đổi cột thời gian
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                date_col = date_columns[0]
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
            
            print(f"\nĐã đọc thành công file RSI: {file_path}")
            print(f"Số lượng dòng dữ liệu: {len(df)}")
            
            return df
        except Exception as e:
            print(f"\nLỗi khi đọc file RSI: {str(e)}")
            return None
    
    def clean_data(self, df):
        """
        Làm sạch dữ liệu RSI
        Args:
            df: DataFrame dữ liệu RSI
        Returns:
            DataFrame: dữ liệu đã được làm sạch
        """
        if df is None or df.empty:
            return None
        
        # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
        cleaned_df = df.copy()
        
        # Đảm bảo index là DatetimeIndex
        if not isinstance(cleaned_df.index, pd.DatetimeIndex):
            # Kiểm tra xem có cột thời gian nào không
            date_columns = [col for col in cleaned_df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                date_col = date_columns[0]
                cleaned_df[date_col] = pd.to_datetime(cleaned_df[date_col])
                cleaned_df.set_index(date_col, inplace=True)
            
        # Lọc dữ liệu từ năm 2004 trở đi
        if isinstance(cleaned_df.index, pd.DatetimeIndex):
            cleaned_df = cleaned_df[cleaned_df.index.year >= self.min_year]
        
        # Phân tích dữ liệu thiếu trước khi xử lý
        missing_data = cleaned_df.isnull().sum()
        missing_percent = (missing_data / len(cleaned_df)) * 100
        print("\nPhân tích dữ liệu thiếu:")
        for col, count in missing_data.items():
            if count > 0:
                print(f"Cột {col}: {count} giá trị thiếu ({missing_percent[col]:.2f}%)")
        
        # Xử lý dữ liệu thiếu theo từng cột
        for col in cleaned_df.columns:
            # Nếu tỷ lệ dữ liệu thiếu > 30%, xóa cột
            if missing_percent[col] > 30:
                print(f"Xóa cột {col} do có quá nhiều dữ liệu thiếu ({missing_percent[col]:.2f}%)")
                cleaned_df = cleaned_df.drop(columns=[col])
                continue
                
            # Nếu là dữ liệu số, sử dụng phương pháp interpolation
            if cleaned_df[col].dtype in [np.float64, np.int64]:
                try:
                    # Sử dụng phương pháp interpolation để điền giá trị thiếu
                    if isinstance(cleaned_df.index, pd.DatetimeIndex):
                        cleaned_df[col] = cleaned_df[col].interpolate(method='time')
                    else:
                        cleaned_df[col] = cleaned_df[col].interpolate(method='linear')
                    
                    # Nếu vẫn còn giá trị thiếu (ở đầu hoặc cuối chuỗi), sử dụng forward/backward fill
                    if cleaned_df[col].isnull().sum() > 0:
                        cleaned_df[col] = cleaned_df[col].fillna(method='ffill').fillna(method='bfill')
                except Exception as e:
                    print(f"Lỗi khi interpolate cột {col}: {str(e)}")
                    # Fallback to simple forward/backward fill
                    cleaned_df[col] = cleaned_df[col].fillna(method='ffill').fillna(method='bfill')
            else:
                # Với dữ liệu không phải số, sử dụng mode (giá trị xuất hiện nhiều nhất)
                mode_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None
                if mode_value is not None:
                    cleaned_df[col] = cleaned_df[col].fillna(mode_value)
                else:
                    # Nếu không có mode, sử dụng forward/backward fill
                    cleaned_df[col] = cleaned_df[col].fillna(method='ffill').fillna(method='bfill')
        
        # Xử lý outliers bằng phương pháp IQR
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Thay thế outliers bằng giá trị biên
            cleaned_df[col] = np.where(cleaned_df[col] < lower_bound, lower_bound, cleaned_df[col])
            cleaned_df[col] = np.where(cleaned_df[col] > upper_bound, upper_bound, cleaned_df[col])
        
        print(f"Dữ liệu sau khi làm sạch: {len(cleaned_df)} dòng")
        return cleaned_df
    
    
    
    def visualize_data(self, original_df, processed_df, symbol):
        """
    Vẽ biểu đồ phân tích dữ liệu RSI (chỉ hiển thị)
    Args:
        original_df: DataFrame dữ liệu gốc
        processed_df: DataFrame dữ liệu đã xử lý
        symbol: Mã cổ phiếu
    """
        if original_df is None or processed_df is None:
            print("Không có dữ liệu để vẽ biểu đồ")
            return
    
    # Tạo figure với 4 subplots
        fig = plt.figure(figsize=(15, 18))
        fig.suptitle(f'Phân tích RSI cho {symbol}', fontsize=16)
    
    # Tạo grid layout
        gs = fig.add_gridspec(4, 2)
    
    # 1. Biểu đồ dữ liệu gốc và sau xử lý
        ax1 = fig.add_subplot(gs[0, :])
        rsi_col = [col for col in original_df.columns if 'rsi' in col.lower()]
        if rsi_col:
            rsi_col = rsi_col[0]
            ax1.plot(original_df.index, original_df[rsi_col], label='RSI gốc', color='blue', alpha=0.7)
            ax1.plot(processed_df.index, processed_df[rsi_col], label='RSI sau xử lý', color='red')
            ax1.axhline(y=70, color='green', linestyle='--', alpha=0.7, label='Vùng quá mua (70)')
            ax1.axhline(y=30, color='orange', linestyle='--', alpha=0.7, label='Vùng quá bán (30)')
            ax1.set_title('So sánh RSI trước và sau xử lý')
            ax1.set_ylabel('Giá trị RSI')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
    
    # 2. Biểu đồ phân phối RSI
        ax2 = fig.add_subplot(gs[1, 0])
        if rsi_col:
            sns.histplot(original_df[rsi_col], kde=True, ax=ax2, color='blue', alpha=0.5, label='RSI gốc')
            sns.histplot(processed_df[rsi_col], kde=True, ax=ax2, color='red', alpha=0.5, label='RSI sau xử lý')
            ax2.set_title('Phân phối giá trị RSI')
            ax2.set_xlabel('Giá trị RSI')
            ax2.set_ylabel('Tần suất')
            ax2.legend()
    
    # 3. Biểu đồ boxplot so sánh RSI trước và sau xử lý
        ax3 = fig.add_subplot(gs[1, 1])
        if rsi_col:
            data_to_plot = pd.DataFrame({
                'RSI gốc': original_df[rsi_col],
                'RSI sau xử lý': processed_df[rsi_col]
        })
            sns.boxplot(data=data_to_plot, ax=ax3)
            ax3.set_title('Boxplot so sánh RSI trước và sau xử lý')
            ax3.set_ylabel('Giá trị RSI')
    
    # 4. Biểu đồ RSI theo thời gian (heatmap theo năm)
        ax4 = fig.add_subplot(gs[2, :])
        if isinstance(processed_df.index, pd.DatetimeIndex) and rsi_col:
            heatmap_df = processed_df.reset_index()
            heatmap_df['Year'] = heatmap_df.iloc[:, 0].dt.year
            heatmap_df['Month'] = heatmap_df.iloc[:, 0].dt.month
            pivot_df = heatmap_df.pivot_table(values=rsi_col, index='Year', columns='Month', aggfunc='mean')
            sns.heatmap(pivot_df, cmap='RdYlGn', ax=ax4, cbar_kws={'label': 'Giá trị RSI trung bình'})
            ax4.set_title('Heatmap RSI theo năm và tháng')
            ax4.set_xlabel('Tháng')
            ax4.set_ylabel('Năm')
    
    # 5. Biểu đồ phân tích xu hướng RSI theo thời gian
        ax5 = fig.add_subplot(gs[3, :])
        if isinstance(processed_df.index, pd.DatetimeIndex) and rsi_col:
            yearly_rsi = processed_df.groupby(processed_df.index.year)[rsi_col].mean()
            ax5.plot(yearly_rsi.index, yearly_rsi.values, marker='o', linestyle='-', linewidth=2, markersize=8)
            ax5.set_title('Xu hướng RSI trung bình theo năm')
            ax5.set_xlabel('Năm')
            ax5.set_ylabel('RSI trung bình')
            ax5.grid(True, alpha=0.3)
        
        # Thêm đường xu hướng
            z = np.polyfit(yearly_rsi.index, yearly_rsi.values, 1)
            p = np.poly1d(z)
            ax5.plot(yearly_rsi.index, p(yearly_rsi.index), "r--", alpha=0.8, 
                    label=f'Đường xu hướng (y={z[0]:.4f}x+{z[1]:.4f})')
            ax5.legend()
    
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    
    def process_all_files(self):
        """Xử lý tất cả các file RSI trong thư mục"""
    # Tìm tất cả các file CSV trong thư mục
        csv_files = list(self.rsi_dir.glob("*.csv"))

        if not csv_files:
            print("\nKhông tìm thấy file CSV nào trong thư mục data/technical_data/rsi")
            return

        for file_path in csv_files:
         # Lấy tên file và symbol
            file_name = file_path.name
            symbol = file_name.split('_')[0] if '_' in file_name else file_name.split('.')[0]
        
            print(f"\n{'='*50}")
            print(f"Đang xử lý dữ liệu RSI cho {symbol}...")
        
        # Đọc dữ liệu
            original_df = self.load_rsi_data(file_path)
            if original_df is None:
                continue
        
        # Làm sạch dữ liệu
            processed_df = self.clean_data(original_df)
            if processed_df is None:
                continue
        
        # Chuẩn hóa dữ liệu
           
        
        # Kiểm tra lại một lần nữa để đảm bảo chỉ có dữ liệu từ 2004
            if isinstance(processed_df.index, pd.DatetimeIndex):
                processed_df = processed_df[processed_df.index.year >= self.min_year]
        
        # Lưu dữ liệu đã xử lý
            output_path = self.processed_dir / f"{symbol}_rsi_processed.csv"
            processed_df.to_csv(output_path)
            print(f"Đã lưu dữ liệu đã xử lý từ năm {self.min_year} vào: {output_path}")
        
        # Kiểm tra và hiển thị khoảng thời gian của dữ liệu
            if isinstance(processed_df.index, pd.DatetimeIndex):
                start_year = processed_df.index.min().year
                end_year = processed_df.index.max().year
                print(f"Khoảng thời gian dữ liệu: {start_year} - {end_year}")
        
        # Vẽ biểu đồ phân tích
            self.visualize_data(original_df, processed_df, symbol)

def main():
    processor = RSIDataProcessor()
    processor.process_all_files()

if __name__ == "__main__":
    main()
