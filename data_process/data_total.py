import pandas as pd
import os
from functools import reduce

def load_and_prepare_df(file_path, columns_to_use=None, columns_rename=None):
    """
    Load và chuẩn bị DataFrame từ file CSV
    """
    try:
        df = pd.read_csv(file_path)
        
        # Chuyển đổi cột thời gian thành datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Chọn các cột cụ thể nếu được chỉ định
        if columns_to_use:
            df = df[['Timestamp'] + columns_to_use]
            
        # Đổi tên cột nếu được chỉ định
        if columns_rename:
            df = df.rename(columns=columns_rename)
            
        return df
        
    except Exception as e:
        print(f"Lỗi khi đọc {file_path}: {str(e)}")
        return None

def merge_financial_data():
    # Kiểm tra và tạo thư mục data nếu chưa tồn tại
    base_path = os.path.join(os.getcwd(), "data", "processed_data")
    if not os.path.exists(base_path):
        print(f"Tạo thư mục: {base_path}")
        os.makedirs(base_path)
    
    # Định nghĩa cấu hình cho từng nguồn dữ liệu
    configs = {
        'price': {
            'file': 'processed_AAPL_daily.csv',
            'columns': ['close', 'open', 'high', 'low', 'volume'],
            'rename': {
                'close': 'AAPL_Close',
                'open': 'AAPL_Open', 
                'high': 'AAPL_High',
                'low': 'AAPL_Low',
                'volume': 'AAPL_Volume'
            }
        },
        'rsi': {
            'file': 'processed_AAPL_rsi.csv',
            'columns': ['RSI', 'RSI_normalized'],
            'rename': None
        },
        'bbands': {
            'file': 'processed_AAPL_BBANDS.csv',
            'columns': ['Real Upper Band', 'Real Middle Band', 'Real Lower Band'],
            'rename': {
                'Real Upper Band': 'BB_Upper',
                'Real Middle Band': 'BB_Middle',
                'Real Lower Band': 'BB_Lower'
            }
        },
        'volatility': {
            'file': 'processed_AAPL_volatility.csv', 
            'columns': ['atr'],
            'rename': {'atr': 'ATR'}
        },
        'market': {
            'file': 'processed_NDAQ_daily.csv',
            'columns': ['close', 'volume', 'daily_return', 'volatility', 'MA5', 'MA20'],
            'rename': {
                'close': 'NDAQ_Close',
                'volume': 'NDAQ_Volume',
                'daily_return': 'NDAQ_Return',
                'volatility': 'NDAQ_Volatility',
                'MA5': 'NDAQ_MA5',
                'MA20': 'NDAQ_MA20'
            }
        },
        'macro': {
            'file': 'processed_data_vi_mo.csv',
            'columns': ['CPIAUCSL', 'GDP'],
            'rename': {
                'CPIAUCSL': 'CPI',
                'GDP': 'GDP'
            }
        }
    }
    
    # Load tất cả DataFrames
    print("Đang đọc files dữ liệu...")
    dfs = {}
    for key, config in configs.items():
        file_path = os.path.join(base_path, config['file'])
        print(f"Xử lý {key} từ {file_path}")
        
        if not os.path.exists(file_path):
            print(f"File không tồn tại: {file_path}")
            continue
            
        df = load_and_prepare_df(
            file_path,
            columns_to_use=config['columns'],
            columns_rename=config['rename']
        )
        
        if df is not None:
            if 'Timestamp' in df.columns:
                df = df[df['Timestamp'] >= '2004-01-01']
            dfs[key] = df
            print(f"Đã load thành công {key} với kích thước {df.shape}")
    
    if not dfs:
        print("Không có dữ liệu nào được load thành công!")
        return None
        
    # Merge tất cả DataFrames
    print("\nĐang merge các datasets...")
    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on='Timestamp', how='outer'),
        dfs.values()
    )
    
    # Sắp xếp theo thời gian và xử lý missing values
    print("Xử lý dữ liệu đã merge...")
    merged_df = merged_df.sort_values('Timestamp')
    
    # Forward fill dữ liệu vĩ mô (giữ nguyên giá trị trong quý)
    macro_cols = ['CPI', 'GDP']
    merged_df[macro_cols] = merged_df[macro_cols].fillna(method='ffill')
    
    # Interpolate cho các dữ liệu khác
    non_macro_cols = [col for col in merged_df.columns if col not in ['Timestamp'] + macro_cols]
    merged_df[non_macro_cols] = merged_df[non_macro_cols].interpolate(method='time')
    
    # Tính toán các features bổ sung
    print("Tính toán features bổ sung...")
    
    try:
        # Tỷ suất sinh lời của AAPL
        merged_df['AAPL_Return'] = merged_df['AAPL_Close'].pct_change()
        
        # Độ biến động (20 ngày rolling std của returns)
        merged_df['AAPL_Volatility_20d'] = merged_df['AAPL_Return'].rolling(window=20).std()
        
        # Tỷ lệ giá so với SMA
        merged_df['AAPL_Price_to_NDAQ_MA20'] = merged_df['AAPL_Close'] / merged_df['NDAQ_MA20']
        merged_df['NDAQ_Price_to_MA20'] = merged_df['NDAQ_Close'] / merged_df['NDAQ_MA20']
        
        # Tỷ lệ volume AAPL/NDAQ
        merged_df['Volume_Ratio'] = merged_df['AAPL_Volume'] / merged_df['NDAQ_Volume']
    except Exception as e:
        print(f"Lỗi khi tính toán features: {str(e)}")
    
    # Xử lý missing values còn lại
    print("Xử lý missing values cuối cùng...")
    merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')
    
    # Lưu kết quả
    output_path = os.path.join(base_path, 'merged_financial_data.csv')
    merged_df.to_csv(output_path, index=False)
    
    print(f"\nKích thước dataset cuối cùng: {merged_df.shape}")
    print(f"Phạm vi thời gian: {merged_df['Timestamp'].min()} đến {merged_df['Timestamp'].max()}")
    print(f"\nCác cột trong dataset cuối:")
    for col in merged_df.columns:
        print(f"- {col}")
    print(f"\nĐã lưu dataset tại: {output_path}")
    
    return merged_df

if __name__ == "__main__":
    merged_data = merge_financial_data()