
import pandas as pd

# Bước 1: Đọc dữ liệu từ các file CSV
cpi_data = pd.read_csv('data/vi_mo/CPI.csv', parse_dates=['observation_date'])
gdp_data = pd.read_csv('data/vi_mo/GDP.csv', parse_dates=['observation_date'])

# Bước 2: Loại bỏ outliers
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR))]

cpi_data = remove_outliers(cpi_data, 'CPIAUCSL')
gdp_data = remove_outliers(gdp_data, 'GDP')

# Bước 3: Backfill
cpi_data['CPIAUCSL'] = cpi_data['CPIAUCSL'].bfill()
gdp_data['GDP'] = gdp_data['GDP'].bfill()

# Bước 4: Chuẩn hóa dữ liệu
cpi_data['CPIAUCSL'] = (cpi_data['CPIAUCSL'] - cpi_data['CPIAUCSL'].mean()) / cpi_data['CPIAUCSL'].std()
gdp_data['GDP'] = (gdp_data['GDP'] - gdp_data['GDP'].mean()) / gdp_data['GDP'].std()

# Bước 5: Ghép dữ liệu
merged_data = pd.merge(cpi_data, gdp_data, on='observation_date', how='inner')

# Bước 6: Lưu dữ liệu đã xử lý
merged_data = merged_data[merged_data['observation_date'] >= '2004-01-01']
merged_data.to_csv('data/vi_mo/processed_data.csv', index=False)