import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from scipy import stats

def load_and_clean_data():
    # Read the data
    df = pd.read_csv('data/financial_metrics/earnings/AAPL_quarterly_earnings.csv')
    
    # Convert dates to datetime
    df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
    df['reportedDate'] = pd.to_datetime(df['reportedDate'])
    
    # Filter data from 2004 onwards
    df = df[df['fiscalDateEnding'].dt.year >= 2004].copy()
    
    # Sort by date
    df = df.sort_values('fiscalDateEnding')
    
    # Convert string values to numeric, handling any non-numeric values
    df['reportedEPS'] = pd.to_numeric(df['reportedEPS'], errors='coerce')
    df['estimatedEPS'] = pd.to_numeric(df['estimatedEPS'], errors='coerce')
    
    return df

def remove_outliers_and_fill(df):
    """
    Remove outliers using Z-score method and fill missing values
    """
    # Calculate z-scores for reportedEPS
    z_scores = stats.zscore(df['reportedEPS'], nan_policy='omit')
    
    # Define outliers as points with |z-score| > 3
    outliers = np.abs(z_scores) > 3
    
    # Create a copy of the dataframe for cleaned data
    df_cleaned = df.copy()
    
    # Replace outliers with NaN
    df_cleaned.loc[outliers, 'reportedEPS'] = np.nan
    
    # Forward fill and then backward fill to handle any remaining NaN values
    df_cleaned['reportedEPS'] = df_cleaned['reportedEPS'].ffill().bfill()
    df_cleaned['estimatedEPS'] = df_cleaned['estimatedEPS'].ffill().bfill()
    
    return df_cleaned

def generate_daily_data_linear_interpolation(quarterly_data):
    """
    Generate daily data using linear interpolation from quarterly data
    """
    # Convert dates to numerical values for interpolation
    date_nums = (quarterly_data['fiscalDateEnding'] - quarterly_data['fiscalDateEnding'].min()).dt.days
    
    # Create interpolation functions for actual and estimated EPS
    f_actual = interpolate.interp1d(date_nums, quarterly_data['reportedEPS'], kind='linear')
    f_estimated = interpolate.interp1d(date_nums, quarterly_data['estimatedEPS'], kind='linear')
    
    # Generate daily dates
    start_date = quarterly_data['fiscalDateEnding'].min()
    end_date = quarterly_data['fiscalDateEnding'].max()
    daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Convert daily dates to numerical values
    daily_nums = (daily_dates - start_date).days
    
    # Generate interpolated values
    daily_actual = f_actual(daily_nums)
    daily_estimated = f_estimated(daily_nums)
    
    # Create daily DataFrame
    daily_data = pd.DataFrame({
        'date': daily_dates,
        'daily_eps': daily_actual,
        'daily_estimated_eps': daily_estimated
    })
    
    return daily_data

def visualize_data(quarterly_data, daily_data):
    """
    Create visualizations for both quarterly and interpolated daily data
    """
    # Set the style to a simple, clean style
    plt.style.use('default')
    sns.set_style("whitegrid")
    
    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 18))
    
    # Plot 1: Original Quarterly Data
    ax1.plot(quarterly_data['fiscalDateEnding'], quarterly_data['reportedEPS'], 
             'o-', label='Original Reported EPS', markersize=8, alpha=0.5)
    ax1.plot(quarterly_data['fiscalDateEnding'], quarterly_data['estimatedEPS'], 
             'o--', label='Original Estimated EPS', markersize=8, alpha=0.5)
    ax1.set_title('AAPL Original Quarterly Earnings Per Share (2004+)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('EPS ($)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Cleaned Quarterly Data (after outlier removal)
    ax2.plot(quarterly_data['fiscalDateEnding'], quarterly_data['reportedEPS'], 
             'o-', label='Cleaned Reported EPS', markersize=8)
    ax2.plot(quarterly_data['fiscalDateEnding'], quarterly_data['estimatedEPS'], 
             'o--', label='Cleaned Estimated EPS', markersize=8)
    ax2.set_title('AAPL Cleaned Quarterly Earnings (After Outlier Removal)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('EPS ($)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Daily interpolated data
    ax3.plot(daily_data['date'], daily_data['daily_eps'], 
             label='Daily EPS (Interpolated)', linewidth=2)
    ax3.plot(daily_data['date'], daily_data['daily_estimated_eps'], 
             '--', label='Daily Estimated EPS (Interpolated)', linewidth=2)
    ax3.set_title('Daily Interpolated Earnings Per Share')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Daily EPS ($)')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('earnings_analysis_full.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load and process data
    print("Loading and cleaning data...")
    quarterly_data = load_and_clean_data()
    
    # Remove outliers and fill missing values
    print("Removing outliers and filling missing values...")
    quarterly_data_cleaned = remove_outliers_and_fill(quarterly_data)
    
    # Generate daily data using linear interpolation
    print("Generating daily data through linear interpolation...")
    daily_data = generate_daily_data_linear_interpolation(quarterly_data_cleaned)
    
    # Create visualizations
    print("Creating visualizations...")
    visualize_data(quarterly_data_cleaned, daily_data)
    
    # Save processed data
    print("Saving processed data...")
    quarterly_data_cleaned.to_csv('processed_quarterly_earnings.csv', index=False)
    daily_data.to_csv('interpolated_daily_earnings.csv', index=False)
    
    print("\nProcessing complete!")
    print("Files generated:")
    print("1. processed_quarterly_earnings.csv - Cleaned quarterly data")
    print("2. interpolated_daily_earnings.csv - Interpolated daily data")
    print("3. earnings_analysis_full.png - Visualization of all stages")

if __name__ == "__main__":
    main()