import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')

def load_and_process_data(start_date='2004-01-01'):
    """
    Load and process stock and market data
    
    Args:
        start_date: Starting date for data analysis (default: '2004-01-01')
    """
    processed_dir = Path("data/processed_data")
    
    try:
        # Load stock data
        stock_path = processed_dir / "processed_AAPL_daily.csv"
        market_path = processed_dir / "processed_NDAQ_daily.csv"
        
        if not all(p.exists() for p in [stock_path, market_path]):
            raise FileNotFoundError("Required data files not found")
        
        # Read data files
        stock_df = pd.read_csv(stock_path)
        market_df = pd.read_csv(market_path)
        
        # Process dates
        for df in [stock_df, market_df]:
            date_col = next((col for col in df.columns 
                           if any(x in col.lower() for x in ['date', 'time', 'timestamp'])), None)
            
            if date_col:
                df['Date'] = pd.to_datetime(df[date_col])
            else:
                df['Date'] = pd.to_datetime(df.iloc[:, 0])
            
            df.set_index('Date', inplace=True)
            if date_col:
                df.drop(columns=[date_col], errors='ignore', inplace=True)
        
        # Calculate returns
        def get_returns(df):
            close_col = next((col for col in df.columns 
                            if 'close' in col.lower()), df.columns[3])
            return df[close_col].pct_change()
        
        # Create combined DataFrame
        combined_df = pd.DataFrame({
            'stock_return': get_returns(stock_df),
            'market_return': get_returns(market_df)
        })
        
        # Additional processing
        combined_df = combined_df.dropna()
        combined_df = combined_df[combined_df.index >= start_date]
        combined_df['excess_return'] = combined_df['stock_return'] - combined_df['market_return']
        
        # Calculate rolling statistics
        combined_df['volatility'] = combined_df['stock_return'].rolling(window=20).std()
        combined_df['market_volatility'] = combined_df['market_return'].rolling(window=20).std()
        
        print(f"\nData loaded successfully:")
        print(f"Period: {combined_df.index.min()} to {combined_df.index.max()}")
        print(f"Number of observations: {len(combined_df)}")
        
        return combined_df
        
    except Exception as e:
        print(f"Error in data loading: {str(e)}")
        return 
    

def neutralize_returns(df, method='regression', window=20):
    """
    Neutralize stock returns using multiple methods
    
    Args:
        df: DataFrame with stock and market returns
        method: 'regression', 'residual', or 'both'
        window: Rolling window size
    """
    result_df = df.copy()
    
    try:
        # 1. Regression-based neutralization
        if method in ['regression', 'both']:
            print("Performing regression-based neutralization...")
            neutralized_returns = np.zeros(len(df))
            betas = np.zeros(len(df))
            alphas = np.zeros(len(df))
            
            for i in range(window, len(df)):
                # Get window data
                window_data = df.iloc[i-window:i]
                y = window_data['stock_return']
                X = sm.add_constant(window_data['market_return'])
                
                try:
                    # Fit model
                    model = sm.RLM(y, X).fit()
                    
                    # Store parameters using iloc
                    betas[i] = model.params.iloc[1]  # Beta is the second parameter
                    alphas[i] = model.params.iloc[0]  # Alpha is the first parameter
                    
                    # Calculate neutralized return for current observation
                    current_market = df['market_return'].iloc[i]
                    expected_return = alphas[i] + betas[i] * current_market
                    neutralized_returns[i] = df['stock_return'].iloc[i] - expected_return
                    
                except Exception as e:
                    print(f"Warning: Error in regression for window ending at index {i}: {str(e)}")
                    # Use previous values if available, otherwise use 0
                    if i > window:
                        betas[i] = betas[i-1]
                        alphas[i] = alphas[i-1]
                        neutralized_returns[i] = neutralized_returns[i-1]
            
            result_df['regression_neutral'] = neutralized_returns
            result_df['rolling_beta'] = betas
            result_df['rolling_alpha'] = alphas
        
        # 2. Residual-based neutralization
        if method in ['residual', 'both']:
            print("Performing residual-based neutralization...")
            for col in ['stock_return', 'market_return']:
                rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
                rolling_std = df[col].rolling(window=window, min_periods=1).std()
                # Handle zero standard deviation
                rolling_std = rolling_std.replace(0, np.nan)
                rolling_std = rolling_std.fillna(rolling_std.mean())
                result_df[f'{col}_zscore'] = ((df[col] - rolling_mean) / rolling_std)
            
            result_df['residual_neutral'] = (result_df['stock_return_zscore'] - 
                                           result_df['market_return_zscore'])
        
        # 3. Combined approach
        if method == 'both':
            print("Calculating combined neutralization...")
            result_df['combined_neutral'] = (0.5 * result_df['regression_neutral'] + 
                                           0.5 * result_df['residual_neutral'])
        
        # Remove NaN values from the rolling window
        result_df = result_df.iloc[window:]
        
        # Print neutralization statistics
        print("\nNeutralization Results:")
        print("-" * 50)
        
        original_corr = df['stock_return'].corr(df['market_return'])
        print(f"Original market correlation: {original_corr:.4f}")
        
        for col in ['regression_neutral', 'residual_neutral', 'combined_neutral']:
            if col in result_df.columns:
                new_corr = result_df[col].corr(result_df['market_return'])
                print(f"{col} market correlation: {new_corr:.4f}")
                
                # Calculate improvement
                corr_reduction = abs(original_corr) - abs(new_corr)
                print(f"{col} correlation reduction: {corr_reduction:.4f}")
        
        return result_df
        
    except Exception as e:
        print(f"Error in neutralization: {str(e)}")
        print("Debug information:")
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns available: {df.columns.tolist()}")
        print(f"First few rows of data:")
        print(df.head())
        return None
    


def create_visualizations(df, visualizations_dir):
    """Enhanced visualizations for neutralization analysis"""
    
    # Create visual_dir if it doesn't exist
    visual_dir = visualizations_dir / 'visuaizations'
    visual_dir.mkdir(exist_ok=True)
    
    # 1. Returns Distribution Comparison
    plt.figure(figsize=(15, 10))
    for col in ['stock_return', 'regression_neutral', 'residual_neutral', 'combined_neutral']:
        if col in df.columns:
            sns.kdeplot(data=df[col], label=col.replace('_', ' ').title())
    plt.title('Distribution of Returns - All Methods')
    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(visual_dir / 'returns_distribution.png')
    plt.close()
    
    # 2. Time Series Analysis
    plt.figure(figsize=(15, 10))
    last_days = 100
    
    for col in ['stock_return', 'regression_neutral', 'residual_neutral', 'combined_neutral']:
        if col in df.columns:
            plt.plot(df.index[-last_days:], df[col][-last_days:], 
                    label=col.replace('_', ' ').title(), alpha=0.7)
    
    plt.title(f'Returns Time Series (Last {last_days} Days)')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend()
    plt.xticks(rotation=45)
    plt.savefig(visual_dir / 'time_series.png')
    plt.close()
    
    # 3. Market Relationship Analysis
    if 'regression_neutral' in df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Market Relationship Analysis')
        
        # Original returns vs Market
        sns.regplot(data=df, x='market_return', y='stock_return', 
                   ax=axes[0,0], scatter_kws={'alpha':0.5})
        axes[0,0].set_title('Original Returns vs Market')
        
        # Regression neutral vs Market
        sns.regplot(data=df, x='market_return', y='regression_neutral', 
                   ax=axes[0,1], scatter_kws={'alpha':0.5})
        axes[0,1].set_title('Regression Neutral vs Market')
        
        if 'residual_neutral' in df.columns:
            sns.regplot(data=df, x='market_return', y='residual_neutral', 
                       ax=axes[1,0], scatter_kws={'alpha':0.5})
            axes[1,0].set_title('Residual Neutral vs Market')
        
        if 'combined_neutral' in df.columns:
            sns.regplot(data=df, x='market_return', y='combined_neutral', 
                       ax=axes[1,1], scatter_kws={'alpha':0.5})
            axes[1,1].set_title('Combined Neutral vs Market')
        
        plt.tight_layout()
        plt.savefig(visual_dir / 'market_relationship.png')
        plt.close()

def save_results(df, processed_dir):
    """
    Save the processed data and generate summary statistics
    """
    try:
        # Create directories if they don't exist
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        output_file = processed_dir / 'market_neutralized_returns.csv'
        df.to_csv(output_file)
        print(f"\nResults saved to: {output_file}")
        
        # Generate and save summary statistics
        stats_file = processed_dir / 'market_neutralization_stats.txt'
        with open(stats_file, 'w', encoding='utf-8') as f:
            # Basic statistics
            f.write("Summary Statistics:\n")
            f.write("-----------------\n")
            
            # Original returns statistics
            f.write("\nOriginal Returns:\n")
            f.write(df['stock_return'].describe().to_string())
            
            # Neutralized returns statistics for each method
            for method in ['regression_neutral', 'residual_neutral', 'combined_neutral']:
                if method in df.columns:
                    f.write(f"\n\n{method.replace('_', ' ').title()} Returns:\n")
                    f.write(df[method].describe().to_string())
            
            # Rolling statistics if available
            if 'rolling_beta' in df.columns and 'rolling_alpha' in df.columns:
                f.write("\n\nRolling Statistics:\n")
                f.write("-----------------\n")
                f.write("\nBeta Statistics:\n")
                f.write(df['rolling_beta'].describe().to_string())
                f.write("\n\nAlpha Statistics:\n")
                f.write(df['rolling_alpha'].describe().to_string())
            
            # Correlation analysis
            f.write("\n\nCorrelation Analysis:\n")
            f.write("--------------------\n")
            cols_to_correlate = ['stock_return', 'market_return']
            cols_to_correlate.extend([col for col in ['regression_neutral', 'residual_neutral', 'combined_neutral'] 
                                    if col in df.columns])
            correlation_matrix = df[cols_to_correlate].corr()
            f.write(correlation_matrix.to_string())
            
            # Time series information
            f.write("\n\nTime Series Information:\n")
            f.write("----------------------\n")
            f.write(f"Start Date: {df.index.min()}\n")
            f.write(f"End Date: {df.index.max()}\n")
            f.write(f"Number of Trading Days: {len(df)}\n")
            
            # Volatility analysis
            f.write("\nVolatility Analysis:\n")
            f.write("------------------\n")
            f.write(f"Original Returns Volatility: {df['stock_return'].std():.4f}\n")
            f.write(f"Market Returns Volatility: {df['market_return'].std():.4f}\n")
            
            for method in ['regression_neutral', 'residual_neutral', 'combined_neutral']:
                if method in df.columns:
                    f.write(f"{method.replace('_', ' ').title()} Volatility: {df[method].std():.4f}\n")
        
        print(f"Statistics saved to: {stats_file}")
        
    except PermissionError as e:
        print(f"\nError: Không có quyền ghi file. Hãy chắc chắn rằng các file không đang được mở.")
        print(f"Chi tiết lỗi: {str(e)}")
    except Exception as e:
        print(f"\nLỗi khi lưu kết quả: {str(e)}")
        print(f"Thử tạo file với tên khác...")
        try:
            alternative_file = processed_dir / f'market_neutralized_returns_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
            df.to_csv(alternative_file)
            print(f"Đã lưu kết quả vào file thay thế: {alternative_file}")
        except Exception as e2:
            print(f"Không thể lưu file thay thế: {str(e2)}")

def main():
    # Setup
    processed_dir = Path("data/processed_data")
    visualizations_dir = Path("data/visualizations")
    
    try:
        # Load and process data
        print("Loading and processing data...")
        df = load_and_process_data(start_date='2004-01-01')
        if df is None:
            print("Lỗi: Không thể tải dữ liệu")
            return
        
        # Perform neutralization using both methods
        print("\nPerforming market neutralization...")
        df = neutralize_returns(df, method='both', window=20)
        if df is None:
            print("Lỗi: Không thể thực hiện neutralization")
            return
        
        # Create visualizations
        print("\nCreating visualizations...")
        create_visualizations(df, visualizations_dir)
        
        # Save results
        print("\nSaving results...")
        save_results(df, processed_dir)
        
        print("\nXử lý hoàn tất!")
        
    except Exception as e:
        print(f"\nLỗi trong quá trình xử lý: {str(e)}")

if __name__ == "__main__":
    main()


    