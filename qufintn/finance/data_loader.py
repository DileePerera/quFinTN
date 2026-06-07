

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict


def load_portfolio_data(tickers: List[str], period: str = "4y") -> Dict:
    """
    Download and preprocess stock data with multiple fallback strategies.
    """
    print(f"Downloading data for {len(tickers)} assets...")

    try:
        # Strategy 1: Standard download with group_by
        data = yf.download(tickers, period=period, group_by='ticker', auto_adjust=True, progress=False)
        
        # Extract Close prices safely
        if isinstance(data.columns, pd.MultiIndex):
            # MultiIndex case (most common for multiple tickers)
            if 'Close' in data.columns.get_level_values(1):
                prices = data.xs('Close', axis=1, level=1)
            elif 'Adj Close' in data.columns.get_level_values(1):
                prices = data.xs('Adj Close', axis=1, level=1)
            else:
                # Fallback: take first level if needed
                prices = data.iloc[:, data.columns.get_level_values(1) == data.columns.get_level_values(1)[0]]
                prices = prices.droplevel(0, axis=1) if isinstance(prices.columns, pd.MultiIndex) else prices
        else:
            # Single ticker or flat columns
            prices = data['Close'] if 'Close' in data.columns else data['Adj Close']
        
        # Ensure DataFrame
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])
        
        # Clean data
        prices = prices.dropna(how='any')
        
        if prices.empty:
            raise ValueError("No data downloaded")
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Annualized metrics
        mu = returns.mean() * 252
        sigma = returns.cov() * 252
        
        print(f"Success! Period: {prices.index[0].date()} → {prices.index[-1].date()}")
        print(f"   Assets: {len(tickers)} | Days: {len(returns)}")
        
        return {
            'prices': prices,
            'returns': returns,
            'mu': mu,
            'sigma': sigma,
            'tickers': tickers
        }
        
    except Exception as e:
        print(f"⚠️ Error downloading data: {e}")
        print("Trying fallback method...")
        
        # Fallback: Download one by one
        prices_dict = {}
        for ticker in tickers:
            try:
                df = yf.download(ticker, period=period, progress=False)
                prices_dict[ticker] = df['Close'] if 'Close' in df.columns else df['Adj Close']
            except:
                print(f"   Failed to download {ticker}")
        
        prices = pd.DataFrame(prices_dict)
        prices = prices.dropna(how='any')
        returns = prices.pct_change().dropna()
        
        mu = returns.mean() * 252
        sigma = returns.cov() * 252
        
        print(f"Fallback success! Shape: {returns.shape}")
        
        return {
            'prices': prices,
            'returns': returns,
            'mu': mu,
            'sigma': sigma,
            'tickers': tickers
        }