#!/usr/bin/env python3
"""
Stock Predictor using Datadog Toto Model
Complete implementation ready to run
"""

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from loguru import logger
import warnings
import csv
import yfinance as yf
from typing import Dict, List, Optional, Tuple, Any

warnings.filterwarnings('ignore')

# Try to import Toto model from DataDog
try:
    import torch
    from toto.model.toto import Toto
    from toto.inference.forecaster import TotoForecaster
    from toto.data.util.dataset import MaskedTimeseries
    MODEL_AVAILABLE = True
    MODEL_TYPE = "toto"
    logger.info("Datadog Toto model available")
except ImportError:
    try:
        # Fallback to Chronos model
        from chronos import ChronosPipeline
        MODEL_AVAILABLE = True
        MODEL_TYPE = "chronos"
        logger.info("Chronos model available as fallback")
    except ImportError:
        logger.warning("No time series model available. Using simple predictions.")
        MODEL_AVAILABLE = False
        MODEL_TYPE = "none"


def download_data(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Download stock data using yfinance
    
    Args:
        symbol: Stock symbol (e.g., "AAPL", "SPY")
        period: Time period to download (default: 2 years)
        interval: Data interval (default: daily)
    
    Returns:
        DataFrame with stock data
    """
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            logger.warning(f"No data retrieved for {symbol}")
            return pd.DataFrame()
        
        # Ensure column names are properly formatted
        data.columns = [col.replace(' ', '') for col in data.columns]
        
        return data
    except Exception as e:
        logger.error(f"Error downloading data for {symbol}: {e}")
        return pd.DataFrame()


class StockPredictor:
    def __init__(self, model_name: str = "Datadog/Toto-Open-Base-1.0"):
        self.model = None
        self.forecaster = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler.fit_transform([[-1, 1]])  # Initialize scaler
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_pipeline(self):
        """Load the time series forecasting model"""
        if self.model is None and MODEL_AVAILABLE:
            try:
                if MODEL_TYPE == "toto":
                    # Load Datadog Toto model
                    self.model = Toto.from_pretrained(self.model_name)
                    self.model.to(self.device)
                    
                    # Optionally compile for faster inference
                    if torch.cuda.is_available():
                        try:
                            self.model.compile()
                            logger.info("Model compiled successfully")
                        except:
                            logger.info("Model compilation failed, continuing without compilation")
                    
                    self.forecaster = TotoForecaster(self.model.model)
                    logger.info(f"Loaded Datadog Toto model {self.model_name} successfully")
                    
                elif MODEL_TYPE == "chronos":
                    # Fallback to Chronos
                    self.model = ChronosPipeline.from_pretrained(
                        "amazon/chronos-t5-small",
                        device_map=str(self.device),
                        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    )
                    logger.info("Loaded Chronos model as fallback")
                    
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model = None
                self.forecaster = None
    
    def percent_movements_augment(self, data: np.ndarray) -> np.ndarray:
        """Convert price data to percentage changes"""
        if len(data) < 2:
            return np.array([])
        diff_data = np.diff(data.flatten())
        prev_data = data[:-1].flatten()
        # Avoid division by zero
        prev_data = np.where(prev_data == 0, 1e-8, prev_data)
        return diff_data / prev_data
    
    def preprocess_data(self, stock_data: pd.DataFrame, key_to_predict: str = "close") -> pd.DataFrame:
        """Preprocess stock data for prediction"""
        data = stock_data.copy()
        logger.info(f"Preprocessing data for {key_to_predict}")
        
        # Standardize column names to lowercase
        data.columns = data.columns.str.lower()
        
        # Ensure we have the required columns
        required_cols = ["open", "high", "low", "close"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            return pd.DataFrame()
        
        # Convert to percentage changes for each price column
        processed_data = pd.DataFrame()
        for col in required_cols:
            if col in data.columns and len(data[col]) > 1:
                pct_changes = self.percent_movements_augment(data[col].values.reshape(-1, 1))
                if len(pct_changes) > 0:
                    processed_data[col] = pct_changes
                else:
                    processed_data[col] = [0] * max(0, len(data) - 1)
        
        if processed_data.empty:
            logger.error("No processed data available")
            return pd.DataFrame()
        
        # Create target variable (next day's percentage change)
        processed_data['y'] = processed_data[key_to_predict].shift(-1)
        
        # Create trade weight (1 for positive moves, -1 for negative)
        processed_data['trade_weight'] = (processed_data['y'] > 0) * 2 - 1
        
        # Add time index and date columns
        processed_data['ds'] = pd.date_range(start="2020-01-01", periods=len(processed_data), freq="D")
        processed_data['id'] = processed_data.index
        processed_data['unique_id'] = 1
        
        # Drop the last row (no target available) and any NaN values
        processed_data = processed_data.dropna()
        
        return processed_data
    
    def make_single_prediction(self, context_values: np.ndarray) -> float:
        """Make a single prediction using the loaded model"""
        if len(context_values) == 0:
            return 0.0
        
        context = torch.tensor(context_values, dtype=torch.float)
        
        # Load model if not already loaded
        self.load_pipeline()
        
        if self.forecaster is not None and MODEL_TYPE == "toto":
            try:
                # Prepare data for Toto model
                input_series = context.unsqueeze(0).to(self.device)  # Add channel dimension
                
                # Create timestamp information
                timestamp_seconds = torch.zeros_like(input_series).to(self.device)
                time_interval_seconds = torch.full((1,), 86400).to(self.device)  # Daily intervals
                
                # Create MaskedTimeseries object
                inputs = MaskedTimeseries(
                    series=input_series,
                    padding_mask=torch.full_like(input_series, True, dtype=torch.bool),
                    id_mask=torch.zeros_like(input_series),
                    timestamp_seconds=timestamp_seconds,
                    time_interval_seconds=time_interval_seconds,
                )
                
                # Generate forecast
                forecast = self.forecaster.forecast(
                    inputs,
                    prediction_length=1,
                    num_samples=32,
                    samples_per_batch=32,
                )
                
                # Get median prediction
                prediction = forecast.mean[0, 0].item()
                
            except Exception as e:
                logger.warning(f"Toto prediction failed: {e}. Using simple forecast.")
                prediction = np.mean(context_values[-5:]) if len(context_values) >= 5 else 0.0
                
        elif self.model is not None and MODEL_TYPE == "chronos":
            try:
                # Make prediction using Chronos model
                forecast = self.model.predict(
                    context.unsqueeze(0),  # Add batch dimension
                    prediction_length=1
                )
                # Chronos returns quantiles, we take the median
                prediction = torch.median(forecast[0]).item()
            except Exception as e:
                logger.warning(f"Chronos prediction failed: {e}. Using simple forecast.")
                prediction = np.mean(context_values[-5:]) if len(context_values) >= 5 else 0.0
        else:
            # Simple momentum-based prediction if model not available
            recent_trend = np.mean(context_values[-5:]) if len(context_values) >= 5 else 0.0
            volatility = np.std(context_values[-10:]) if len(context_values) >= 10 else 0.01
            # Add some randomness based on recent volatility
            prediction = recent_trend + np.random.normal(0, volatility * 0.1)
        
        return prediction
    
    def calculate_trading_metrics(self, predictions: List[float], actuals: np.ndarray) -> Dict[str, float]:
        """Calculate trading performance metrics"""
        if len(predictions) == 0 or len(actuals) == 0:
            return {
                'mae': None,
                'total_return': 0,
                'win_rate': 0,
                'sharpe_ratio': 0
            }
        
        # Ensure same length
        min_len = min(len(predictions), len(actuals))
        predictions = predictions[:min_len]
        actuals = actuals[:min_len]
        
        # Calculate MAE
        mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
        
        # Trading signals (buy if predicted positive change)
        trading_signals = (np.array(predictions) > 0) * 2 - 1
        
        # Calculate returns
        returns = np.array(actuals) * trading_signals
        total_return = np.sum(returns)
        
        # Win rate
        winning_trades = np.sum(returns > 0)
        total_trades = len(returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Sharpe ratio (simplified)
        if len(returns) > 1:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'mae': float(mae),
            'total_return': float(total_return),
            'win_rate': float(win_rate),
            'sharpe_ratio': float(sharpe_ratio)
        }
    
    def make_prediction(self, symbol: str, lookback_days: int = 30, prediction_days: int = 7) -> Optional[Dict[str, Any]]:
        """
        Make predictions for a single stock symbol
        
        Args:
            symbol: Stock symbol (e.g., "AAPL", "SPY")
            lookback_days: Number of days to use as context
            prediction_days: Number of days to predict ahead
            
        Returns:
            Dictionary with predictions and analysis
        """
        logger.info(f"Making predictions for {symbol}")
        
        try:
            # Download stock data
            stock_data = download_data(symbol)
            
            if stock_data.empty:
                logger.warning(f"No data available for {symbol}")
                return None
                
            logger.info(f"Downloaded {len(stock_data)} rows for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to download data for {symbol}: {e}")
            return None
        
        results = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'predictions': {}
        }
        
        # Predict for close, high, and low prices
        for key_to_predict in ["close", "high", "low"]:
            logger.info(f"Predicting {key_to_predict} for {symbol}")
            
            try:
                # Preprocess data
                processed_data = self.preprocess_data(stock_data, key_to_predict)
                
                if processed_data.empty:
                    logger.warning(f"No processed data for {symbol} {key_to_predict}")
                    continue
                
                if len(processed_data) < lookback_days + prediction_days:
                    logger.warning(f"Not enough data for {symbol} {key_to_predict}")
                    continue
                
                # Split into training and validation
                validation_data = processed_data[-prediction_days:] if len(processed_data) > prediction_days else processed_data
                
                # Make predictions for each day
                predictions = []
                for pred_idx in reversed(range(1, min(prediction_days + 1, len(processed_data)))):
                    current_context = processed_data[:-pred_idx] if pred_idx > 0 else processed_data
                    
                    if 'y' not in current_context.columns:
                        logger.error("Target variable 'y' not found")
                        break
                    
                    context_values = current_context['y'].dropna().values
                    
                    if len(context_values) == 0:
                        continue
                    
                    # Use last lookback_days as context
                    if len(context_values) > lookback_days:
                        context_values = context_values[-lookback_days:]
                    
                    # Make prediction
                    prediction = self.make_single_prediction(context_values)
                    predictions.append(prediction)
                
                if not predictions:
                    logger.warning(f"No predictions generated for {symbol} {key_to_predict}")
                    continue
                
                # Calculate metrics
                actual_values = validation_data['y'].dropna().values if len(validation_data) > 0 else np.array([])
                metrics = self.calculate_trading_metrics(predictions[:-1], actual_values[:-1])
                
                # Store results
                last_price = stock_data[key_to_predict.title()].iloc[-1]
                predicted_change = predictions[-1] if predictions else 0
                predicted_price = last_price * (1 + predicted_change)
                
                results['predictions'][key_to_predict] = {
                    'last_price': float(last_price),
                    'predicted_change_pct': float(predicted_change * 100),
                    'predicted_price': float(predicted_price),
                    'mae': metrics['mae'],
                    'total_return': metrics['total_return'],
                    'win_rate': metrics['win_rate'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'predictions': [float(p) for p in predictions],
                    'trading_signal': 'BUY' if predicted_change > 0 else 'SELL'
                }
                
                logger.info(f"Successfully predicted {key_to_predict} for {symbol}: {predicted_change:.4f}")
                
            except Exception as e:
                logger.error(f"Error predicting {key_to_predict} for {symbol}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return results
    
    def make_predictions_batch(self, symbols: List[str], lookback_days: int = 30, 
                             prediction_days: int = 7, save_to_csv: bool = True) -> pd.DataFrame:
        """
        Make predictions for multiple stocks and optionally save to CSV
        
        Args:
            symbols: List of stock symbols
            lookback_days: Number of days to use as context
            prediction_days: Number of days to predict ahead
            save_to_csv: Whether to save results to CSV file
            
        Returns:
            DataFrame with all predictions
        """
        all_predictions = []
        
        for symbol in symbols:
            try:
                prediction = self.make_prediction(symbol, lookback_days, prediction_days)
                if prediction and prediction['predictions']:
                    # Flatten the results for CSV
                    row = {
                        'symbol': prediction['symbol'],
                        'timestamp': prediction['timestamp']
                    }
                    
                    for price_type, data in prediction['predictions'].items():
                        for key, value in data.items():
                            if key != 'predictions':  # Skip the full predictions array
                                row[f'{price_type}_{key}'] = value
                    
                    all_predictions.append(row)
                    
            except Exception as e:
                logger.error(f"Failed to predict {symbol}: {e}")
                continue
        
        # Convert to DataFrame
        df_results = pd.DataFrame(all_predictions)
        
        # Save to CSV if requested
        if save_to_csv and not df_results.empty:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stock_predictions_toto_{timestamp}.csv"
            df_results.to_csv(filename, index=False)
            logger.info(f"Predictions saved to {filename}")
        
        return df_results
    
    def format_predictions(self, predictions: List[Dict[str, Any]]) -> str:
        """Format predictions for easy reading"""
        if not predictions:
            return "No predictions available"
        
        output = []
        output.append("\n" + "="*60)
        output.append("STOCK PREDICTIONS")
        output.append("="*60)
        
        for pred in predictions:
            if not pred or 'predictions' not in pred:
                continue
                
            symbol = pred['symbol']
            output.append(f"\n{symbol}")
            output.append("-"*20)
            
            for price_type, data in pred['predictions'].items():
                signal = data['trading_signal']
                output.append(f"\n{price_type.upper()}:")
                output.append(f"  Current Price: ${data['last_price']:.2f}")
                output.append(f"  Predicted Change: {data['predicted_change_pct']:.2f}%")
                output.append(f"  Predicted Price: ${data['predicted_price']:.2f}")
                output.append(f"  Signal: {signal}")
                
                if data['mae'] is not None:
                    output.append(f"  MAE: {data['mae']:.4f}")
                    output.append(f"  Win Rate: {data['win_rate']:.1%}")
                    output.append(f"  Sharpe Ratio: {data['sharpe_ratio']:.2f}")
        
        return "\n".join(output)
    
    def run_simple_backtest(self, symbol: str, test_days: int = 30, lookback_days: int = 30) -> pd.DataFrame:
        """
        Run a simple backtest to evaluate prediction accuracy
        
        Args:
            symbol: Stock symbol to test
            test_days: Number of days to test
            lookback_days: Days of historical data for each prediction
            
        Returns:
            DataFrame with prediction results
        """
        logger.info(f"Running backtest for {symbol} over {test_days} days")
        
        try:
            # Get historical data
            stock_data = download_data(symbol)
            
            if stock_data.empty or len(stock_data) < test_days + lookback_days:
                logger.error(f"Not enough data for backtesting {symbol}")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Failed to get data for backtesting: {e}")
            return pd.DataFrame()
        
        results = []
        
        # Test predictions for the last test_days
        for i in range(test_days, 0, -1):
            test_date = stock_data.index[-i]
            
            # Get data up to test date (no look-ahead bias)
            historical_data = stock_data[:-i]
            
            # Preprocess data
            processed_data = self.preprocess_data(historical_data, "close")
            
            if processed_data.empty or len(processed_data) < lookback_days:
                continue
            
            # Get context for prediction
            context_values = processed_data['y'].dropna().values[-lookback_days:]
            
            # Make prediction
            predicted_change = self.make_single_prediction(context_values)
            
            # Get actual change
            current_price = historical_data['Close'].iloc[-1]
            next_price = stock_data['Close'].iloc[-i]
            actual_change = (next_price - current_price) / current_price
            
            # Record results
            results.append({
                'date': test_date,
                'current_price': current_price,
                'predicted_change': predicted_change,
                'actual_change': actual_change,
                'predicted_price': current_price * (1 + predicted_change),
                'actual_price': next_price,
                'error': abs(predicted_change - actual_change),
                'direction_correct': (predicted_change > 0) == (actual_change > 0)
            })
        
        df_results = pd.DataFrame(results)
        
        if not df_results.empty:
            # Print summary
            accuracy = df_results['direction_correct'].mean()
            avg_error = df_results['error'].mean()
            
            print(f"\nBacktest Results for {symbol}")
            print(f"Direction Accuracy: {accuracy:.1%}")
            print(f"Average Error: {avg_error:.4f}")
            print(f"\nLast 5 predictions:")
            print(df_results[['date', 'predicted_change', 'actual_change', 'direction_correct']])
        
        return df_results


def main():
    """Example usage"""
    print("="*60)
    print("STOCK PREDICTOR WITH DATADOG TOTO MODEL")
    print("="*60)
    
    # Check model availability
    if MODEL_TYPE == "toto":
        print("✓ Using Datadog Toto model")
    elif MODEL_TYPE == "chronos":
        print("⚠ Using Chronos model as fallback")
    else:
        print("⚠ Using simple predictions (no model available)")
        print("\nTo install Datadog Toto model:")
        print("pip install git+https://github.com/DataDog/toto.git")
    
    # Initialize predictor
    predictor = StockPredictor()
    
    # Example 1: Single stock prediction
    print("\n1. Single Stock Prediction")
    print("-"*30)
    result = predictor.make_prediction("AAPL", lookback_days=30, prediction_days=7)
    if result:
        print(predictor.format_predictions([result]))
    
    # Example 2: Multiple stocks
    print("\n2. Multiple Stock Predictions")
    print("-"*30)
    symbols = ["SPY", "AAPL", "MSFT", "GOOGL", "TSLA"]
    df_results = predictor.make_predictions_batch(symbols, lookback_days=50, prediction_days=7, save_to_csv=True)
    
    if not df_results.empty:
        print(f"\nProcessed {len(df_results)} stocks")
        print("\nTop predictions by expected return:")
        df_results['expected_return'] = df_results['close_predicted_change_pct'].fillna(0)
        print(df_results.nlargest(5, 'expected_return')[['symbol', 'close_last_price', 'close_predicted_change_pct', 'close_trading_signal']])
    
    # Example 3: Backtest
    print("\n3. Backtest Example")
    print("-"*30)
    backtest_results = predictor.run_simple_backtest("META", test_days=20, lookback_days=50)
    
    print("\nDone! Check the generated CSV file for detailed results.")


if __name__ == "__main__":
    main()