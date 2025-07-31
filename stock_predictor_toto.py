#stock_predictor_toto.py
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Import your data function
from your_pattern_module import download_data as download_data

# Try to import Toto model from DataDog (install with: pip install toto-ts)
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
        # Fallback to Chronos model (install with: pip install chronos-forecasting)
        from chronos import ChronosPipeline
        MODEL_AVAILABLE = True
        MODEL_TYPE = "chronos"
        logger.info("Chronos model available as fallback")
    except ImportError:
        logger.warning("No time series model available. Using simple predictions.")
        MODEL_AVAILABLE = False
        MODEL_TYPE = "none"

class StockPredictor:
    def __init__(self, model_name="Datadog/Toto-Open-Base-1.0"):
        self.model = None
        self.forecaster = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler.fit_transform([[-1, 1]])  # Initialize scaler
        self.model_name = model_name
        
    def load_pipeline(self):
        """Load the time series forecasting model"""
        if self.model is None and MODEL_AVAILABLE:
            try:
                if MODEL_TYPE == "toto":
                    # Load Datadog Toto model
                    self.model = Toto.from_pretrained(self.model_name)
                    if torch.cuda.is_available():
                        self.model.to('cuda')
                        # Optionally compile for faster inference
                        try:
                            self.model.compile()
                        except:
                            logger.info("Model compilation failed, continuing without compilation")
                    else:
                        self.model.to('cpu')
                    self.forecaster = TotoForecaster(self.model.model)
                    logger.info(f"Loaded Datadog Toto model {self.model_name} successfully")
                    
                elif MODEL_TYPE == "chronos":
                    # Fallback to Chronos
                    from chronos import ChronosPipeline
                    self.model = ChronosPipeline.from_pretrained(
                        "amazon/chronos-t5-small",
                        device_map="cuda" if torch.cuda.is_available() else "cpu",
                        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    )
                    logger.info(f"Loaded Chronos model as fallback")
                    
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model = None
                self.forecaster = None
        elif not MODEL_AVAILABLE:
            logger.warning("No time series model available. Using simple predictions.")
            
    def percent_movements_augment(self, data):
        """Convert price data to percentage changes"""
        if len(data) < 2:
            return np.array([])
        diff_data = np.diff(data.flatten())
        prev_data = data[:-1].flatten()
        # Avoid division by zero
        prev_data = np.where(prev_data == 0, 1e-8, prev_data)
        return diff_data / prev_data
    
    def preprocess_data(self, stock_data, key_to_predict="close"):
        """Preprocess stock data for prediction"""
        data = stock_data.copy()
        logger.info(f"Original data shape: {data.shape}")
        logger.info(f"Columns: {data.columns.tolist()}")
        
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
                    logger.warning(f"No percentage changes calculated for {col}")
                    processed_data[col] = [0] * max(0, len(data) - 1)
        
        if processed_data.empty:
            logger.error("No processed data available")
            return pd.DataFrame()
        
        logger.info(f"Processed data shape: {processed_data.shape}")
        
        # Create target variable (next day's percentage change)
        if key_to_predict in processed_data.columns:
            processed_data['y'] = processed_data[key_to_predict].shift(-1)
        else:
            logger.error(f"Key {key_to_predict} not found in processed data")
            return pd.DataFrame()
        
        # Create trade weight (1 for positive moves, -1 for negative)
        processed_data['trade_weight'] = (processed_data['y'] > 0) * 2 - 1
        
        # Add time index and date columns
        processed_data['ds'] = pd.date_range(start="2020-01-01", periods=len(processed_data), freq="D")
        processed_data['id'] = processed_data.index
        processed_data['unique_id'] = 1
        
        # Drop the last row (no target available) and any NaN values
        processed_data = processed_data.dropna()
        logger.info(f"Final processed data shape: {processed_data.shape}")
        
        return processed_data
    
    def make_prediction(self, symbol, lookback_days=100, prediction_days=7):
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
            logger.info(f"Downloaded {len(stock_data)} rows for {symbol}")
            logger.info(f"Data columns: {stock_data.columns.tolist()}")
            logger.info(f"Data date range: {stock_data.index[0] if len(stock_data) > 0 else 'N/A'} to {stock_data.index[-1] if len(stock_data) > 0 else 'N/A'}")
            
            if stock_data.empty:
                logger.warning(f"No data available for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to download data for {symbol}: {e}")
            return None
        
        results = {}
        
        # Predict for close, high, and low prices (lowercase)
        for key_to_predict in ["close", "high", "low"]:
            # Standardize column names to lowercase
            stock_data_normalized = stock_data.copy()
            stock_data_normalized.columns = stock_data_normalized.columns.str.lower()
            
            if key_to_predict not in stock_data_normalized.columns:
                logger.warning(f"Column {key_to_predict} not found in data for {symbol}")
                continue
                
            logger.info(f"Predicting {key_to_predict} for {symbol}")
            
            try:
                # Preprocess data
                processed_data = self.preprocess_data(stock_data_normalized, key_to_predict)
                
                if processed_data.empty:
                    logger.warning(f"No processed data for {symbol} {key_to_predict}")
                    continue
                
                if len(processed_data) < max(lookback_days, prediction_days):
                    logger.warning(f"Not enough data for {symbol} {key_to_predict}. Have {len(processed_data)}, need {max(lookback_days, prediction_days)}")
                    # Reduce requirements if not enough data
                    lookback_days = min(lookback_days, len(processed_data) // 2)
                    prediction_days = min(prediction_days, len(processed_data) // 4)
                    if lookback_days < 10 or prediction_days < 1:
                        logger.warning(f"Still not enough data after reduction")
                        continue
                
                # Split into training and validation
                train_data = processed_data[:-prediction_days] if len(processed_data) > prediction_days else processed_data[:-1]
                validation_data = processed_data[-prediction_days:] if len(processed_data) > prediction_days else processed_data[-1:]
                
                # Load model if not already loaded
                self.load_pipeline()
                
                predictions = []
                
                # Make predictions for each day
                for pred_idx in reversed(range(1, min(prediction_days + 1, len(processed_data)))):
                    current_context = processed_data[:-pred_idx] if pred_idx > 0 else processed_data
                    
                    if 'y' not in current_context.columns:
                        logger.error("Target variable 'y' not found in processed data")
                        break
                        
                    context_values = current_context['y'].dropna().values
                    
                    if len(context_values) == 0:
                        logger.warning(f"No context values available for prediction {pred_idx}")
                        continue
                    
                    # Use last lookback_days as context
                    if len(context_values) > lookback_days:
                        context_values = context_values[-lookback_days:]
                    
                    context = torch.tensor(context_values, dtype=torch.float)
                    
                    # Load model if not already loaded
                    self.load_pipeline()
                    
                    if self.forecaster is not None and MODEL_TYPE == "toto":
                        try:
                            # Prepare data for Toto model
                            device = 'cuda' if torch.cuda.is_available() else 'cpu'
                            
                            # Toto expects (channels, time_steps) format
                            input_series = context.unsqueeze(0).to(device)  # Add channel dimension
                            
                            # Create timestamp information (required by API but not used by current model)
                            timestamp_seconds = torch.zeros_like(input_series).to(device)
                            time_interval_seconds = torch.full((1,), 86400).to(device)  # Daily intervals (86400 seconds)
                            
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
                                num_samples=32,  # Fewer samples for speed
                                samples_per_batch=32,
                            )
                            
                            # Get median prediction
                            prediction = forecast.mean[0, 0].item()  # First channel, first timestep
                            
                        except Exception as e:
                            logger.warning(f"Toto prediction failed: {e}. Using simple forecast.")
                            # Fallback to simple momentum prediction
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
                            # Fallback to simple momentum prediction
                            prediction = np.mean(context_values[-5:]) if len(context_values) >= 5 else 0.0
                    else:
                        # Simple momentum-based prediction if model not available
                        recent_trend = np.mean(context_values[-5:]) if len(context_values) >= 5 else 0.0
                        volatility = np.std(context_values[-10:]) if len(context_values) >= 10 else 0.01
                        # Add some randomness based on recent volatility
                        prediction = recent_trend + np.random.normal(0, volatility * 0.1)
                    
                    predictions.append(prediction)
                
                if not predictions:
                    logger.warning(f"No predictions generated for {symbol} {key_to_predict}")
                    continue
                
                # Calculate validation metrics
                if len(validation_data) > 0:
                    actual_values = validation_data['y'].dropna().values
                    pred_values = predictions[:len(actual_values)]
                    
                    if len(actual_values) > 0 and len(pred_values) > 0:
                        min_len = min(len(actual_values), len(pred_values))
                        actual_values = actual_values[:min_len]
                        pred_values = pred_values[:min_len]
                        
                        mae = np.mean(np.abs(np.array(actual_values) - np.array(pred_values)))
                        
                        # Trading signals (buy if predicted positive change)
                        trading_signals = (np.array(pred_values) > 0) * 2 - 1
                        
                        # Simple profit calculation (assuming we follow the signals)
                        returns = np.array(actual_values) * trading_signals
                        total_return = np.sum(returns)
                        
                    else:
                        mae = float('inf')
                        total_return = 0
                else:
                    mae = float('inf')
                    total_return = 0
                
                # Store results
                last_price = stock_data_normalized[key_to_predict].iloc[-1]
                predicted_change = predictions[-1] if predictions else 0
                predicted_price = last_price * (1 + predicted_change)
                
                results[key_to_predict.lower()] = {
                    'last_price': float(last_price),
                    'predicted_change_pct': float(predicted_change),
                    'predicted_price': float(predicted_price),
                    'mae': float(mae) if mae != float('inf') else None,
                    'total_return': float(total_return),
                    'predictions': [float(p) for p in predictions],
                    'trading_signal': 1 if predicted_change > 0 else -1
                }
                
                logger.info(f"Successfully predicted {key_to_predict} for {symbol}: {predicted_change:.4f}")
                
            except Exception as e:
                logger.error(f"Error predicting {key_to_predict} for {symbol}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'predictions': results
        }
    
    def predict_multiple_stocks(self, symbols, lookback_days=100, prediction_days=7):
        """
        Make predictions for multiple stocks
        
        Args:
            symbols: List of stock symbols
            lookback_days: Number of days to use as context
            prediction_days: Number of days to predict ahead
            
        Returns:
            List of prediction dictionaries
        """
        all_predictions = []
        
        for symbol in symbols:
            try:
                prediction = self.make_prediction(symbol, lookback_days, prediction_days)
                if prediction:
                    all_predictions.append(prediction)
            except Exception as e:
                logger.error(f"Failed to predict {symbol}: {e}")
                continue
        
        return all_predictions
    
    def format_predictions(self, predictions):
        """Format predictions for easy reading"""
        if not predictions:
            return "No predictions available"
        
        output = []
        for pred in predictions:
            symbol = pred['symbol']
            output.append(f"\n=== {symbol} Predictions ===")
            
            for price_type, data in pred['predictions'].items():
                signal = "BUY" if data['trading_signal'] == 1 else "SELL"
                output.append(f"{price_type.upper()}:")
                output.append(f"  Current Price: ${data['last_price']:.2f}")
                output.append(f"  Predicted Change: {data['predicted_change_pct']:.4f} ({data['predicted_change_pct']*100:.2f}%)")
                output.append(f"  Predicted Price: ${data['predicted_price']:.2f}")
                output.append(f"  Trading Signal: {signal}")
                output.append(f"  Validation MAE: {data['mae']:.4f}")
                output.append("")
        
        return "\n".join(output)
    
    def backtest_strategy(self, symbol, backtest_days=30, lookback_days=50, initial_capital=10000):
        """
        Backtest the trading strategy over the last N days
        
        Args:
            symbol: Stock symbol to backtest
            backtest_days: Number of days to backtest
            lookback_days: Days of historical data to use for each prediction
            initial_capital: Starting capital for backtest
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest for {symbol} over {backtest_days} days")
        
        try:
            # Get full historical data
            full_data = download_data(symbol)
            full_data.columns = full_data.columns.str.lower()
            
            if len(full_data) < backtest_days + lookback_days:
                logger.error(f"Not enough data for backtesting. Need {backtest_days + lookback_days}, have {len(full_data)}")
                return None
                
            logger.info(f"Full data shape: {full_data.shape}, backtesting last {backtest_days} days")
            
        except Exception as e:
            logger.error(f"Failed to get data for backtesting {symbol}: {e}")
            return None
        
        # Initialize backtest variables
        capital = initial_capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        daily_pnl = []
        trade_log = []
        
        # Start backtesting from the point where we have enough lookback data
        start_idx = len(full_data) - backtest_days
        
        for day_idx in range(start_idx, len(full_data) - 1):  # -1 because we need next day's data to execute
            current_date = full_data.index[day_idx]
            next_date = full_data.index[day_idx + 1]
            
            # Get data up to current day (no look-ahead bias)
            historical_data = full_data.iloc[:day_idx + 1]
            current_price = historical_data['close'].iloc[-1]
            next_day_data = full_data.iloc[day_idx + 1]  # This is what actually happens next day
            
            # Format date properly - handle both datetime and integer indices
            if hasattr(current_date, 'strftime'):
                date_str = current_date.strftime('%Y-%m-%d')
            else:
                date_str = str(current_date)
            
            logger.info(f"Backtesting day {day_idx - start_idx + 1}/{backtest_days}: {date_str} at ${current_price:.2f}")
            
            daily_start_capital = capital
            daily_trades = []
            
            # Check if we need to close existing position first
            if position != 0:
                should_close = False
                exit_price = None
                exit_reason = ""
                
                # Check stop loss and take profit using next day's high/low
                if position == 1:  # Long position
                    # Check if stop loss hit (next day's low goes below stop loss)
                    if next_day_data['low'] <= stop_loss:
                        should_close = True
                        exit_price = stop_loss
                        exit_reason = "Stop Loss"
                    # Check if take profit hit (next day's high goes above take profit)
                    elif next_day_data['high'] >= take_profit:
                        should_close = True
                        exit_price = take_profit
                        exit_reason = "Take Profit"
                    # Otherwise close at next day's open
                    else:
                        exit_price = next_day_data['open']
                        exit_reason = "Market Close"
                        
                elif position == -1:  # Short position
                    # Check if stop loss hit (next day's high goes above stop loss)
                    if next_day_data['high'] >= stop_loss:
                        should_close = True
                        exit_price = stop_loss
                        exit_reason = "Stop Loss"
                    # Check if take profit hit (next day's low goes below take profit)
                    elif next_day_data['low'] <= take_profit:
                        should_close = True
                        exit_price = take_profit
                        exit_reason = "Take Profit"
                    # Otherwise close at next day's open
                    else:
                        exit_price = next_day_data['open']
                        exit_reason = "Market Close"
                
                if exit_price:
                    # Calculate P&L
                    if position == 1:  # Long
                        pnl = (exit_price - entry_price) / entry_price * capital
                    else:  # Short
                        pnl = (entry_price - exit_price) / entry_price * capital
                    
                    capital += pnl
                    
                    trade_log.append({
                        'entry_date': current_date,
                        'exit_date': next_date,
                        'position': 'LONG' if position == 1 else 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'pnl': pnl,
                        'capital': capital
                    })
                    
                    daily_trades.append({
                        'action': 'CLOSE',
                        'position': 'LONG' if position == 1 else 'SHORT',
                        'price': exit_price,
                        'reason': exit_reason,
                        'pnl': pnl
                    })
                    
                    logger.info(f"  Closed {('LONG' if position == 1 else 'SHORT')} at ${exit_price:.2f} ({exit_reason}), P&L: ${pnl:.2f}")
                    
                    # Reset position
                    position = 0
                    entry_price = 0
                    stop_loss = 0
                    take_profit = 0
            
            # Make prediction for tomorrow using only data up to today
            try:
                # Use sufficient lookback data
                prediction_data = historical_data.iloc[-lookback_days:] if len(historical_data) > lookback_days else historical_data
                
                if len(prediction_data) < 10:  # Need minimum data
                    logger.warning(f"Not enough data for prediction on {date_str}")
                    daily_pnl.append({
                        'date': current_date,
                        'start_capital': daily_start_capital,
                        'end_capital': capital,
                        'daily_pnl': capital - daily_start_capital,
                        'trades': daily_trades
                    })
                    continue
                
                # Get predictions for close, high, low
                predictions = {}
                for price_type in ['close', 'high', 'low']:
                    processed_data = self.preprocess_data(prediction_data.copy(), price_type)
                    
                    if processed_data.empty or len(processed_data) < 5:
                        predictions[price_type] = 0
                        continue
                    
                    context_values = processed_data['y'].dropna().values
                    if len(context_values) == 0:
                        predictions[price_type] = 0
                        continue
                    
                    # Use last 30 days as context
                    context_values = context_values[-30:] if len(context_values) > 30 else context_values
                    context = torch.tensor(context_values, dtype=torch.float)
                    
                    # Load model if not already loaded
                    self.load_pipeline()
                    
                    # Make prediction
                    if self.forecaster is not None and MODEL_TYPE == "toto":
                        try:
                            device = 'cuda' if torch.cuda.is_available() else 'cpu'
                            input_series = context.unsqueeze(0).to(device)
                            timestamp_seconds = torch.zeros_like(input_series).to(device)
                            time_interval_seconds = torch.full((1,), 86400).to(device)
                            
                            inputs = MaskedTimeseries(
                                series=input_series,
                                padding_mask=torch.full_like(input_series, True, dtype=torch.bool),
                                id_mask=torch.zeros_like(input_series),
                                timestamp_seconds=timestamp_seconds,
                                time_interval_seconds=time_interval_seconds,
                            )
                            
                            forecast = self.forecaster.forecast(inputs, prediction_length=1, num_samples=32, samples_per_batch=32)
                            prediction = forecast.mean[0, 0].item()
                        except Exception as e:
                            logger.warning(f"Toto prediction failed: {e}")
                            prediction = np.mean(context_values[-5:]) if len(context_values) >= 5 else 0.0
                    elif self.model is not None and MODEL_TYPE == "chronos":
                        try:
                            forecast = self.model.predict(context.unsqueeze(0), prediction_length=1)
                            prediction = torch.median(forecast[0]).item()
                        except:
                            prediction = np.mean(context_values[-5:]) if len(context_values) >= 5 else 0.0
                    else:
                        prediction = np.mean(context_values[-5:]) if len(context_values) >= 5 else 0.0
                    
                    predictions[price_type] = prediction
                
                # Generate trading signal
                close_prediction = predictions.get('close', 0)
                high_prediction = predictions.get('high', 0)
                low_prediction = predictions.get('low', 0)
                
                # Only trade if we have a strong signal (> 0.5% predicted move)
                min_signal_threshold = 0.005
                
                if position == 0 and abs(close_prediction) > min_signal_threshold:  # No current position
                    # Determine position based on predictions
                    if close_prediction > 0:  # Bullish prediction
                        position = 1  # Go long
                        entry_price = next_day_data['open']  # Enter at next day's open
                        
                        # Set stop loss and take profit based on predictions
                        predicted_high_price = current_price * (1 + high_prediction)
                        predicted_low_price = current_price * (1 + low_prediction)
                        
                        # Conservative stop loss (2% or predicted low, whichever is closer)
                        stop_loss = max(predicted_low_price, current_price * 0.98)
                        # Conservative take profit (1.5% or predicted high, whichever is closer)  
                        take_profit = min(predicted_high_price, current_price * 1.015)
                        
                        daily_trades.append({
                            'action': 'OPEN LONG',
                            'price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'prediction': close_prediction
                        })
                        
                        logger.info(f"  Opened LONG at ${entry_price:.2f}, SL: ${stop_loss:.2f}, TP: ${take_profit:.2f}")
                        
                    elif close_prediction < 0:  # Bearish prediction
                        position = -1  # Go short
                        entry_price = next_day_data['open']
                        
                        # Set stop loss and take profit for short position
                        predicted_high_price = current_price * (1 + high_prediction)
                        predicted_low_price = current_price * (1 + low_prediction)
                        
                        # Conservative stop loss (2% above or predicted high)
                        stop_loss = min(predicted_high_price, current_price * 1.02)
                        # Conservative take profit (1.5% below or predicted low)
                        take_profit = max(predicted_low_price, current_price * 0.985)
                        
                        daily_trades.append({
                            'action': 'OPEN SHORT',
                            'price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'prediction': close_prediction
                        })
                        
                        logger.info(f"  Opened SHORT at ${entry_price:.2f}, SL: ${stop_loss:.2f}, TP: ${take_profit:.2f}")
                
            except Exception as e:
                logger.error(f"Error making prediction for {date_str}: {e}")
            
            # Record daily P&L
            daily_pnl.append({
                'date': current_date,
                'price': current_price,
                'start_capital': daily_start_capital,
                'end_capital': capital,
                'daily_pnl': capital - daily_start_capital,
                'position': position,
                'trades': daily_trades,
                'predictions': predictions
            })
        
        # Close any remaining position at the end
        if position != 0:
            final_price = full_data['close'].iloc[-1]
            if position == 1:
                final_pnl = (final_price - entry_price) / entry_price * capital
            else:
                final_pnl = (entry_price - final_price) / entry_price * capital
            
            capital += final_pnl
            trade_log.append({
                'entry_date': current_date,
                'exit_date': full_data.index[-1],
                'position': 'LONG' if position == 1 else 'SHORT',
                'entry_price': entry_price,
                'exit_price': final_price,
                'exit_reason': 'End of Backtest',
                'pnl': final_pnl,
                'capital': capital
            })
        
        # Calculate performance metrics
        total_return = capital - initial_capital
        total_return_pct = (capital / initial_capital - 1) * 100
        
        daily_returns = [d['daily_pnl'] / d['start_capital'] for d in daily_pnl if d['start_capital'] > 0]
        avg_daily_return = np.mean(daily_returns) if daily_returns else 0
        volatility = np.std(daily_returns) if daily_returns else 0
        sharpe_ratio = (avg_daily_return / volatility * np.sqrt(252)) if volatility > 0 else 0
        
        max_capital = initial_capital
        max_drawdown = 0
        for d in daily_pnl:
            max_capital = max(max_capital, d['end_capital'])
            drawdown = (max_capital - d['end_capital']) / max_capital
            max_drawdown = max(max_drawdown, drawdown)
        
        winning_trades = [t for t in trade_log if t['pnl'] > 0]
        losing_trades = [t for t in trade_log if t['pnl'] < 0]
        win_rate = len(winning_trades) / len(trade_log) if trade_log else 0
        
        return {
            'symbol': symbol,
            'backtest_period': f"{backtest_days} days",
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'total_trades': len(trade_log),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_daily_return': avg_daily_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'daily_pnl': daily_pnl,
            'trade_log': trade_log
        }
    
    def print_backtest_results(self, backtest_result):
        """Print formatted backtest results"""
        if not backtest_result:
            print("No backtest results to display")
            return
        
        result = backtest_result
        print("\n" + "="*60)
        print(f"BACKTEST RESULTS FOR {result['symbol']}")
        print("="*60)
        print(f"Period: {result['backtest_period']}")
        print(f"Initial Capital: ${result['initial_capital']:,.2f}")
        print(f"Final Capital: ${result['final_capital']:,.2f}")
        print(f"Total Return: ${result['total_return']:,.2f} ({result['total_return_pct']:.2f}%)")
        print(f"Total Trades: {result['total_trades']}")
        print(f"Win Rate: {result['win_rate']:.1%}")
        print(f"Average Daily Return: {result['avg_daily_return']:.4f}")
        print(f"Volatility: {result['volatility']:.4f}")
        print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {result['max_drawdown']:.1%}")
        
        print("\n" + "="*60)
        print("DAILY P&L SUMMARY")
        print("="*60)
        print("Date         Price     Daily P&L   Total      Trades")
        print("-" * 60)
        
        for day in result['daily_pnl'][-10:]:  # Show last 10 days
            trades_summary = ", ".join([t['action'] for t in day['trades']]) if day['trades'] else "No trades"
            # Handle date formatting safely
            if hasattr(day['date'], 'strftime'):
                date_str = day['date'].strftime('%Y-%m-%d')
            else:
                date_str = str(day['date'])
            price_str = f"${day['price']:.2f}"
            pnl_str = f"${day['daily_pnl']:.2f}"
            total_str = f"${day['end_capital']:.2f}"
            print(f"{date_str:<12} {price_str:<9} {pnl_str:<11} {total_str:<10} {trades_summary}")
        
        if len(result['trade_log']) > 0:
            print("\n" + "="*60)
            print("TRADE LOG")
            print("="*60)
            print("Entry        Exit         Type    Entry$    Exit$     Reason       P&L")
            print("-" * 80)
            
            for trade in result['trade_log'][-5:]:  # Show last 5 trades
                # Handle date formatting safely for trade log
                if hasattr(trade['entry_date'], 'strftime'):
                    entry_date = trade['entry_date'].strftime('%Y-%m-%d')
                else:
                    entry_date = str(trade['entry_date'])
                    
                if hasattr(trade['exit_date'], 'strftime'):
                    exit_date = trade['exit_date'].strftime('%Y-%m-%d')
                else:
                    exit_date = str(trade['exit_date'])
                    
                position = trade['position']
                entry_price = f"${trade['entry_price']:.2f}"
                exit_price = f"${trade['exit_price']:.2f}"
                exit_reason = trade['exit_reason']
                pnl = f"${trade['pnl']:.2f}"
                
                print(f"{entry_date:<12} {exit_date:<12} {position:<7} {entry_price:<9} {exit_price:<9} {exit_reason:<12} {pnl}")
        
        print("\n" + "="*60)
    
    def run_multiple_backtests(self, symbols, backtest_days=30, lookback_days=50, initial_capital=10000):
        """Run backtests for multiple symbols"""
        all_results = []
        
        for symbol in symbols:
            logger.info(f"Running backtest for {symbol}")
            result = self.backtest_strategy(symbol, backtest_days, lookback_days, initial_capital)
            if result:
                all_results.append(result)
                self.print_backtest_results(result)
            else:
                logger.error(f"Backtest failed for {symbol}")
        
        return all_results
    
    def simple_prediction_test(self, symbol, test_days=20, lookback_days=50):
        """
        Simple test to compare predictions vs actual results
        
        Args:
            symbol: Stock symbol to test
            test_days: Number of days to test predictions
            lookback_days: Days of historical data to use for each prediction
            
        Returns:
            DataFrame with predictions vs actual results
        """
        logger.info(f"Starting simple prediction test for {symbol} over {test_days} days")
        
        try:
            # Get full historical data
            full_data = download_data(symbol)
            full_data.columns = full_data.columns.str.lower()
            
            if len(full_data) < test_days + lookback_days + 1:
                logger.error(f"Not enough data for testing. Need {test_days + lookback_days + 1}, have {len(full_data)}")
                return None
                
            logger.info(f"Full data shape: {full_data.shape}, testing last {test_days} days")
            
        except Exception as e:
            logger.error(f"Failed to get data for testing {symbol}: {e}")
            return None
        
        results = []
        
        # Start testing from the point where we have enough lookback data
        start_idx = len(full_data) - test_days - 1  # -1 because we need next day's actual data
        
        for day_idx in range(start_idx, len(full_data) - 1):  # -1 because we need next day's data
            current_date = full_data.index[day_idx]
            next_date = full_data.index[day_idx + 1]
            
            # Get data up to current day (no look-ahead bias)
            historical_data = full_data.iloc[:day_idx + 1]
            current_price = historical_data['close'].iloc[-1]
            next_day_actual_price = full_data.iloc[day_idx + 1]['close']  # What actually happened
            
            # Format date properly
            if hasattr(current_date, 'strftime'):
                date_str = current_date.strftime('%Y-%m-%d')
            else:
                date_str = str(current_date)
            
            logger.info(f"Testing day {day_idx - start_idx + 1}/{test_days}: {date_str} - Current: ${current_price:.2f}")
            
            # Make prediction for tomorrow using only data up to today
            try:
                # Use sufficient lookback data for prediction
                prediction_data = historical_data.iloc[-lookback_days:] if len(historical_data) > lookback_days else historical_data
                
                if len(prediction_data) < 10:  # Need minimum data
                    logger.warning(f"Not enough data for prediction on {date_str}")
                    continue
                
                # Get prediction for close price
                processed_data = self.preprocess_data(prediction_data.copy(), 'close')
                
                if processed_data.empty or len(processed_data) < 5:
                    predicted_change = 0
                else:
                    context_values = processed_data['y'].dropna().values
                    if len(context_values) == 0:
                        predicted_change = 0
                    else:
                        # Use last 30 days as context
                        context_values = context_values[-30:] if len(context_values) > 30 else context_values
                        context = torch.tensor(context_values, dtype=torch.float)
                        
                        # Load model if not already loaded
                        self.load_pipeline()
                        
                        # Make prediction
                        if self.forecaster is not None and MODEL_TYPE == "toto":
                            try:
                                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                                input_series = context.unsqueeze(0).to(device)
                                timestamp_seconds = torch.zeros_like(input_series).to(device)
                                time_interval_seconds = torch.full((1,), 86400).to(device)
                                
                                inputs = MaskedTimeseries(
                                    series=input_series,
                                    padding_mask=torch.full_like(input_series, True, dtype=torch.bool),
                                    id_mask=torch.zeros_like(input_series),
                                    timestamp_seconds=timestamp_seconds,
                                    time_interval_seconds=time_interval_seconds,
                                )
                                
                                forecast = self.forecaster.forecast(inputs, prediction_length=1, num_samples=32, samples_per_batch=32)
                                predicted_change = forecast.mean[0, 0].item()
                            except Exception as e:
                                logger.warning(f"Toto prediction failed: {e}. Using simple forecast.")
                                predicted_change = np.mean(context_values[-5:]) if len(context_values) >= 5 else 0.0
                        elif self.model is not None and MODEL_TYPE == "chronos":
                            try:
                                forecast = self.model.predict(context.unsqueeze(0), prediction_length=1)
                                predicted_change = torch.median(forecast[0]).item()
                            except Exception as e:
                                logger.warning(f"Chronos prediction failed: {e}. Using simple forecast.")
                                predicted_change = np.mean(context_values[-5:]) if len(context_values) >= 5 else 0.0
                        else:
                            predicted_change = np.mean(context_values[-5:]) if len(context_values) >= 5 else 0.0
                
                # Calculate actual change
                actual_change = (next_day_actual_price - current_price) / current_price
                
                # Calculate predicted price
                predicted_price = current_price * (1 + predicted_change)
                
                # Calculate prediction error
                price_error = abs(predicted_price - next_day_actual_price)
                change_error = abs(predicted_change - actual_change)
                
                # Determine if prediction direction was correct
                predicted_direction = "UP" if predicted_change > 0 else "DOWN" if predicted_change < 0 else "FLAT"
                actual_direction = "UP" if actual_change > 0 else "DOWN" if actual_change < 0 else "FLAT"
                direction_correct = predicted_direction == actual_direction
                
                result = {
                    'date': date_str,
                    'current_price': current_price,
                    'predicted_change_pct': predicted_change * 100,  # Convert to percentage
                    'actual_change_pct': actual_change * 100,
                    'predicted_price': predicted_price,
                    'actual_price': next_day_actual_price,
                    'price_error': price_error,
                    'change_error_pct': change_error * 100,
                    'predicted_direction': predicted_direction,
                    'actual_direction': actual_direction,
                    'direction_correct': direction_correct
                }
                
                results.append(result)
                
                logger.info(f"  Predicted: {predicted_change*100:.2f}% (${predicted_price:.2f})")
                logger.info(f"  Actual: {actual_change*100:.2f}% (${next_day_actual_price:.2f})")
                logger.info(f"  Direction: {predicted_direction} vs {actual_direction} ({'✓' if direction_correct else '✗'})")
                
            except Exception as e:
                logger.error(f"Error making prediction for {date_str}: {e}")
                continue
        
        # Convert to DataFrame for easy analysis
        df_results = pd.DataFrame(results)
        
        if not df_results.empty:
            # Calculate summary statistics
            accuracy = df_results['direction_correct'].mean()
            avg_change_error = df_results['change_error_pct'].mean()
            avg_price_error = df_results['price_error'].mean()
            
            print(f"\n{'='*80}")
            print(f"PREDICTION TEST RESULTS FOR {symbol}")
            print(f"{'='*80}")
            print(f"Test Period: {test_days} days")
            print(f"Direction Accuracy: {accuracy:.1%}")
            print(f"Average Change Error: {avg_change_error:.2f}%")
            print(f"Average Price Error: ${avg_price_error:.2f}")
            
            print(f"\n{'='*80}")
            print("DETAILED PREDICTIONS vs ACTUAL")
            print(f"{'='*80}")
            print("Date         Current$  Pred%    Actual%  Pred$    Actual$  Direction    Correct")
            print("-" * 80)
            
            for _, row in df_results.tail(10).iterrows():  # Show last 10 days
                correct_symbol = "✓" if row['direction_correct'] else "✗"
                print(f"{row['date']:<12} "
                      f"${row['current_price']:<8.2f} "
                      f"{row['predicted_change_pct']:<7.2f}% "
                      f"{row['actual_change_pct']:<8.2f}% "
                      f"${row['predicted_price']:<7.2f} "
                      f"${row['actual_price']:<8.2f} "
                      f"{row['predicted_direction']:<4} vs {row['actual_direction']:<4} "
                      f"{correct_symbol}")
        
        return df_results


def main():
    """Example usage with Datadog Toto model"""
    print("="*60)
    print("STOCK PREDICTOR WITH DATADOG TOTO MODEL")
    print("="*60)
    
    # Check if Toto is available
    if MODEL_TYPE == "toto":
        print("✓ Using Datadog Toto model")
    elif MODEL_TYPE == "chronos":
        print("⚠ Using Chronos model as fallback")
    else:
        print("⚠ Using simple predictions (no model available)")
        print("\nTo install Datadog Toto model:")
        print("1. pip install toto-ts")
        print("2. Or git clone https://github.com/DataDog/toto.git")
    
    predictor = StockPredictor()
    
    # Test single prediction first
    logger.info("Testing single prediction...")
    result = predictor.make_prediction("SPY", lookback_days=30, prediction_days=3)
    if result and result['predictions']:
        print("Single prediction successful!")
        print(predictor.format_predictions([result]))
    
    # Run simple prediction test to see accuracy
    print("\n" + "="*60)
    print("RUNNING SIMPLE PREDICTION TEST")
    print("="*60)
    
    # Test predictions vs actual results
    test_result = predictor.simple_prediction_test(
        symbol="TSLA", 
        test_days=50,  # Test last 20 days
        lookback_days=300   # Use more history for Toto model
    )
    
    if test_result is not None and not test_result.empty:
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"prediction_test_toto_{timestamp}.csv"
        test_result.to_csv(csv_file, index=False)
        logger.info(f"Detailed results saved to {csv_file}")
        
        # Show summary
        print(f"\nSUMMARY:")
        print(f"- Model used: {MODEL_TYPE.upper()}")
        print(f"- Total predictions: {len(test_result)}")
        print(f"- Direction accuracy: {test_result['direction_correct'].mean():.1%}")
        print(f"- Average error: {test_result['change_error_pct'].mean():.2f}%")
        
        # Check prediction distribution
        up_predictions = (test_result['predicted_change_pct'] > 0).sum()
        down_predictions = (test_result['predicted_change_pct'] < 0).sum()
        flat_predictions = (test_result['predicted_change_pct'] == 0).sum()
        
        print(f"\nPREDICTION DISTRIBUTION:")
        print(f"- UP predictions: {up_predictions} ({up_predictions/len(test_result):.1%})")
        print(f"- DOWN predictions: {down_predictions} ({down_predictions/len(test_result):.1%})")
        print(f"- FLAT predictions: {flat_predictions} ({flat_predictions/len(test_result):.1%})")
        
        # Show best and worst predictions
        if len(test_result) > 0:
            best_pred = test_result.loc[test_result['change_error_pct'].idxmin()]
            worst_pred = test_result.loc[test_result['change_error_pct'].idxmax()]
            
            print(f"\nBest prediction: {best_pred['date']} - Error: {best_pred['change_error_pct']:.2f}%")
            print(f"Worst prediction: {worst_pred['date']} - Error: {worst_pred['change_error_pct']:.2f}%")
    
    print(f"\nToto model should provide better predictions than the always-UP issue you saw before!")


if __name__ == "__main__":
    main()