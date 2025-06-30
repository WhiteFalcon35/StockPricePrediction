def run_continuous_prediction(self, interval_hours=1):
#   """Run predictions at regular intervals"""
    def job():
        print(f"\n[{datetime.now()}] Running scheduled prediction...")
        self.predict_next_price()
        
        # Schedule job
        schedule.every(interval_hours).hours.do(job)
        
    print(f"Starting continuous prediction for {self.symbol}")
    print(f"Predictions will run every {interval_hours} hour(s)")
    print("Press Ctrl+C to stop")
        
        # Run initial prediction
    job()
        
        # Keep running
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("\nStopping continuous prediction...")
    
    def create_streamlit_compatible_version(self):
        #Generate code compatible with your Streamlit app"""
        #streamlit_code = f'''
        
        import streamlit as st
        import numpy as np 
        import pandas as pd
        import yfinance as yf
        from keras.models import load_model
        import plotly.graph_objects as go
        from sklearn.preprocessing import MinMaxScaler
        from datetime import datetime

# Load your trained model
        model = load_model('Stock Predictions Model.keras')

        st.header('Real-Time Stock Market Predictor')
        stock = st.text_input('Enter stock Symbol', '{self.symbol}')

# Get real-time data
        start = '2016-01-01'
        end = datetime.now().strftime('%Y-%m-%d')  # Real-time end date
        df = yf.download(stock, start, end)

        st.subheader('Stock Data')
        st.write(df.tail(10))  # Show latest 10 days

# Chart 1: Price vs MA50 vs MA100
        st.subheader('Price vs MA50 vs MA100')
        ma_50_days = df.Close.rolling(50).mean()
        ma_100_days = df.Close.rolling(100).mean()

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df.index, y=ma_50_days, mode='lines', name='MA50', line=dict(color='red', width=2)))
        fig1.add_trace(go.Scatter(x=df.index, y=ma_100_days, mode='lines', name='MA100', line=dict(color='blue', width=2, dash='dash')))
        fig1.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='green', width=2)))
        fig1.update_layout(title='Price vs MA50 vs MA100', xaxis_title='Date', yaxis_title='Price', legend=dict(x=0, y=1), hovermode='x unified')
        st.plotly_chart(fig1, use_container_width=True)

        # Chart 2: Price vs MA100 vs MA200
        st.subheader('Price vs MA100 vs MA200')
        ma_200_days = df.Close.rolling(200).mean()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=ma_100_days, mode='lines', name='MA100', line=dict(color='red', width=2)))
        fig2.add_trace(go.Scatter(x=df.index, y=ma_200_days, mode='lines', name='MA200', line=dict(color='blue', width=2, dash='dash')))
        fig2.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='green', width=2)))
        fig2.update_layout(title='Price vs MA100 vs MA200', xaxis_title='Date', yaxis_title='Price', legend=dict(x=0, y=1), hovermode='x unified')
        st.plotly_chart(fig2, use_container_width=True)

        # Prediction Chart
        df_train = df[['Close']][0:int(len(df)*0.80)]
        df_test = df[['Close']][int(len(df)*0.80):]
        scaler = MinMaxScaler(feature_range=(0,1))
        past_100_days = df_train.tail(100)
        df_test = pd.concat([past_100_days, df_test], ignore_index=True)
        df_test_scaler = scaler.fit_transform(df_test)

        x, y = [], []
        for i in range(100, df_test_scaler.shape[0]):
            x.append(df_test_scaler[i-100:i])
            y.append(df_test_scaler[i,0])

        x, y = np.array(x), np.array(y)
        predict = model.predict(x)
        scale = 1/scaler.scale_
        predict = predict * scale
        y = y * scale

        st.subheader('Original Price vs Predicted Price')
        time_index = df.index[-len(y):]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=time_index, y=y, mode='lines+markers', name='Original Price', line=dict(color='red', width=2)))
        fig3.add_trace(go.Scatter(x=time_index, y=predict.flatten(), mode='lines+markers', name='Predicted Price', line=dict(color='blue', width=2, dash='dot')))
        fig3.update_layout(title='Original Price vs Predicted Price', xaxis_title='Date', yaxis_title='Price', legend=dict(x=0, y=1), hovermode='x unified')
        st.plotly_chart(fig3, use_container_width=True)

        # Real-time prediction for next day
        latest_100_days = df[['Close']].tail(100)
        latest_100_scaled = scaler.transform(latest_100_days)
        X_pred = np.array([latest_100_scaled])
        next_day_prediction = model.predict(X_pred)
        next_day_price = scaler.inverse_transform(next_day_prediction)[0][0]

        st.subheader('Next Day Prediction')
        current_price = df['Close'].iloc[-1]
        st.metric(
            label="Tomorrow's Predicted Price",
            value=f"${{{next_day_price:.2f}}}",
            delta=f"${{{next_day_price - current_price:.2f}}} ({{{((next_day_price - current_price) / current_price) * 100:.2f}}}%)"
        )



        # Real-Time Stock Price Prediction System
        # pip install alpha_vantage (already installed)
        # pip install --upgrade tensorflow (already installed)
        # pip install schedule (new - for automated predictions)
        # pip install plotly (new - for interactive graphs)

        import numpy as np 
        import pandas as pd
        from pandas_datareader import data as pdr
        import matplotlib.pyplot as plt 
        import yfinance as yf
        import seaborn as sns
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.layers import Dense, Dropout, LSTM
        from tensorflow.keras.models import Sequential, load_model
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        import schedule
        import time
        from datetime import datetime, timedelta
        import warnings
        warnings.filterwarnings('ignore')

        class RealTimeStockPredictor:
            def __init__(self, symbol='TSLA', model_path='Stock Predictions Model.keras'):
                self.symbol = symbol
                self.model_path = model_path
                self.scaler = MinMaxScaler(feature_range=(0,1))
                self.model = None
                self.sequence_length = 100  # Same as your original 100-day lookback
                
            def fetch_real_time_data(self, start_date='2016-01-01'):
                #"""Fetch the most recent data including today"""
                try:
                    # Get data up to today
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    print(f"Fetching data from {start_date} to {end_date}")
                    
                    df = yf.download(self.symbol, start=start_date, end=end_date)
                    df.reset_index(inplace=True)
                    
                    # Add the same technical indicators as your original code
                    ma_50_days = df['Close'].rolling(window=100).mean()
                    ma_150_days = df['Close'].rolling(window=150).mean()
                    
                    # Store for plotting if needed
                    self.ma_50 = ma_50_days
                    self.ma_150 = ma_150_days
                    
                    df.dropna(inplace=True)
                    print(f"Data fetched successfully. Latest date: {df['Date'].iloc[-1]}")
                    return df
                    
                except Exception as e:
                    print(f"Error fetching data: {e}")
                    return None
            
            def prepare_data_for_training(self, df):
                #"""Prepare data the same way as your original code"""
                # Split data (80% train, 20% test) - but now with real-time data
                split_point = int(len(df) * 0.80)
                df_train = df[['Close']][:split_point]
                df_test = df[['Close']][split_point:]
                
                # Scale the data
                df_train_scale = self.scaler.fit_transform(df_train)
                df_test_scale = self.scaler.transform(df_test)
                
                return df_train, df_test, df_train_scale, df_test_scale
            
            def create_sequences(self, data, sequence_length=100):
                #"""Create sequences for LSTM training - same as your original logic"""
                x, y = [], []
                for i in range(sequence_length, len(data)):
                    x.append(data[i-sequence_length:i])
                    y.append(data[i, 0])
                return np.array(x), np.array(y)
            
            def build_model(self, input_shape):
                #"""Build the exact same model architecture as your original"""
                model = Sequential()
                
                # Same architecture as your original model
                model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=input_shape))
                model.add(LSTM(units=50, activation='relu', return_sequences=True))
                model.add(Dropout(0.2))
                
                model.add(LSTM(units=60, activation='relu', return_sequences=True))
                model.add(Dropout(0.3))
                
                model.add(LSTM(units=80, activation='relu', return_sequences=True))
                model.add(Dropout(0.4))
                
                model.add(LSTM(units=120, activation='relu'))
                model.add(Dropout(0.5))
                
                model.add(Dense(units=1))
                
                model.compile(optimizer='adam', loss='mean_squared_error')
                return model
            
            def train_model(self, retrain=False):
                #"""Train model with current data or load existing model"""
                try:
                    if not retrain and self.model_path:
                        print("Loading existing model...")
                        self.model = load_model(self.model_path)
                        print("Model loaded successfully!")
                        return True
                except:
                    print("Could not load existing model. Training new model...")
                    retrain = True
                
                if retrain:
                    print("Fetching data for training...")
                    df = self.fetch_real_time_data()
                    if df is None:
                        return False
                    
                    df_train, df_test, df_train_scale, df_test_scale = self.prepare_data_for_training(df)
                    
                    # Create sequences
                    x_train, y_train = self.create_sequences(df_train_scale, self.sequence_length)
                    
                    print(f"Training data shape: x={x_train.shape}, y={y_train.shape}")
                    
                    # Build and train model
                    self.model = self.build_model((x_train.shape[1], 1))
                    
                    print("Training model...")
                    history = self.model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1)
                    
                    # Save model
                    self.model.save(self.model_path)
                    print(f"Model saved as {self.model_path}")
                    
                return True
            
            def predict_next_price(self, plot_results=True):
                #"""Make real-time prediction for next trading day"""
                if self.model is None:
                    print("Model not loaded. Please train first.")
                    return None
                
                # Fetch latest data
                df = self.fetch_real_time_data()
                if df is None:
                    return None
                
                # Get last 100 days for prediction
                last_100_days = df[['Close']].tail(self.sequence_length)
                
                # Scale the data using the same scaler
                last_100_scaled = self.scaler.transform(last_100_days)
                
                # Reshape for prediction
                X_pred = np.array([last_100_scaled])
                
                # Make prediction
                predicted_scaled = self.model.predict(X_pred, verbose=0)
                
                # Inverse transform to get actual price
                predicted_price = self.scaler.inverse_transform(predicted_scaled)[0][0]
                
                current_price = df['Close'].iloc[-1]
                prediction_date = datetime.now() + timedelta(days=1)
                
                print(f"\n{'='*50}")
                print(f"REAL-TIME STOCK PREDICTION FOR {self.symbol}")
                print(f"{'='*50}")
                print(f"Current Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Latest Price: ${current_price:.2f}")
                print(f"Predicted Price for {prediction_date.strftime('%Y-%m-%d')}: ${predicted_price:.2f}")
                print(f"Predicted Change: ${predicted_price - current_price:.2f} ({((predicted_price - current_price) / current_price) * 100:.2f}%)")
                print(f"{'='*50}")
                
                if plot_results:
                    # Create all three interactive charts
                    self.plot_interactive_charts(df)
                    
                    # Create prediction comparison chart
                    self.plot_prediction_results(df)
                
                return {
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'prediction_date': prediction_date.strftime('%Y-%m-%d'),
                    'change': predicted_price - current_price,
                    'change_percent': ((predicted_price - current_price) / current_price) * 100
                }
            
            def plot_interactive_charts(self, df):
                #"""Create all three interactive charts like your Streamlit app"""
                
                # Calculate moving averages
                ma_50_days = df['Close'].rolling(50).mean()
                ma_100_days = df['Close'].rolling(100).mean()
                ma_200_days = df['Close'].rolling(200).mean()
                
                # Chart 1: Price vs MA50 vs MA100
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=df['Date'], y=ma_50_days,
                    mode='lines',
                    name='MA50',
                    line=dict(color='red', width=2)
                ))
                fig1.add_trace(go.Scatter(
                    x=df['Date'], y=ma_100_days,
                    mode='lines',
                    name='MA100',
                    line=dict(color='blue', width=2, dash='dash')
                ))
                fig1.add_trace(go.Scatter(
                    x=df['Date'], y=df['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='green', width=2)
                ))
                fig1.update_layout(
                    title=f'{self.symbol} - Price vs MA50 vs MA100',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    legend=dict(x=0, y=1),
                    hovermode='x unified',
                    width=1000,
                    height=600
                )
                fig1.show()
                
                # Chart 2: Price vs MA100 vs MA200
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=df['Date'], y=ma_100_days,
                    mode='lines',
                    name='MA100',
                    line=dict(color='red', width=2)
                ))
                fig2.add_trace(go.Scatter(
                    x=df['Date'], y=ma_200_days,
                    mode='lines',
                    name='MA200',
                    line=dict(color='blue', width=2, dash='dash')
                ))
                fig2.add_trace(go.Scatter(
                    x=df['Date'], y=df['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='green', width=2)
                ))
                fig2.update_layout(
                    title=f'{self.symbol} - Price vs MA100 vs MA200',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    legend=dict(x=0, y=1),
                    hovermode='x unified',
                    width=1000,
                    height=600
                )
                fig2.show()
                
                return fig1, fig2
            
            def plot_prediction_results(self, df):
                #"""Create the prediction comparison chart like your Streamlit app"""
                # Prepare data exactly like your original code
                df_train = df[['Close']][0:int(len(df)*0.80)]
                df_test = df[['Close']][int(len(df)*0.80):]
                
                past_100_days = df_train.tail(100)
                df_test_combined = pd.concat([past_100_days, df_test], ignore_index=True)
                
                # Scale the test data
                scaler = MinMaxScaler(feature_range=(0,1))
                df_test_scale = scaler.fit_transform(df_test_combined)
                
                # Create sequences for prediction
                x, y = [], []
                for i in range(100, df_test_scale.shape[0]):
                    x.append(df_test_scale[i-100:i])
                    y.append(df_test_scale[i,0])
                
                x, y = np.array(x), np.array(y)
                
                # Make predictions
                predict = self.model.predict(x)
                
                # Inverse transform
                scale = 1/scaler.scale_
                predict = predict * scale
                y = y * scale
                
                # Create time index for the test period
                test_start_idx = int(len(df) * 0.80)
                time_index = df['Date'].iloc[test_start_idx + 100:]  # Skip first 100 days used for sequences
                
                # Ensure lengths match
                min_len = min(len(time_index), len(y), len(predict))
                time_index = time_index[:min_len]
                y = y[:min_len]
                predict = predict[:min_len]
                
                # Create the interactive prediction comparison chart
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=time_index, y=y,
                    mode='lines+markers',
                    name='Original Price',
                    line=dict(color='red', width=2),
                    marker=dict(size=4)
                ))
                fig3.add_trace(go.Scatter(
                    x=time_index, y=predict.flatten(),
                    mode='lines+markers',
                    name='Predicted Price',
                    line=dict(color='blue', width=2, dash='dot'),
                    marker=dict(size=4)
                ))
                
                # Add hover information showing exact values
                fig3.update_traces(
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Date: %{x}<br>' +
                                'Price: $%{y:.2f}<br>' +
                                '<extra></extra>'
                )
                
                fig3.update_layout(
                    title=f'{self.symbol} - Original Price vs Predicted Price',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    legend=dict(x=0, y=1),
                    hovermode='x unified',
                    width=1000,
                    height=600
                )
                fig3.show()
                
                # Calculate and display accuracy metrics
                mse = np.mean((y - predict.flatten())**2)
                mae = np.mean(np.abs(y - predict.flatten()))
                mape = np.mean(np.abs((y - predict.flatten()) / y)) * 100
                
                print(f"\nPrediction Accuracy Metrics:")
                print(f"Mean Squared Error (MSE): {mse:.2f}")
                print(f"Mean Absolute Error (MAE): {mae:.2f}")
                print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                
                return fig3
            
            def create_comprehensive_analysis(self):
                #"""Create a comprehensive analysis with all charts and predictions"""
                print(f"Creating comprehensive analysis for {self.symbol}...")
                
                # Fetch latest data
                df = self.fetch_real_time_data()
                if df is None:
                    return None
                
                # Create all interactive charts
                print("Generating interactive charts...")
                fig1, fig2 = self.plot_interactive_charts(df)
                
                # Create prediction analysis
                print("Generating prediction analysis...")
                fig3 = self.plot_prediction_results(df)
                
                # Make real-time prediction
                print("Making real-time prediction...")
                prediction_result = self.predict_next_price(plot_results=False)
                
                return {
                    'charts': [fig1, fig2, fig3],
                    'prediction': prediction_result,
                    'data': df
                }
                #"""Run predictions at regular intervals"""
                
            def job():
                print(f"\n[{datetime.now()}] Running scheduled prediction...")
                self.predict_next_price()
                
                # Schedule job
            schedule.every(interval_hours).hours.do(job)
                
            print(f"Starting continuous prediction for {self.symbol}")
            print(f"Predictions will run every {interval_hours} hour(s)")
            print("Press Ctrl+C to stop")
                
                # Run initial prediction
            job()
                
                # Keep running
            try:
                while True:
                        schedule.run_pending()
                        time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                print("\nStopping continuous prediction...")

        # Example usage
        if __name__ == "__main__":
            # Initialize predictor
            predictor = RealTimeStockPredictor(symbol='TSLA')
            
            # Option 1: Train new model with latest data
            print("Training/Loading model...")
            if predictor.train_model(retrain=False):  # Set to True to retrain with latest data
                
                # Option 2: Create comprehensive analysis with all 3 charts
                print("\nCreating comprehensive analysis...")
                analysis = predictor.create_comprehensive_analysis()
                
                # Option 3: Make single prediction (uncomment to use)
                # result = predictor.predict_next_price()
                
                # Option 4: Generate Streamlit-compatible code (uncomment to use)
                # streamlit_code = predictor.create_streamlit_compatible_version()
                # print("Streamlit code generated!")
                
                # Option 5: Run continuous predictions (uncomment to use)
                # predictor.run_continuous_prediction(interval_hours=1)
            
            print("\nReal-time prediction system ready!")
            
            # You can also manually call predictions anytime:
            # result = predictor.predict_next_price()

        # Additional utility functions for extended functionality
        def compare_multiple_stocks(symbols=['TSLA', 'AAPL', 'GOOGL']):
            #"""Compare predictions for multiple stocks"""
            predictions = {}
            
            for symbol in symbols:
                print(f"\nAnalyzing {symbol}...")
                predictor = RealTimeStockPredictor(symbol=symbol)
                
                if predictor.train_model(retrain=False):
                    result = predictor.predict_next_price(plot_results=False)
                    predictions[symbol] = result
            
            # Summary
            print(f"\n{'='*60}")
            print("MULTI-STOCK PREDICTION SUMMARY")
            print(f"{'='*60}")
            for symbol, pred in predictions.items():
                if pred:
                    print(f"{symbol}: ${pred['current_price']:.2f} → ${pred['predicted_price']:.2f} ({pred['change_percent']:.2f}%)")
            
            return predictions

        # Uncomment to run multi-stock analysis
        # compare_multiple_stocks(['TSLA', 'AAPL', 'MSFT'])