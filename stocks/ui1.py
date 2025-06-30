# Real-Time Stock Predictor - Streamlit Web App
# Save this as: streamlit_app.py
# Run with: streamlit run streamlit_app.py

import streamlit as st
import numpy as np 
import pandas as pd
import yfinance as yf
from keras.models import load_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Real-Time Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("📊 Configuration")
st.sidebar.markdown("---")

# Model loading function
@st.cache_resource
def load_prediction_model():
    try:
        model = load_model('Stock Predictions Model.keras')
        return model
    except:
        st.error("⚠️ Model file not found! Please ensure 'Stock Predictions Model.keras' is in the same directory.")
        return None

# Data fetching function with caching
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(symbol, start_date='2016-01-01'):
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        df.reset_index(inplace=True)
        return df, end_date
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None, None

# Prediction function
def make_prediction(model, df, scaler):
    try:
        # Get last 100 days for prediction
        last_100_days = df[['Close']].tail(100)
        last_100_scaled = scaler.transform(last_100_days)
        X_pred = np.array([last_100_scaled])
        
        # Make prediction
        prediction_scaled = model.predict(X_pred, verbose=0)
        prediction = scaler.inverse_transform(prediction_scaled)[0][0]
        
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 Real-Time Stock Market Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar inputs
    stock_symbol = st.sidebar.text_input(
        "📈 Enter Stock Symbol", 
        value="TSLA", 
        help="Enter a valid stock ticker (e.g., TSLA, AAPL, GOOGL)"
    ).upper()
    
    start_date = st.sidebar.date_input(
        "📅 Start Date",
        value=datetime(2016, 1, 1),
        help="Select the start date for historical data"
    )
    
    show_raw_data = st.sidebar.checkbox("📋 Show Raw Data", value=False)
    prediction_days = st.sidebar.slider("🔮 Prediction Horizon (Days)", 1, 7, 1)
    
    
    # Load model
    model = load_prediction_model()
    if model is None:
        return
    
    
    # Fetch data
    with st.spinner(f"🔍 Fetching real-time data for {stock_symbol}..."):
        df, end_date = fetch_stock_data(stock_symbol, start_date.strftime('%Y-%m-%d'))
    
    if df is None or df.empty:
        st.error(f"❌ Could not fetch data for {stock_symbol}. Please check the symbol and try again.")
        return
    
    # Display basic info
    col1, col2, col3, col4 = st.columns(4)

    # Current & previous price
    try:
        current_price = float(df['Close'].iloc[-1])
    except:
        current_price = 0.0

    try:
        prev_price = float(df['Close'].iloc[-2]) if len(df) > 1 else current_price
    except:
        prev_price = current_price

    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0

    # Safe volume handling
    try:
        volume = df['Volume'].iloc[-1]
        volume = int(volume) if pd.notna(volume) else 0
    except:
        volume = 0

    # 52-week High and Low
    try:
        high_52 = df['High'].max()
    except:
        high_52 = 0

    try:
        low_52 = df['Low'].min()
    except:
        low_52 = 0

    # Display metrics
    with col1:
        st.metric("💰 Current Price", f"${current_price:.2f}", delta=f"{price_change:.2f} ({price_change_pct:.2f}%)")

    with col2:
        st.metric("📊 Volume", f"{volume:,}")

    try:
        high_52 = float(df['High'].max())
    except:
        high_52 = 0.0

    try:
        low_52 = float(df['Low'].min())
    except:
        low_52 = 0.0
    
    # Show raw data if requested
    if show_raw_data:
        st.subheader("📋 Latest Stock Data")
        st.dataframe(df.tail(10), use_container_width=True)
    
    # Calculate moving averages
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA100'] = df['Close'].rolling(100).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    
    # Chart 1: Price vs MA50 vs MA100
    st.subheader("📊 Price vs MA50 vs MA100")
    fig1 = go.Figure()
    
    fig1.add_trace(go.Scatter(
        x=df['Date'], y=df['MA50'],
        mode='lines',
        name='MA50',
        line=dict(color='red', width=2),
        hovertemplate='<b>MA50</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    fig1.add_trace(go.Scatter(
        x=df['Date'], y=df['MA100'],
        mode='lines',
        name='MA100',
        line=dict(color='blue', width=2, dash='dash'),
        hovertemplate='<b>MA100</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    fig1.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='green', width=2),
        hovertemplate='<b>Close Price</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    fig1.update_layout(
        title=f'{stock_symbol} - Price vs MA50 vs MA100',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        legend=dict(x=1, y=0),
        hovermode='x unified',
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Chart 2: Price vs MA100 vs MA200
    st.subheader("📈 Price vs MA100 vs MA200")
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=df['Date'], y=df['MA100'],
        mode='lines',
        name='MA100',
        line=dict(color='red', width=2),
        hovertemplate='<b>MA100</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    fig2.add_trace(go.Scatter(
        x=df['Date'], y=df['MA200'],
        mode='lines',
        name='MA200',
        line=dict(color='blue', width=2, dash='dash'),
        hovertemplate='<b>MA200</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    fig2.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='green', width=2),
        hovertemplate='<b>Close Price</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    fig2.update_layout(
        title=f'{stock_symbol} - Price vs MA100 vs MA200',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        legend=dict(x=1, y=0),
        hovermode='x unified',
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Prepare data for prediction model
    df_train = df[['Close']][0:int(len(df)*0.80)]
    df_test = df[['Close']][int(len(df)*0.80):]
    
    scaler = MinMaxScaler(feature_range=(0,1))
    past_100_days = df_train.tail(100)
    df_test_combined = pd.concat([past_100_days, df_test], ignore_index=True)
    df_test_scaled = scaler.fit_transform(df_test_combined)
    
    # Create sequences for prediction
    x_test, y_test = [], []
    for i in range(100, df_test_scaled.shape[0]):
        x_test.append(df_test_scaled[i-100:i])
        y_test.append(df_test_scaled[i,0])
    
    if len(x_test) > 0:
        x_test, y_test = np.array(x_test), np.array(y_test)
        
        # Make predictions on test data
        predictions_scaled = model.predict(x_test, verbose=0)
        scale = 1/scaler.scale_[0]
        predictions = predictions_scaled * scale
        actual = y_test * scale
        
        # Chart 3: Original vs Predicted
        st.subheader("🎯 Original Price vs Predicted Price")
        
        # Create time index for test period
        test_start_idx = int(len(df) * 0.80) + 100
        time_index = df['Date'].iloc[test_start_idx:test_start_idx + len(actual)]
        
        fig3 = go.Figure()
        
        fig3.add_trace(go.Scatter(
            x=time_index, y=actual,
            mode='lines+markers',
            name='Original Price',
            line=dict(color='red', width=2),
            marker=dict(size=4),
            hovertemplate='<b>Original Price</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        fig3.add_trace(go.Scatter(
            x=time_index, y=predictions.flatten(),
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='blue', width=2, dash='dot'),
            marker=dict(size=4),
            hovertemplate='<b>Predicted Price</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        fig3.update_layout(
            title=f'{stock_symbol} - Original Price vs Predicted Price',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            legend=dict(x=0, y=1),
            hovermode='x unified',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Calculate accuracy metrics
        mse = np.mean((actual - predictions.flatten())**2)
        mae = np.mean(np.abs(actual - predictions.flatten()))
        mape = np.mean(np.abs((actual - predictions.flatten()) / actual)) * 100
        accuracy = max(0, 100 - mape)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Accuracy", f"{accuracy:.1f}%")
        with col2:
            st.metric("📊 Mean Squared Error (MSE)", f"{mse:.2f}")
        with col3:
            st.metric("📈 Mean Absolute Error (MAE)", f"{mae:.2f}")
        with col4:
            st.metric("📉 Mean Absolute Percentage Error (MAPE)", f"{mape:.2f}%")
    
    # Real-time prediction section
    st.markdown("---")
    st.subheader("🔮 Real-Time Predictions")
    
    try:
        predictions_list = []
        
        for day in range(1, prediction_days + 1):
            prediction = make_prediction(model, df, scaler)
            if prediction:
                future_date = datetime.now() + timedelta(days=day)
                change = prediction - current_price
                change_pct = (change / current_price) * 100
                
                predictions_list.append({
                    'Day': f"Day +{day}",
                    'Date': future_date.strftime('%Y-%m-%d'),
                    'Predicted Price': f"${prediction:.2f}",
                    'Change': f"${change:.2f}",
                    'Change %': f"{change_pct:.2f}%"
                })
        
        if predictions_list:
            # Main prediction box
            main_prediction = predictions_list[0]
            st.markdown(f"""
            <div class="prediction-box">
                <h2>🎯 Tomorrow's Prediction</h2>
                <h1>{main_prediction['Predicted Price']}</h1>
                <h3>{main_prediction['Change']} ({main_prediction['Change %']})</h3>
                <p>Prediction for {main_prediction['Date']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Extended predictions table
            if prediction_days > 1:
                st.subheader("📅 Extended Predictions")
                predictions_df = pd.DataFrame(predictions_list)
                st.dataframe(predictions_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align: center; color: #666;">
            <p>📊 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>💡 Data provided by Yahoo Finance | Model: LSTM Neural Network</p>
        </div>
        """, unsafe_allow_html=True)
    


if __name__ == "__main__":
    main()