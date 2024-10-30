import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import numpy as np
import random
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="Forecast App (Prophet)", layout="wide")

# Title
st.title("Forecast App (Prophet)")

# File upload or generate data option
data_option = st.radio("Select Data Input Method", ("Upload CSV", "Generate Data"))

if data_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Ask user to select columns
        date_column = st.selectbox("Select DATE column", df.columns.tolist())
        market_column = st.selectbox("Select MARKET column", df.columns.tolist())
        conversions_column = st.selectbox("Select CONVERSIONS column", df.columns.tolist())
        
        # Data preprocessing
        pivoted_npu_df = df.pivot(index=date_column, columns=market_column, values=conversions_column)
        pivoted_npu_df.fillna(0, inplace=True)
        pivoted_npu_df = pivoted_npu_df.astype(int)
        pivoted_npu_df = pivoted_npu_df.sort_index()
        pivoted_npu_df.index.name = 'DATE'
        
        # Market selector
        markets = pivoted_npu_df.columns.tolist()
        market_selector = st.selectbox('Select Market', markets)
        
        if market_selector:
            # Get market data
            df_market = pivoted_npu_df[market_selector]
            df_market = df_market.reset_index()
            df_market.columns = ['ds', 'y']
            
            # Display timeline chart
            fig = px.line(
                df_market,
                x='ds',
                y='y',
                title=f"{market_selector} - NPU timeline"
            )
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="NPU",
                template='plotly_white',
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Prophet section
            st.header("Prophet Forecast")
            
            with st.expander("Forecast Settings"):
                periods = st.number_input(
                    "Forecast Period (days)", 
                    min_value=1,
                    value=365,
                    step=1,
                    help="Enter the number of days to forecast"
                )
                daily_seasonality = st.checkbox("Daily Seasonality", value=True)
                weekly_seasonality = st.checkbox("Weekly Seasonality", value=True)
                yearly_seasonality = st.checkbox("Yearly Seasonality", value=True)
                
                # Add country holiday selector (only if market isn't ROW)
                include_holidays = False
                country_code = None
                if market_selector != 'ROW':
                    include_holidays = st.checkbox("Include Country Holidays", value=False)
                    if include_holidays:
                        country_mapping = {
                            'AU': 'AU',
                            'CA': 'CA',
                            'US': 'US',
                            'GB': 'GB',
                            'DE': 'DE',
                            'SE': 'SE',
                            'SG': 'SG'
                            # Add more country mappings as needed
                        }
                        country_code = country_mapping.get(market_selector)
                        if country_code:
                            st.info(f"Will include holidays for {country_code}")
                        else:
                            st.warning("No holiday data available for this market")
                            include_holidays = False
            
            if st.button("Generate Forecast"):
                with st.spinner("Generating forecast..."):
                    # Create and fit Prophet model
                    m = Prophet(
                        daily_seasonality=daily_seasonality,
                        weekly_seasonality=weekly_seasonality,
                        yearly_seasonality=yearly_seasonality
                    )
                    
                    # Add holidays if selected and available
                    if include_holidays and country_code:
                        m.add_country_holidays(country_name=country_code)
                    
                    # Fit the model
                    m.fit(df_market)
                    
                    # Create future dataframe
                    future = m.make_future_dataframe(periods=periods)
                    
                    # Make predictions
                    forecast = m.predict(future)
                    
                    # Display forecast plot
                    st.subheader("Forecast Plot")
                    fig_forecast = plot_plotly(m, forecast)
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Display components
                    st.subheader("Forecast Components")
                    fig_components = plot_components_plotly(m, forecast)
                    st.plotly_chart(fig_components, use_container_width=True)
                    
                    # Display forecast data
                    st.subheader("Forecast Data")
                    forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
                    st.dataframe(forecast_data)
                    
                    # Add download button for full forecast data
                    csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
                    st.download_button(
                        label="Download full forecast data as CSV",
                        data=csv,
                        file_name=f'forecast_{market_selector}.csv',
                        mime='text/csv',
                    )

elif data_option == "Generate Data":
    # Generate synthetic data
    num_days = 365  # Number of days for the data
    start_date = datetime.now() - timedelta(days=num_days)
    date_range = [start_date + timedelta(days=i) for i in range(num_days)]
    
    markets = ['AU', 'CA', 'US', 'GB', 'DE', 'SE', 'SG']
    data = {
        'DATE': [],
        'MARKET': [],
        'CONVERSIONS': []
    }
    
    for date in date_range:
        for market in markets:
            data['DATE'].append(date)
            data['MARKET'].append(market)
            data['CONVERSIONS'].append(random.randint(0, 100))  # Random conversions

    df = pd.DataFrame(data)
    
    # Display the generated data
    st.write("Generated Data:")
    st.dataframe(df)

    # Ask user to select columns
    date_column = 'DATE'
    market_column = 'MARKET'
    conversions_column = 'CONVERSIONS'
    
    # Data preprocessing
    pivoted_npu_df = df.pivot(index=date_column, columns=market_column, values=conversions_column)
    pivoted_npu_df.fillna(0, inplace=True)
    pivoted_npu_df = pivoted_npu_df.astype(int)
    pivoted_npu_df = pivoted_npu_df.sort_index()
    pivoted_npu_df.index.name = 'DATE'
    
    # Market selector
    markets = pivoted_npu_df.columns.tolist()
    market_selector = st.selectbox('Select Market', markets)
    
    if market_selector:
        # Get market data
        df_market = pivoted_npu_df[market_selector]
        df_market = df_market.reset_index()
        df_market.columns = ['ds', 'y']
        
        # Display timeline chart
        fig = px.line(
            df_market,
            x='ds',
            y='y',
            title=f"{market_selector} - NPU timeline"
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="NPU",
            template='plotly_white',
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Prophet section
        st.header("Prophet Forecast")
        
        with st.expander("Forecast Settings"):
            periods = st.number_input(
                "Forecast Period (days)", 
                min_value=1,
                value=365,
                step=1,
                help="Enter the number of days to forecast"
            )
            daily_seasonality = st.checkbox("Daily Seasonality", value=True)
            weekly_seasonality = st.checkbox("Weekly Seasonality", value=True)
            yearly_seasonality = st.checkbox("Yearly Seasonality", value=True)
            
            # Add country holiday selector (only if market isn't ROW)
            include_holidays = False
            country_code = None
            if market_selector != 'ROW':
                include_holidays = st.checkbox("Include Country Holidays", value=False)
                if include_holidays:
                    country_mapping = {
                        'AU': 'AU',
                        'CA': 'CA',
                        'US': 'US',
                        'GB': 'GB',
                        'DE': 'DE',
                        'SE': 'SE',
                        'SG': 'SG'
                        # Add more country mappings as needed
                    }
                    country_code = country_mapping.get(market_selector)
                    if country_code:
                        st.info(f"Will include holidays for {country_code}")
                    else:
                        st.warning("No holiday data available for this market")
                        include_holidays = False
        
        if st.button("Generate Forecast"):
            with st.spinner("Generating forecast..."):
                # Create and fit Prophet model
                m = Prophet(
                    daily_seasonality=daily_seasonality,
                    weekly_seasonality=weekly_seasonality,
                    yearly_seasonality=yearly_seasonality
                )
                
                # Add holidays if selected and available
                if include_holidays and country_code:
                    m.add_country_holidays(country_name=country_code)
                
                # Fit the model
                m.fit(df_market)
                
                # Create future dataframe
                future = m.make_future_dataframe(periods=periods)
                
                # Make predictions
                forecast = m.predict(future)
                
                # Display forecast plot
                st.subheader("Forecast Plot")
                fig_forecast = plot_plotly(m, forecast)
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Display components
                st.subheader("Forecast Components")
                fig_components = plot_components_plotly(m, forecast)
                st.plotly_chart(fig_components, use_container_width=True)
                
                # Display forecast data
                st.subheader("Forecast Data")
                forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
                st.dataframe(forecast_data)
                
                # Add download button for full forecast data
                csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
                st.download_button(
                    label="Download full forecast data as CSV",
                    data=csv,
                    file_name=f'forecast_{market_selector}.csv',
                    mime='text/csv',
                )

else:
    st.info("Please upload a CSV file or generate data to begin. The file should contain columns: DATE, MARKET, and CONVERSIONS")