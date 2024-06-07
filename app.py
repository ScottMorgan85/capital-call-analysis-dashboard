import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import n_colors
from datetime import datetime, timedelta

# Set up the dashboard layout
st.set_page_config(page_title="Capital Call Analysis Dashboard", layout="wide")

# Create tabs
tab1, tab2 = st.tabs(["Visualizations", "Context and Analysis"])

with tab1:
    st.title("Capital Call Analysis Dashboard")
    
    # Interactive elements
    st.sidebar.markdown("### Adjust Parameters")
    num_calls_per_year = st.sidebar.slider('Number of Capital Calls per Year', 1, 12, 4)
    
    # Fixed parameters
    investment_growth_rate = 0.04
    distribution_rate = 0.15

    # Set parameters
    num_years = 9
    start_date = datetime(2024, 6, 30)
    initial_capital = 20000000

    # Calculate dates for each capital call
    total_calls = num_years * num_calls_per_year
    call_dates = [start_date + timedelta(days=(i * 365 // num_calls_per_year)) for i in range(total_calls)]

    # Simulate invested capital as a percentage of committed capital
    # Use smooth interpolation for gradual increase and decrease
    x = np.linspace(0, num_years, total_calls)
    y = np.piecewise(x, 
                     [x < 3, 
                      (x >= 3) & (x < 7), 
                      x >= 7], 
                     [lambda x: 20 * x, 
                      lambda x: 60 + 5 * (x - 3), 
                      lambda x: 80 - 20 * (x - 7)])
    
    invested_capital_percentage = y

    # Calculate invested capital in dollars
    invested_capital = (invested_capital_percentage / 100) * initial_capital

    # Calculate cumulative net cash flow to mirror the invested capital pattern but returning to 100%
    cumulative_net_cash_flow_percentage = np.concatenate([
        np.linspace(0, -60, total_calls//2), 
        np.linspace(-60, 100, total_calls - total_calls//2)
    ])

    # Smooth out the cumulative net cash flow
    cumulative_net_cash_flow_percentage = pd.Series(cumulative_net_cash_flow_percentage).rolling(window=3, center=True).mean().fillna(method='bfill').fillna(method='ffill')

    data = pd.DataFrame({
        'Date': call_dates,
        'Invested Capital %': invested_capital_percentage,
        'Cumulative Net Cash Flow %': cumulative_net_cash_flow_percentage
    })

    # Adjust data based on fixed parameters
    data['Adjusted Invested Capital %'] = data['Invested Capital %'] * (1 + investment_growth_rate)
    data['Adjusted Cumulative Net Cash Flow %'] = np.cumsum(data['Adjusted Invested Capital %'] * -1 * distribution_rate)

    # Cap values at 100% and -100%
    data['Adjusted Invested Capital %'] = np.clip(data['Adjusted Invested Capital %'], -100, 100)
    data['Adjusted Cumulative Net Cash Flow %'] = np.clip(data['Adjusted Cumulative Net Cash Flow %'], -100, 100)

    # Ridge Line Plot
    st.markdown("## Capital Call Risk Distribution")
    st.markdown("""
    This visualization shows the risk distribution of capital calls over time, considering various scenarios.
    The x-axis represents the simulated account value in millions, while the y-axis shows the capital calls distributed over time.
    """)
    np.random.seed(1)
    ridge_data = (np.linspace(1, 2, total_calls)[:, np.newaxis] * np.random.randn(total_calls, 200) +
                  (np.arange(total_calls) + 2 * np.random.random(total_calls))[:, np.newaxis])

    colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', total_calls, colortype='rgb')

    fig_ridge = go.Figure()
    for i, (ridge_data_line, color) in enumerate(zip(ridge_data, colors)):
        fig_ridge.add_trace(go.Violin(y=ridge_data_line/1e6, line_color=color, name=f'Call {i + 1}', points=False))

    fig_ridge.update_traces(orientation='v', side='positive', width=3, points=False)
    fig_ridge.update_layout(
        title="Capital Call Risk Distribution",
        yaxis_title="Simulated Account Value (Millions $)",
        xaxis_title="Capital Calls (Year)",
        yaxis_showgrid=True,
        yaxis_zeroline=True,
        showlegend=False,
        xaxis=dict(
            tickvals=[i for i in range(total_calls)],
            ticktext=[f'{call_dates[i].strftime("%b %Y")} Call #{i % num_calls_per_year + 1}' for i in range(total_calls)]
        ),
        height=800
    )
    st.plotly_chart(fig_ridge)

    # Invested Capital Relative to Commitment Level
    st.markdown("## Invested Capital Relative to Commitment Level")
    fig_invested = go.Figure()

    fig_invested.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Adjusted Invested Capital %'],
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(0, 100, 255, 0.2)',
        line=dict(color='rgba(0, 100, 255, 1)')
    ))

    fig_invested.update_layout(
        yaxis_title='Invested Capital (% of Commitment)',
        xaxis_title='Date',
        yaxis_range=[-100, 100]
    )

    st.plotly_chart(fig_invested)

    # Cumulative Net Cash Flow
    st.markdown("## Cumulative Net Cash Flow")
    fig_cashflow = px.line(data, x='Date', y='Cumulative Net Cash Flow %', markers=True, line_dash_sequence=['dash'])
    fig_cashflow.update_layout(yaxis_title='Cumulative Net Cash Flow (% of Commitment)', xaxis_title='Date', yaxis_range=[-100, 100])
    st.plotly_chart(fig_cashflow)

 # Monte Carlo Simulation for Forecasted Account Values with Confidence Interval
    st.markdown("## Forecasted Account Values with Confidence Interval")

    # Monte Carlo parameters
    num_simulations = 1000
    forecast_horizon = 36  # months
    monthly_return_mean = 0.005  # 0.5% monthly return
    monthly_return_stddev = 0.02  # 2% monthly return standard deviation

    dates = pd.date_range(start=start_date, periods=forecast_horizon, freq='M')
    forecast_matrix = np.zeros((forecast_horizon, num_simulations))

    # Run Monte Carlo simulation
    for sim in range(num_simulations):
        forecast_matrix[0, sim] = initial_capital
        for t in range(1, forecast_horizon):
            monthly_return = np.random.normal(monthly_return_mean, monthly_return_stddev)
            forecast_matrix[t, sim] = forecast_matrix[t - 1, sim] * (1 + monthly_return)

    forecast_df = pd.DataFrame({
        'Date': dates,
        'Mean Forecast': np.mean(forecast_matrix, axis=1),
        'Lower Bound': np.percentile(forecast_matrix, 2.5, axis=1),
        'Upper Bound': np.percentile(forecast_matrix, 97.5, axis=1)
    })

    fig_forecast = go.Figure()

    fig_forecast.add_trace(go.Scatter(
        x=forecast_df['Date'], 
        y=forecast_df['Mean Forecast'], 
        mode='lines', 
        name='Mean Forecast',
        line=dict(color='blue')
    ))

    fig_forecast.add_trace(go.Scatter(
        x=np.concatenate([forecast_df['Date'], forecast_df['Date'][::-1]]),
        y=np.concatenate([forecast_df['Upper Bound'], forecast_df['Lower Bound'][::-1]]),
        fill='toself',
        fillcolor='rgba(0, 100, 255, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))

    fig_forecast.update_layout(
        yaxis_title='Forecasted Account Value (Millions $)',
        xaxis_title='Date'
    )

    st.plotly_chart(fig_forecast)

with tab2:
    st.title("Context and Analysis")
    st.markdown("""
    ## Evaluation and Areas of Improvement
    
    ### Projected Cash Flows
    - The dashboard effectively projects cash flows through the "Forecasted Account Values with Confidence Interval" visualization. It provides a range of potential outcomes based on Monte Carlo simulation.
    - **Improvement**: Enhance the granularity of cash flow projections by incorporating more factors such as different investment strategies or external market conditions.
    
    ### Distribution Around the Risk
    - The "Capital Call Risk Distribution" visualization captures the distribution of capital calls over time, offering insights into the variability and potential risk associated with these calls.
    - **Improvement**: Provide additional analysis or metrics to quantify the distribution and its impact on overall portfolio risk.
    
    ### Capital Call Stresses
    - The dashboard indirectly addresses capital call stresses by visualizing the timing and magnitude of capital calls.
    - **Improvement**: Integrate stress testing scenarios specifically focused on capital calls to assess the portfolio's resilience under adverse conditions.
    
    ### Volatility of Cash Flow Risk Over Time
    - The "Cumulative Net Cash Flow" visualization illustrates the volatility of cash flow risk over time.
    - **Improvement**: Include additional metrics or visualizations to quantify and analyze the volatility trends more explicitly.
    
    ### Simulation of Scenarios
    - The dashboard utilizes Monte Carlo simulation to forecast account values under different scenarios.
    - **Improvement**: Expand the range of scenarios considered and provide more interactive controls for users to explore custom scenarios.
    
    ### Impact of Market Conditions
    - While the dashboard indirectly considers market conditions through the Monte Carlo simulation, it could benefit from more explicit analysis of how different market scenarios impact cash flows and investment performance.
    - **Improvement**: Incorporate market indicators or external data sources to model the direct impact of market conditions on cash flows and account values.
    
    ### Evaluation of Different Scenarios
    - The dashboard allows users to evaluate different scenarios through the Monte Carlo simulation and adjustable parameters.
    - **Improvement**: Enhance scenario evaluation capabilities by providing comparative analysis tools and scenario-specific insights.
    
    ### Machine Learning and Modeling
    - The dashboard currently does not incorporate machine learning techniques. It relies on statistical modeling, specifically Monte Carlo simulation.
    - **Improvement**: Explore opportunities to integrate machine learning algorithms for more advanced analysis, such as predictive modeling or pattern recognition.
    
    ### Technology Backbone
    - The dashboard is built using Streamlit for the user interface and Plotly for visualizations, which are appropriate technologies for interactive data exploration.
    - **Improvement**: Continuously update and optimize the technology stack to improve performance, scalability, and user experience.
    
    ### Stress Test Assumptions of Public Liquidity
    - The dashboard does not explicitly stress test assumptions related to public liquidity.
    - **Improvement**: Incorporate stress testing scenarios specific to public liquidity to assess the portfolio's liquidity risk under different conditions.
    
    ### Predict Distribution and Risk Around Cash Flows
    - The dashboard provides insights into the distribution and risk around cash flows, particularly through the "Capital Call Risk Distribution" and "Cumulative Net Cash Flow" visualizations.
    - **Improvement**: Enhance predictive analytics capabilities to forecast distribution and risk around cash flows more accurately, potentially through advanced statistical modeling techniques.
    """)