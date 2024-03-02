import streamlit as st
import time
import numpy as np
import pickle
import plotly.express as px
from streamlit import session_state as ss

# load data
with open('Data/spotcheckout_output.pkl','rb') as file:
    data = pickle.load(file)

st.set_page_config(page_title="Time Machine", page_icon="📈")
st.markdown("# Time Machine")
st.sidebar.header("Time Machine")
st.write(
    """Wanna know the performance of our model? Pick a date in history and examine the performance!"""
)

## Initialize session states
if 'pair_found_flag' not in ss:
    ss['pair_found_flag'] = False


target_date = st.selectbox(
    label = 'Pick a date (at least 120 trading days before today)',
    options = data.dropna().Date.sort_values(ascending=False).unique()
)

find_pair_button = st.button(
    label = 'Find pair trading strategies'
)

if find_pair_button:
    # find the pair with highest predicted pnl
    filtered = data[data['Date']==target_date].sort_values('pnls', ascending=False)[['Ticker_P1', 'Ticker_P2']].head(1)

    # get the tickers
    ticker1=filtered.Ticker_P1.values[0]
    ticker2=filtered.Ticker_P2.values[0]
    # log into session states
    ss['ticker1'] = ticker1
    ss['ticker2'] = ticker2

    # print the message
    pairs = st.markdown(f'High profit potential pairs on **{target_date}**: **{ticker1}** and **{ticker2}**')

    # Get the historical data
    filtered_tb = data[(data.Ticker_P1==ticker1) & (data.Ticker_P2==ticker2)].reset_index(drop=True)
    target_date_idx = np.where(filtered_tb.Date==target_date)[0][0]

    historical_data = filtered_tb[(target_date_idx-500):(target_date_idx+1)]
    historical_fig = px.line(
        historical_data, 
        x="Date", 
        y=['Close_P1','Close_P2', 'abs_spread'],
        title=f'Past 500 Closing Price for {ticker1} and {ticker2} as of {target_date}'
    )

    # Display the recommended trading strategy
    abs_spread_mean_on_target_date = filtered_tb.loc[target_date_idx].abs_spread_mean
    abs_spread_std_on_target_date = filtered_tb.loc[target_date_idx].abs_spread_std
    entry_spread=abs_spread_mean_on_target_date + 1.5*abs_spread_std_on_target_date 
    st.text(
        f"""
        Strategy:
        Enter long and short positions when the absolute spread of the this two stocks is above ~${round(entry_spread,2)}
        """
    )
    st.plotly_chart(historical_fig)

    # Update the session states
    ss['filtered_tb'] = filtered_tb
    ss['target_date_idx'] = target_date_idx 
    ss['pair_found_flag'] = True

if ss.pair_found_flag:
    target_date_idx = ss['target_date_idx']
    filtered_tb = ss['filtered_tb']

    test_data = filtered_tb[(target_date_idx+1):(target_date_idx+121)][[
        'Date','Close_P1','Close_P2', 'abs_spread',
    ]].reset_index(drop=True)

    print(filtered_tb.loc[target_date_idx].pnls)    
    trading_strategy_tb = filtered_tb.loc[target_date_idx].trade_executions

    # historical_fig = px.line(
    #     historical_data, 
    #     x="Date", 
    #     y=['Close_P1','Close_P2', 'abs_spread'],
    #     title=f'Past 500 Closing Price for {ticker1} and {ticker2} as of {target_date}'
    # )

    testing_button = st.button(
            label = 'Execute on this pair'
        )
    if testing_button:
        st.text('Executing')
        output_description = ""
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        for i in range(0, test_data.shape[0]):
            status_text.text(f"{round(i/test_data.shape[0],2)*100}% Complete")
            progress_bar.progress((i/test_data.shape[0]))
            time.sleep(0.05)

        test_data_sub_col = test_data[['Date','Close_P1','Close_P2', 'abs_spread']]
        test_data_sub_col.columns = ['Date',ss['ticker1'],ss['ticker2'], 'Absolute Spread']
        test_fig = px.line(
                test_data_sub_col, 
                x="Date", 
                y=[ss['ticker1'],ss['ticker2']]
                # title=f'Past 500 Closing Price for {ticker1} and {ticker2} as of {target_date}'
            )
        
        for idx in range((trading_strategy_tb.shape[0])):
            test_fig.add_vline(x=trading_strategy_tb.entry_dates.values[idx], line_width=3, line_dash="dash", line_color="green")
            test_fig.add_vline(x=trading_strategy_tb.exit_dates.values[idx], line_width=3, line_dash="dash", line_color="red")

            # Construct the messages
            output_description+=f"Entered positions on {trading_strategy_tb.entry_dates.values[idx]} \n"
            if trading_strategy_tb.long_stock_1.values[idx]:
                output_description+=f"Longing {ss['ticker1']} and shoring {ss['ticker2']} \n"
            else:
                output_description+=f"Longing {ss['ticker2']} and shoring {ss['ticker1']} \n"

            output_description+=f"Closed positions on {trading_strategy_tb.exit_dates.values[idx]} \n"
            output_description+=f"Total PnL from this trade: {round(trading_strategy_tb.pnl.values[idx],2)}% \n"
            output_description+='--------- \n'
        st.text(output_description)
        st.plotly_chart(test_fig)
        progress_bar.empty()



    