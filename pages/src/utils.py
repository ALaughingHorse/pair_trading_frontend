import pickle
import numpy as np
import pandas as pd

class ExecutePairTrading:
    def __init__(self, abs_spread_mean, abs_spread_std, entry_signal, exit_signal):
        self.abs_spread_mean = abs_spread_mean
        self.abs_spread_std = abs_spread_std
        self.entry_signal=entry_signal
        self.exit_signal=exit_signal
        self.final_pl=0
        self.final_pl_pct=0

    def long_stock1_flag(self, stock1, stock2, idx):
        """
        This function is a ultility function to determine which stock to long/short given an entry signal.
        It:
            1. Takes the prices of two stocks, and the position where the entry signal appears 
            2. Calculate the percentage deltas between the current price and the price 7 days ago (or the earliest record) for each stock

        Then we will tell the ago to short the one with higher percentage delta and long the other.

        The function returns a boolean on whether we should long the stock 1.
        """
        stock1_current = stock1[idx]
        stock1_ref = stock1[max(0, idx-7)]

        stock2_current = stock2[idx]
        stock2_ref = stock2[max(0, idx-7)]

        pct_delta_1 = (stock1_current/stock1_ref) - 1
        pct_delta_2 = (stock2_current/stock2_ref) - 1

        if pct_delta_1 >= pct_delta_2:
            return False
        else:
            return True
        
    def execute(self, vec1, vec2, dates, beta_p1, beta_p2, base_fund=100, verbose=False):

        # Weight the spend using beta
        ticker1_spend_share = beta_p1/(beta_p1 + beta_p2)

        abs_spread = abs(vec1 - vec2)
        entry_thresh = self.abs_spread_mean + self.entry_signal*self.abs_spread_std
        exit_thresh = self.abs_spread_mean + self.exit_signal*self.abs_spread_std

        # get the positions where the entry/exit signals appears
        entry_signals = np.array([1 if abs_spread[i] >= entry_thresh else 0 for i in range(0, len(abs_spread))])
        exit_signals = np.array([1 if abs_spread[i] <= exit_thresh else 0 for i in range(0, len(abs_spread))])

        # Always exist on the last day
        exit_signals[len(abs_spread)-1] = 1
        
        signal_pairs = []
        has_entered = 0
        last_entered = 0
        for idx, entry_pos in enumerate(entry_signals):
            if has_entered == 0 and entry_pos == 0:
                continue # No entry
             
            if has_entered == 1 and exit_signals[idx] == 1:
                signal_pairs.append((last_entered, idx))
                has_entered = 0 # Exited
            elif has_entered == 0 and entry_signals[idx] == 1:
                last_entered = idx
                has_entered = 1 # Entered
           
        # Always exit on the last day if still in position
        if has_entered == 1:
            signal_pairs.append((last_entered, len(entry_signals)-1))

        self.final_pl = 0
        temp_tb = None
        if len(signal_pairs) > 0:
            # Create a dataframe to store the results
            temp_tb = pd.DataFrame(signal_pairs)
            temp_tb.columns = ['entry_idx', 'exit_idx']
            temp_tb = temp_tb.groupby('exit_idx').min().reset_index()
            # Make sure its ranked by entries
            temp_tb = temp_tb.sort_values('entry_idx',ascending=True).reset_index(drop=True)

            temp_tb['stock1_price_entry'] = vec1[temp_tb['entry_idx']] 
            temp_tb['stock1_price_exit'] = vec1[temp_tb['exit_idx']] 
            temp_tb['stock2_price_entry'] = vec2[temp_tb['entry_idx']] 
            temp_tb['stock2_price_exit'] = vec2[temp_tb['exit_idx']] 
            temp_tb['long_stock_1'] = [self.long_stock1_flag(vec1, vec2, x) for x in temp_tb.entry_idx]

            pnls = []
            both_long_short_profit = []
            for row in range(temp_tb.shape[0]):
                long_pnl=0
                short_pnl=0
                if temp_tb.long_stock_1[row]:
                    # calculate pnl when we long stock 1 and short stock 2
                    long_pnl = base_fund * ticker1_spend_share  * ((temp_tb.stock1_price_exit.values[row] - temp_tb.stock1_price_entry.values[row])/temp_tb.stock1_price_entry.values[row])
                    short_pnl = base_fund * (1-ticker1_spend_share ) * ((temp_tb.stock2_price_entry.values[row] - temp_tb.stock2_price_exit.values[row])/temp_tb.stock2_price_entry.values[row])
                else:
                    # calculate pnl when we long stock 2 and short stock 1
                    long_pnl = base_fund * (1-ticker1_spend_share ) * ((temp_tb.stock2_price_exit.values[row] - temp_tb.stock2_price_entry.values[row])/temp_tb.stock2_price_entry.values[row])
                    short_pnl = base_fund * (ticker1_spend_share ) * ((temp_tb.stock1_price_entry.values[row] - temp_tb.stock1_price_exit.values[row])/temp_tb.stock1_price_entry.values[row])
                pnls.append(long_pnl+short_pnl)
                
            temp_tb['pnl'] = pnls
            temp_tb['entry_dates'] = temp_tb.entry_idx.apply(lambda x: dates[x])
            temp_tb['exit_dates'] = temp_tb.exit_idx.apply(lambda x: dates[x])

            # temp_tb is sorted
            self.final_pl = temp_tb.pnl.sum()
        self.final_pl_pct = self.final_pl/base_fund
        self.execution_table = temp_tb
        
        return self
    
def run_simulation(starting_fund, sim_start, sim_end, transformed_data, refresh_cadence=60):
    # Determine the dates where we are refreshing the pair trading profolio
    unique_dates = transformed_data.Date[
         (transformed_data.Date >= sim_start) &
        (transformed_data.Date <= sim_end)
    ].unique()

    # total number of such dates
    total_date_num = len(unique_dates)
    idx_refresh_date = [refresh_cadence*i if (i*refresh_cadence)<=total_date_num else -1 for i in range(total_date_num)]
    idx_refresh_date = np.array(idx_refresh_date)[np.array(idx_refresh_date)>=0]
    refresh_dates = unique_dates[idx_refresh_date]

    # Get the data as of refresh dates
    refresh_date_data = transformed_data[
        transformed_data.Date.isin(refresh_dates)
    ]

    # Get the qualifying pairs for each date and the abs mean spread/std
    refresh_date_data_pairs = refresh_date_data[
        (refresh_date_data.entry_appears == 1) & 
        (refresh_date_data.stock2vec_cos_sim>0.95)
    ][
        ['Date', 'Ticker_P1', 'Ticker_P2', 'abs_spread_mean_MA','abs_spread_std_MA', 'beta_P1', 'beta_P2']
    ]
    current_fund = starting_fund
    future_daily_pnl_tb_agg = pd.DataFrame()
    all_execution_history = pd.DataFrame()
    for temp_date in refresh_dates:
        # filter the data to be each date when we refresh the strategy
        refresh_date_rec_pair_tb = refresh_date_data_pairs[refresh_date_data_pairs.Date==temp_date]
        num_paris_at_refresh_date = refresh_date_rec_pair_tb.shape[0]

        # Evenly distributed fund to each date
        fund_per_pair = current_fund/num_paris_at_refresh_date

        future_daily_pnl_tb = pd.DataFrame()

        for idx in refresh_date_rec_pair_tb.index:
            temp = refresh_date_rec_pair_tb.loc[idx]
            
            # Get future 60 trade date price data
            future_price=transformed_data[
                (transformed_data.Ticker_P1 == temp.Ticker_P1) & 
                (transformed_data.Ticker_P2 == temp.Ticker_P2) &
                (transformed_data.Date >= temp.Date)
            ]\
            .reset_index(drop=True)\
            .loc[:refresh_cadence, ['Date','Ticker_P1', 'Ticker_P2','Close_P1', 'Close_P2']]
        
            trade_class = ExecutePairTrading(
                abs_spread_mean=temp.abs_spread_mean_MA,
                abs_spread_std=temp.abs_spread_std_MA,
                entry_signal=2,
                exit_signal=0.5
            )
            cumu_pnl_pct = [0]
            for i in range(1,future_price.shape[0]):
                res = trade_class.execute(
                        vec1 = future_price.Close_P1.values[:i],
                        vec2 = future_price.Close_P2.values[:i],
                        dates= future_price.Date.values,
                        beta_p1=temp.beta_P1,
                        beta_p2=temp.beta_P2,
                        base_fund=fund_per_pair
                    )
                cumu_pnl_pct.append(
                    res.final_pl_pct
                )
                if res.execution_table is not None:
                    res.execution_table['ticker1'] = temp.Ticker_P1
                    res.execution_table['ticker2'] = temp.Ticker_P2
                    all_execution_history = pd.concat(
                        [
                            all_execution_history,
                            res.execution_table
                        ]
                    )
            future_price['cumu_pnl_pct'] = cumu_pnl_pct
            future_price['total_asset'] = fund_per_pair*(1+future_price['cumu_pnl_pct'])
            future_daily_pnl_tb = pd.concat(
                [future_daily_pnl_tb, future_price]
            )
            
        future_daily_pnl_tb_agg = pd.concat(
            [
                future_daily_pnl_tb_agg,
                future_daily_pnl_tb[[
                    'Date',
                    'total_asset'
                ]].groupby('Date').sum().reset_index()
            ]
        )
        # Update the money to invest for the next cycle
        current_fund = future_daily_pnl_tb_agg.total_asset.values[-1]

    future_daily_pnl_tb_agg = future_daily_pnl_tb_agg[
        future_daily_pnl_tb_agg.Date <= sim_end
    ]
    return future_daily_pnl_tb_agg, all_execution_history