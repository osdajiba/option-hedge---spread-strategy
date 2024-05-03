#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from src import Option
import pandas as pd


def merge_data(long_df, short_df, selected_columns):
    def subtract_columns(df, suffixes):
        for suffix in suffixes:
            cols = [f'{suffix}_x', f'{suffix}_y']
            df[suffix] = df[cols[0]] - df[cols[1]]
        df.drop([f'{suffix}_x' for suffix in suffixes] + [f'{suffix}_y' for suffix in suffixes], axis=1, inplace=True)

    def sum_columns(df, suffixes):
        for suffix in suffixes:
            cols = [f'{suffix}_x', f'{suffix}_y']
            df[suffix] = df[cols].sum(axis=1)
        df.drop([f'{suffix}_x' for suffix in suffixes] + [f'{suffix}_y' for suffix in suffixes], axis=1, inplace=True)

    long_df = long_df[selected_columns]
    short_df = short_df[selected_columns]
    merged_df = pd.merge(long_df, short_df, on='date', how='outer')

    # 合并 contract_code
    merged_df['contract_code'] = merged_df['contract_code_x'] + '-' + merged_df['contract_code_y'].str.split('-').str[
        -1]

    # 合并希腊字母
    sub_suffixes = ['DELTA', 'THETA', 'VEGA', 'RHO']
    sum_suffixes = ['GAMMA', 'exact_return']
    sum_columns(merged_df, sum_suffixes)
    subtract_columns(merged_df, sub_suffixes)
    # 使用均值合并波动率
    merged_df['volatility'] = merged_df[['volatility_x', 'volatility_y']].mean(axis=1)

    # 删除带有后缀的列
    merged_df.drop(['contract_code_x', 'volatility_x', 'contract_code_y', 'volatility_y'], axis=1, inplace=True)

    # 重新排序列
    merged_df = merged_df[selected_columns]
    merged_df = merged_df.sort_values(by='date')
    return merged_df


def bull_spread(contract_pair, id_prefix):
    """
    牛市看涨期权套利：买进一个执行价格较低的看涨期权, 同时卖出一个到期日相同、但执行价格较高的看涨期权，以利用两种期权之间的价差波动寻求获利，要求先付权利金，更适合波动率低的情况。

    牛市看跌期权套利：买进一个执行价格较低的看跌期权, 同时operation,output_option期日相同、但执行价格较高的看跌期权, 可以先收权利金, 适合波动率高的情况。
    """

    long_contract, short_contract = contract_pair
    long_contract.operation = "buy"
    short_contract.operation = "sell"

    if "-C-" in long_contract.contract_code:
        strategy = (
            f"Bull-{id_prefix}-C-{long_contract.contract_code.split('-')[-1]}-{short_contract.contract_code.split('-')[-1]}")
    elif "-P-" in long_contract.contract_code:
        strategy = (
            f"Bull-{id_prefix}-P-{long_contract.contract_code.split('-')[-1]}-{short_contract.contract_code.split('-')[-1]}")
    else:
        raise ValueError("Contracts type error")

    file_path = '../db/result/strategies'
    filename = f"{file_path}/{strategy}.xlsx"
    if os.path.exists(filename):
        return -1, strategy

    long_df = align_contract(long_contract)
    short_df = align_contract(short_contract)

    selected_columns = ['date', 'contract_code', 'volatility', 'DELTA', 'THETA', 'GAMMA', 'VEGA', 'RHO', 'exact_return']
    merged_df = merge_data(long_df, short_df, selected_columns)
    return merged_df, strategy


def bear_spread(option_pairs, id_prefix):
    """
    Bear Spread strategy: Buy a higher strike put option and sell a lower strike put option with the same expiration
    date to profit from the difference in their prices. Requires paying the premium upfront, more suitable for
    low volatility.

    Bear Call Spread strategy: Buy a higher strike call option and sell a lower strike call option with the same
    expiration date. Allows receiving the premium upfront, more suitable for high volatility.
    """

    short_contract, long_contract = option_pairs
    long_contract.operation = "buy"
    short_contract.operation = "sell"

    if "-P-" in long_contract.contract_code:
        strategy = f"Bear-{id_prefix}-P-{long_contract.contract_code.split('-')[-1]}-{short_contract.contract_code.split('-')[-1]}"
    elif "-C-" in long_contract.contract_code:
        strategy = f"Bear-{id_prefix}-C-{long_contract.contract_code.split('-')[-1]}-{short_contract.contract_code.split('-')[-1]}"
    else:
        raise ValueError("Contract type error")

    file_path = '../db/result/strategies'
    filename = f"{file_path}/{strategy}.xlsx"
    if os.path.exists(filename):
        return -1, strategy

    long_df = align_contract(long_contract)
    short_df = align_contract(short_contract)

    selected_columns = ['date', 'contract_code', 'volatility', 'DELTA', 'THETA', 'GAMMA', 'VEGA', 'RHO', 'exact_return']
    merged_df = merge_data(long_df, short_df, selected_columns)
    return merged_df, strategy


def align_contract(input_option):
    df = pd.DataFrame.from_dict(input_option._args())
    option_df = pd.DataFrame()

    for index in range(len(df)):
        new_option = Option.Option(
            contract_code=df.iloc[index]['contract_code'],
            option_price=df.iloc[index]['option_price'],
            underlying_price=df.iloc[index]['underlying_price'],
            strike_price=df.iloc[index]['strike_price'],
            risk_free_rate=0.02,
            operation=input_option.operation,
            volatility=df.iloc[index]['volatility'],
            implied_volatility=None,
            date=df.iloc[index]['date'].strftime('%Y-%m-%d'),
            Time_to_maturity=float((df.iloc[index]['end_date'] - df.iloc[index]['date']).days),
            tau=float((df.iloc[index]['end_date'] - df.iloc[index]['date']).days / 365.0),
        )
        new_option._calculate_black_scholes_price()
        new_option.calculate_greeks()
        option_df = pd.concat([option_df, pd.DataFrame([new_option.__dict__])], ignore_index=True)
        option_price = new_option.option_price

        a = 1 if input_option.operation == 'buy' else -1 if input_option.operation == 'sell' else None
        if a is None:
            raise ValueError('operation should be buy or sell')
        option_df['bs_expected_return'] = a * (new_option.calculate_price() - option_price)
        option_df['volatility'] = new_option.implied_volatility

    option_df['exact_return'] = a * (df['option_price'] - df['option_price'].iloc[-1])

    return option_df


def asset_table(contract):
    """
    构建资产表.

    Args:
        contract (Option): 合约类.
    
    Returns:
        df (DataFrame): 包含合约信息的表.
    """
    option_price = contract.option_price
    expected_return = (contract.calculate_price() - option_price) / option_price
    implied_volatility = contract.implied_volatility
    greeks = contract.calculate_greeks()
    data = {
        'Current Expected Return': [expected_return],
        'Implied Volatility': [implied_volatility],
        'Delta': [greeks['delta']],
        'Gamma': [greeks['gamma']],
        'Theta': [greeks['theta']],
        'Vega': [greeks['vega']],
        'Rho': [greeks['rho']],
    }

    # 指定行名称
    index = ['Metrics']

    df = pd.DataFrame(data, index=index)
    return df
