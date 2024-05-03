#!/usr/bin/env python
# -*- coding: utf-8 -*-
import multiprocessing
import os
import pandas as pd
from conf import settings
from src import Option, Analysis
from itertools import combinations


def process_contract(id_prefix, same_contracts):
    """
    处理一个合约前缀及其对应的合约列表.

    Args:
        id_prefix (str): 合约编号前缀.
        same_contracts (dict): 包含合约列表的字典，键为'C'和'P'.

    Returns:
        str: 处理结果字符串.
    """

    call_contracts = same_contracts['C']
    put_contracts = same_contracts['P']

    if len(call_contracts) > 0 and len(put_contracts) > 0:

        # 生成所有可能的组合
        call_pairs = pair_elements(call_contracts)
        put_pairs = pair_elements(put_contracts)

        for contract_pair in call_pairs:

            bull_portfolio_df, strategy = Analysis.bull_spread(contract_pair, id_prefix)
            if isinstance(bull_portfolio_df, pd.DataFrame):
                bull_PNL_df = pd.concat([bull_portfolio_df], ignore_index=True)
                save_strategy_pal(bull_PNL_df, strategy)
            else:
                print(f"{strategy} exist!")
                continue

            bear_portfolio_df, strategy = Analysis.bear_spread(contract_pair, id_prefix)
            if isinstance(bull_portfolio_df, pd.DataFrame):
                bear_PNL_df = pd.concat([bear_portfolio_df], ignore_index=True)
                save_strategy_pal(bear_PNL_df, strategy)
            else:
                print(f"{strategy} exist!")
                continue

        for contract_pair in put_pairs:

            bull_portfolio_df, strategy = Analysis.bull_spread(contract_pair, id_prefix)
            if isinstance(bull_portfolio_df, pd.DataFrame):
                bull_PNL_df = pd.concat([bull_portfolio_df], ignore_index=True)
                save_strategy_pal(bull_PNL_df, strategy)
            else:
                print(f"{strategy} exist!")
                continue

            bear_portfolio_df, strategy = Analysis.bear_spread(contract_pair, id_prefix)
            if isinstance(bull_portfolio_df, pd.DataFrame):
                bear_PNL_df = pd.concat([bear_portfolio_df], ignore_index=True)
                save_strategy_pal(bear_PNL_df, strategy)
            else:
                print(f"{strategy} exist!")
                continue
    else:
        raise ValueError("No contracts available")


def save_strategy_pal(PNL_df, strategy, file_path='../db/result/strategies'):
    filename = f"{file_path}/{strategy}.xlsx"
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        PNL_df.to_excel(writer, sheet_name=f'{strategy}', index=False)


def pair_elements(lst):
    pairs = list(combinations(lst, 2))
    return pairs


def find_contracts_with_same_id(contracts):
    """
    找出具有相同合约编号前缀和合约类型的合约，并返回一个字典，其中键是合约编号前缀，值是另一个字典，
    其中键是合约类型( Call 或 Put )，值是具有相同前缀和类型的合约列表。

    Args:
    contracts (list): 合约列表。

    Returns:
    dict: 键是合约编号前缀，值是另一个字典，其键是合约类型，值是具有相同前缀和类型的合约列表。
    """
    contracts_dict = {}
    for option in contracts:
        contract_code = option.contract_code
        id_prefix, option_type, _ = contract_code.split('-')
        if id_prefix not in contracts_dict:
            contracts_dict[id_prefix] = {'C': [], 'P': []}
        contracts_dict[id_prefix][option_type].append(option)
    return contracts_dict


def cat_con_and_ul(contract, underlying):
    """
    匹配相同日期的行, 合并两个 DataFrame, 传回一个合成的 DataFrame.

    Args:
        contract (DataFrame): 合约交易数据表.
        underlying (DataFrame): 标的资产交易数据表.

    Returns:
        DataFrame: 合成数据表.
    """
    # 
    contract['date'] = pd.to_datetime(contract['date'])
    underlying['date'] = pd.to_datetime(underlying['date'])
    merged_df = pd.merge(contract, underlying, on='date')
    merged_df = merged_df.sort_values(by='date')
    merged_df = merged_df.reset_index(drop=True)
    return merged_df


def load_data(file_path, dataset_name):
    data_files_path = os.path.join(file_path, dataset_name)
    all_data = []

    for file_name in os.listdir(data_files_path):
        if file_name.endswith(".csv"):
            file = os.path.join(data_files_path, file_name)
            df = pd.read_csv(file)
            all_data.append(df)
    return all_data


def implied_contract(contract: pd.DataFrame, underlying: pd.DataFrame, operation: str) -> object:
    df = cat_con_and_ul(contract, underlying)
    option = Option.Option(
        contract_code=df["合约代码"][0],
        operation=operation,
        option_price=df["今收盘"],
        underlying_price=df["close"],
        date=df["date"],
        end_date=df["date"].iloc[-1],
        strike_price=df["K"].iloc[0],
        risk_free_rate=None,
        implied_volatility=None,
        tau=None,
        Time_to_maturity=None
    )
    return option


def load_option(operation='Buy', config_path="../conf/config.json"):
    """
    Load default config

    Args:
        operation (str, optional): The direction of contract. Defaults to 'Buy'.
        config_path (str, optional): The file path of config. Defaults to "conf/config.json".

    Returns:
        list(Option): List of options in the default data path.
    """

    prefix = None
    if config_path == "../conf/config.json":
        prefix = "../"
    op_list = []
    settings.build_config(config_path)
    config = settings.load_config(config_path)
    option_path = config['datasets']
    underlying_data_path = config['underlying']
    underlying_index = config['underlying_index']
    if prefix:
        option_path = prefix + option_path
        underlying_data_path = prefix + underlying_data_path

    files = []
    for file_name in os.listdir(option_path):
        files.append(file_name)
    datasets = files

    for key, value in underlying_index.items():
        if key in datasets:
            Option_data = load_data(option_path, key)
            underlying = pd.read_csv(underlying_data_path + '/' + value + '.csv', )
            for contract in Option_data:
                option = implied_contract(contract, underlying, operation)
                op_list.append(option)

    return op_list


def run():
    contracts_dict = find_contracts_with_same_id(load_option())

    # 创建进程池, 默认调用 10 个进程, 数量根据需要调整
    pool = multiprocessing.Pool(processes=16)
    pool.starmap(process_contract, contracts_dict.items())

    pool.close()
    pool.join()
