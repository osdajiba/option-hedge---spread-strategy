import datetime
import os
import warnings
import akshare as ak
import pandas as pd


def handle_data_type(data):
    if isinstance(data, list):
        return pd.concat(data, ignore_index=True)
    elif isinstance(data, dict):
        # 将字典的值转换为 DataFrame，并进行连接
        return pd.concat([pd.DataFrame(value) for value in data.values()], ignore_index=True)
    elif isinstance(data, pd.DataFrame):
        return data
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def save_data(data_name, ak_path, data):
    # try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 忽略所有警告
    data = handle_data_type(data)
    data.to_csv(os.path.join(ak_path, f"{data_name}.csv"))  # 保存每个 DataFrame 到 CSV 文件
    print(f"data has saved to file: {ak_path}/{data_name}")


# except Exception as e:
#     print(f"An error occurred: {e}")
Option_underlying_Index = {
    "IO": "沪深300",
    "MO": "中证1000",
    "HO": "上证50"
}


def test():
    option_data_name = "option_value_analysis_em_df"
    ak_path = "../db/akdata"
    type_method = "date"
    start_day = "20240101"
    end_day = "20240401"

    option_sse_underlying_spot_price_sina = ak.option_sse_underlying_spot_price_sina()
    save_data(option_data_name, ak_path, option_sse_underlying_spot_price_sina)

    stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20210101", end_date='20240401',
                                            adjust="")
    print(stock_zh_a_hist_df)

    for key, value in Option_underlying_Index.items():
        option_data_name = key + "_Underlying"
        stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=value, period="daily", start_date=start_day, end_date=end_day,
                                                adjust="")
        save_data(option_data_name, ak_path, stock_zh_a_hist_df)
        print(stock_zh_a_hist_df)


"option_finance_board"  # 获取金融期权数据
"option_cffex_sz50_list_sina"  # 上证50期权列表
"option_cffex_sz50_spot_sina"  # 沪深300期权实时行情
"option_cffex_sz50_daily_sina"  # 沪深300期权历史行情-日频
"option_cffex_hs300_list_sina"  # 沪深300期权列表
"option_cffex_hs300_spot_sina"  # 沪深300期权实时行情
"option_cffex_hs300_daily_sina"  # 沪深300期权历史行情-日频
"option_cffex_zz1000_list_sina"  # 中证1000期权列表
"option_cffex_zz1000_spot_sina"  # 中证1000期权实时行情
"option_cffex_zz1000_daily_sina"  # 中证1000期权历史行情-日频
"option_sse_list_sina"  # 上交所期权列表
"option_sse_expire_day_sina"  # 上交所期权剩余到期日
"option_sse_codes_sina"  # 上交所期权代码
"option_sse_spot_price_sina"  # 上交所期权实时行情
"option_sse_underlying_spot_price_sina"  # 上交所期权标的物实时行情
"option_sse_greeks_sina"  # 上交所期权希腊字母
"option_sse_minute_sina"  # 上交所期权分钟数据
"option_sse_daily_sina"  # 上交所期权日频数据
"option_finance_minute_sina"  # 金融股票期权分时数据
"option_minute_em"  # 股票期权分时数据
