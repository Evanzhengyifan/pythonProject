import pandas as pd
import numpy as np
import h5py
from rqalpha_plus.apis import *
from rqalpha_plus import run_func
import rqdatac as rq
import os
import sys
import datetime
import time
import re
from joblib import Parallel, delayed
from functools import reduce
import warnings

warnings.filterwarnings("ignore")


def adjust(n: int, k: list or dict, start: str, end: str,
           trade_freq: int, symbol_freq: int, n_largest: int, n_std: int, n_bias: int):
    # 回测报告存放路径
    path = os.path.join(os.getcwd(), f'Overnight_{start}_{end}_{k}') if isinstance(k, list) else os.path.join(
        os.getcwd(), f'Overnight_{start}_{end}')
    path = os.path.join(path, '测试不同分钟调仓不同频率更新品种')
    path = os.path.join(path, f"每{trade_freq}分钟调仓每{symbol_freq}天更新品种取前{n_largest}（BIAS_{n_bias}_{n_std}_std)")
    if not os.path.exists(path):
        os.makedirs(path)
    config = {
        "base": {
            "accounts": {
                "future": 5000000,
            },
            "frequency": "1m",
            "start_date": start,
            "end_date": end,
        },
        'mod': {
            # 模拟撮合模块
            "sys_simulation": {
                "matching_type": "current_bar",
                "volume_limit": False
            },
            "sys_analyser": {
                "plot": False,
                "plot_save_file": os.path.join(path, '净值图.png'),
                "report_save_path": path
            }
        }
    }

    def init(context):

        # 1.订阅行情
        all_sub_item = rq.all_instruments(type='Future', date=context.now)["underlying_symbol"].tolist()
        all_sub_item = set(all_sub_item) - {'IF', 'IH', 'IC', 'IM', 'TF', 'TS', 'T', 'SI'}
        all_sublist = [i + "889" for i in all_sub_item]
        for i in all_sublist:
            subscribe(i)

        # 2.参数 （窗口 上下轨系数）
        context.n = n
        context.k = k
        context.n_std = n_std
        context.n_bias = n_bias
        context.trade_count = 0
        context.trade_freq = trade_freq
        context.symbol_count = 0
        context.symbol_freq = symbol_freq
        context.n_largest = n_largest
        # 隔夜策略暂时用不到这个参数
        context.check_time = "14:00:00"
        # 3.尝试不同分钟使用计数器，以及不同频率更换品种
        context.signal_of_index = True
        # 4.其他初始化
        # 存放回测日期已经上市的品种
        context.pass_items = set()
        # 夜盘品种
        context.night_item = []
        # 目标持仓
        context.AllItem_contract = {}  # 存放计算后的目标持仓信息，key是品种，value是合约
        context.target_pos = {}  # 存放计算后的目标持仓信息，key是品种，value是合约加多空头仓位的字典
        context.AllItem_Range = {}  # 存放计算后目标品种的range值
        context.AllItem_buy = {}  # 存放计算后目标品种的上轨
        context.AllItem_sell = {}  # 存放计算后目标品种的下轨
        context.AllItem_hold = {}  # 存放当前持仓信息，key是品种，value是合约
        context.hold_item = {}  # 存放当前持仓信息，key是品种，value是合约加多空头仓位的字典
        context.trading_time = {}
        logger.info("RunInfo: {}".format(context.run_info))

    def dual_thrust(data: pd.DataFrame, window: int) -> pd.DataFrame:
        """计算DT策略的上下轨和range"""
        data["HC"] = data["close"].rolling(window).max()
        data["HH"] = data["high"].rolling(window).max()
        data["LC"] = data["close"].rolling(window).min()
        data["LL"] = data["low"].rolling(window).min()
        data["range1"] = data["HH"] - data["LC"]
        data["range2"] = data["HC"] - data["LL"]
        data['Range'] = np.where(data["range1"] > data["range2"], data["range1"], data["range2"])
        return data

    def handle_bar(context, bar_dict):
        # 1.区分夜盘和日盘品种
        if context.now.time() == datetime.datetime.strptime("21:05:00", "%H:%M:%S").time():
            if context.pass_items:
                context.night_item = [item for item in context.pass_items if
                                      '21' in rq.get_trading_hours(context.AllItem_contract[item], context.now)]
                # logger.info("【夜盘品种{0}】".format(context.night_item))
        # 30分钟代表每半小时进行一次交易
        if context.signal_of_index and context.trade_count % context.trade_freq == 0:
            context.trade_count = 0
            # 在夜盘和日盘的交易时间段都更新仓位信息至context.AllItem_hold和context.hold_item
            if is_night(context) | is_day(context):
                cal_positions(context)

            # 2.夜盘和日盘计算交易
            for item in context.pass_items:
                if is_night(context):
                    if item in context.night_item:
                        if is_trading_time(context, item):
                            trade(context, item)
                elif is_day(context):
                    trade(context, item)

            for item in set(context.AllItem_hold.keys()) - context.pass_items:
                key = context.AllItem_hold[item]
                if context.hold_item[item][key]['longpos'] > 0 or context.hold_item[item][key]['shortpos'] > 0:
                    order_to(context.AllItem_hold[item], 0)
                    logger.info(f"{item}没有入选交易品种，因此全部平仓")
                del context.AllItem_hold[item]
                del context.hold_item[item]

        if context.now.time() == datetime.datetime.strptime(context.check_time, "%H:%M:%S").time():
            for item in context.pass_items:
                adjust(context, item)
        context.trade_count += 1

    def after_trading(context):
        # logger.info(f"after_trading: {context.now}")
        # 1.日期和时间
        context.Pre_day = rq.get_previous_trading_date(context.now, 1)
        context.next_date = rq.get_next_trading_date(context.now, n=1)
        # 每天更新一次交易品种
        if context.symbol_count % context.symbol_freq == 0:
            context.symbol_count = 0
            context.items = get_items(context) if context.n_largest != 0 else ['RB', 'JM', 'I', 'CU', 'AL', 'NI', 'TA', 'PP']
        context.symbol_count += 1
        # 2.判断是否上市
        all_pass_items = [i.upper() for i in context.items]
        no_list = [i for i in all_pass_items if rq.futures.get_dominant(i, context.now) is None]
        context.pass_items = set(all_pass_items) - set(no_list)
        # logger.info(f'已上市品种{context.pass_items}')
        if no_list:
            logger.info("【{0}该品种{1}未上市】".format(context.now.date().strftime('%Y-%m-%d'), no_list))

        # 3.各品种等权分配资金
        context.single_value = context.portfolio.total_value / len(context.pass_items)

        # 4.获取当日持仓
        context.AllItem_contract = {}  # 存放计算后的目标持仓信息，key是品种，value是合约
        context.target_pos = {}  # 存放计算后的目标持仓信息，key是品种，value是合约加多空头仓位的字典
        context.AllItem_Range = {}  # 存放计算后目标品种的range值
        context.AllItem_buy = {}  # 存放计算后目标品种的上轨
        context.AllItem_sell = {}  # 存放计算后目标品种的下轨
        context.AllItem_hold = {}  # 存放当前持仓信息，key是品种，value是合约
        context.hold_item = {}  # 存放当前持仓信息，key是品种，value是合约加多空头仓位的字典
        context.trading_time = {}
        context.signal_of_index = True

        # 当日结算仓位
        cal_positions(context)
        # 当前上市品种中没有持仓的品种，也记录其合约以及持仓
        for item in context.pass_items - set(context.hold_item.keys()):
            i_main = rq.futures.get_dominant(item, context.now).values[0]
            context.AllItem_hold[item] = i_main
            context.hold_item[item] = {i_main: {"longpos": 0, "shortpos": 0}}
        # logger.info("持仓{0}".format(context.hold_item['RB']))
        # 5.计算上下轨 计算目标仓位
        for item in context.pass_items:
            # 设置品种对应的主力合约
            main = rq.futures.get_dominant(item, context.now).values[0]
            context.AllItem_contract[item] = main
            item_bar = rq.get_price(main, start_date=context.Pre_day, end_date=context.now, frequency='120m',
                                    fields=['open', 'close', 'low', 'high'])
            result = dual_thrust(item_bar, context.n)

            context.AllItem_Range[item] = result["Range"].values[-1]
            try:
                # 回测中姑且在after_trading就获取下一日的开盘价，来计算上下轨，是盘中不能这么写
                today_open = rq.get_price(main, start_date=context.next_date, end_date=context.next_date,
                                          fields='open')['open'].values[0]

                if isinstance(context.k, list):
                    context.AllItem_buy[item] = today_open + (float(context.k[0]) * context.AllItem_Range[item])
                    context.AllItem_sell[item] = today_open + (float(context.k[1]) * context.AllItem_Range[item])
                elif isinstance(context.k, dict):
                    if len(list(context.k.keys())[0]) == 6:
                        date_month_now = context.now.strftime("%Y%m")
                        context.AllItem_buy[item] = today_open + (
                                context.k[date_month_now]['k1'] * context.AllItem_Range[item])
                        context.AllItem_sell[item] = today_open + (
                                context.k[date_month_now]['k2'] * context.AllItem_Range[item])
                    elif len(list(context.k.keys())[0]) == 4:
                        date_year_now = context.now.strftime("%Y")
                        context.AllItem_buy[item] = today_open + (
                                context.k[date_year_now]['k1'] * context.AllItem_Range[item])
                        context.AllItem_sell[item] = today_open + (
                                context.k[date_year_now]['k2'] * context.AllItem_Range[item])
                else:
                    pass

                long_pos = int(
                    context.single_value / rq.instruments(item + "889").contract_multiplier / context.AllItem_buy[item])
                short_pos = int(
                    context.single_value / rq.instruments(item + "889").contract_multiplier / context.AllItem_sell[
                        item])
                context.target_pos[item] = {main: {"longpos": long_pos, "shortpos": short_pos}}
            except (TypeError, KeyError, ValueError):
                print('遇到报错直接跳过')
            context.trading_time.update({item: rq.get_trading_hours(item + '889', context.now)})

        if cal_index(context):
            context.signal_of_index = False

        # 夜盘品种
        context.night_item = []
        context.trade_count = 0

    def cal_index(context):
        pre_day = rq.get_previous_trading_date(context.now, 20)
        df = rq.get_price('H11061.XSHG', start_date=pre_day, end_date=context.now, fields='close')
        df = df.reset_index(level=0, drop=True)
        for i in [5, 10, 20]:
            df[f'MA_{i}'] = df['close'].rolling(i).mean()
            df[f'BIAS_{i}'] = (df['close'] - df[f'MA_{i}']) / df[f'MA_{i}'] * 100
            df[f'BIAS_{i}_top'] = df[f'BIAS_{i}'].mean() + context.n_std * df[f'BIAS_{i}'].std()
            df[f'BIAS_{i}_bottom'] = df[f'BIAS_{i}'].mean() - context.n_std * df[f'BIAS_{i}'].std()
        signal_list = (df[f'BIAS_{context.n_bias}'] > df[f'BIAS_{context.n_bias}_top']) | \
                      (df[f'BIAS_{context.n_bias}'] < df[f'BIAS_{context.n_bias}_bottom'])
        return signal_list.to_numpy()[-1]

    def get_items(context):
        # 获取当前回测日期的所有期货合约
        data = rq.all_instruments(type='Future', date=context.now)[['underlying_symbol', 'listed_date']]
        # 剔除上市时间等于'0000-00-00'， 之后再根据品种获得该品种下合约的最早的上市时间
        data = data.query("listed_date != '0000-00-00'").groupby('underlying_symbol')['listed_date'].min()
        # 将data['listed_date']转成datetime格式的数据，方便后续与context.now直接相减
        data = pd.to_datetime(data)
        # 筛选得到上市满半年的品种，半年按照180天计算
        underlying_symbol = data[(context.now - data).dt.days > 180].index.tolist()
        # 剔除股指期货{'IF', 'IH', 'IC'}和国债期货{'TF', 'TS', 'T'}
        underlying_symbol = set(underlying_symbol) - {'IF', 'IH', 'IC', 'TF', 'TS', 'T', 'IM'}
        # 构造筛选后所有品种的主力连续合约
        book_id = [i + '889' for i in underlying_symbol]
        # 获得所有主力连续合约前20个交易日的交易量数据
        group_vol = rq.get_price(order_book_ids=book_id, start_date=get_previous_trading_date(date=context.now, n=20),
                                 end_date=context.now, fields='volume', adjust_type='pre_volume')
        # 按照合约名称对交易量数据求均值，再筛选交易量大于10000的合约，再取前K个
        if isinstance(context.k, list):
            top_book_id = group_vol.groupby(level=0).median().query('volume>10000').nlargest(context.k[-1],
                                                                                             'volume').index.tolist()
            # group_vol = rq.get_price(order_book_ids=top_book_id,
            #                               start_date=get_previous_trading_date(date=context.now, n=20),
            #                               end_date=context.now, fields='close')
            # top_book_id = group_vol.groupby(level=0).std().nlargest(10, 'close').index.tolist()
        elif isinstance(context.k, dict):
            if len(list(context.k.keys())[0]) == 6:
                date_month_now = context.now.strftime("%Y%m")
                top_book_id = group_vol.groupby(level=0).median().query('volume>10000').nlargest(
                    int(context.k[date_month_now]['tp']), 'volume').index.tolist()
            elif len(list(context.k.keys())[0]) == 4:
                date_year_now = context.now.strftime("%Y")
                top_book_id = group_vol.groupby(level=0).median().query('volume>10000').nlargest(
                    int(context.k[date_year_now]['tp']), 'volume').index.tolist()
            else:
                top_book_id = group_vol.groupby(level=0).median().query('volume>10000').index.tolist()
        else:
            top_book_id = group_vol.groupby(level=0).median().query('volume>10000').index.tolist()
        return [i[:-3] for i in top_book_id]

    def cal_positions(context):
        """结算仓位，记录到context.AllItem_hold和context.hold_item中"""
        try:
            if get_positions():
                for pos in get_positions():
                    i = re.sub(r"\d+", "", pos.order_book_id)
                    long_hold = context.portfolio.positions[pos.order_book_id].buy_quantity
                    short_hold = context.portfolio.positions[pos.order_book_id].sell_quantity
                    context.AllItem_hold[i] = pos.order_book_id
                    context.hold_item[i] = {pos.order_book_id: {"longpos": long_hold, "shortpos": short_hold}}
        except RuntimeError:
            print('遇到价格不合法')

    def is_trading_time(context, item):
        """判断是否处于该品种的夜盘交易时间"""
        night_time = context.trading_time[item].split(',')[0].split('-')[-1]
        night_begin = context.now.time() > datetime.datetime.strptime("21:05", "%H:%M").time()
        night_end = context.now.time() < datetime.datetime.strptime(night_time, "%H:%M").time()
        if datetime.datetime.strptime(night_time, "%H:%M").time() < datetime.datetime.strptime("03:00", "%H:%M").time():
            return night_begin | night_end
        else:
            return night_begin & night_end

    def is_night(context):
        """判断是否处于夜盘交易时间"""
        night_begin = context.now.time() > datetime.datetime.strptime("21:05:00", "%H:%M:%S").time()
        night_end = context.now.time() < datetime.datetime.strptime("03:00:00", "%H:%M:%S").time()
        return night_begin | night_end

    def is_day(context):
        """判断是否处于日盘交易时间"""
        day_begin = context.now.time() > datetime.datetime.strptime("09:00:00", "%H:%M:%S").time()
        before_noon_end = context.now.time() < datetime.datetime.strptime("10:15:00", "%H:%M:%S").time()
        before_noon_beign = context.now.time() > datetime.datetime.strptime("10:30:00", "%H:%M:%S").time()
        noon = context.now.time() < datetime.datetime.strptime("11:30:00", "%H:%M:%S").time()
        after_noon_begin = context.now.time() > datetime.datetime.strptime("13:30:00", "%H:%M:%S").time()
        day_end = context.now.time() < datetime.datetime.strptime("15:00:00", "%H:%M:%S").time()
        return (day_begin & before_noon_end) | (before_noon_beign & noon) | (after_noon_begin & day_end)

    def trade(context, item):
        """根据信号进行交易"""
        now_market = history_bars(context.AllItem_contract[item], bar_count=1, frequency='1m', fields="close")[0]
        hold_pos = context.AllItem_hold[item]
        target_pos = context.AllItem_contract[item]
        instrument = rq.instruments(target_pos)
        try:
            # 突破上轨则做多
            if now_market > context.AllItem_buy[item]:
                if hold_pos == target_pos:
                    if instrument.days_to_expire(context.now) < 20 or instrument.margin_rate > 0.2:
                        if context.hold_item[item][hold_pos]["longpos"] != 0:
                            order_to(hold_pos, 0)
                    else:
                        if context.hold_item[item][hold_pos]["longpos"] != context.target_pos[item][target_pos]["longpos"]:
                            order_to(context.AllItem_contract[item], context.target_pos[item][target_pos]["longpos"])
                            logger.info(f"{context.AllItem_contract[item]}买开")
                else:
                    order_to(context.AllItem_hold[item], 0)
                    if instrument.days_to_expire(context.now) >= 20 and instrument.margin_rate <= 0.2:
                        order_to(context.AllItem_contract[item], context.target_pos[item][target_pos]["longpos"])
                        logger.info(f"先平{context.AllItem_hold[item]}，再买开{context.AllItem_contract[item]}")
            # 突破下轨则做空
            elif now_market < context.AllItem_sell[item]:
                if hold_pos == target_pos:
                    if instrument.days_to_expire(context.now) < 20 or instrument.margin_rate > 0.2:
                        if context.hold_item[item][hold_pos]["shortpos"] != 0:
                            order_to(hold_pos, 0)
                    else:
                        if context.hold_item[item][hold_pos]["shortpos"] != context.target_pos[item][target_pos]["shortpos"]:
                            order_to(context.AllItem_contract[item], -context.target_pos[item][target_pos]["shortpos"])
                            logger.info(f"{context.AllItem_contract[item]}卖开")
                else:
                    order_to(context.AllItem_hold[item], 0)
                    if instrument.days_to_expire(context.now) >= 20 and instrument.margin_rate <= 0.2:
                        order_to(context.AllItem_contract[item], -context.target_pos[item][target_pos]["shortpos"])
                        logger.info(f"先平{context.AllItem_hold[item]}，再卖开{context.AllItem_contract[item]}")
        except RuntimeError:
            print('遇到价格不合法')

    def adjust(context, item):
        check_dominant(context, item)
        check_maturity_date(context, item)
        check_margin_rate(context, item)

    def check_dominant(context, item):
        """判断持仓合约是不是主力合约，如果不是，则调整为主力合约"""
        if rq.futures.get_dominant(item, context.now) is None:
            now = rq.futures.get_dominant(item, rq.get_previous_trading_date(context.now, 1)).values[0]
        else:
            now = rq.futures.get_dominant(item, context.now).values[0]
        key = context.AllItem_hold[item]
        value = context.hold_item[item]
        if key != now:
            context.AllItem_hold.update({item: now})
            value.update({now: value[key]})
            del value[key]
            context.hold_item.update({item: value})
            order_to(key, 0)
            logger.info(f"品种{item}的主力合约不是{key}，因此全部平仓")
            if value[now]["longpos"] > 0:
                order_to(now, value[now]["longpos"])
                logger.info(f"品种{item}的主力合约是{now}， 因此买开{value[now]['longpos']}手")
            elif value[now]["shortpos"] > 0:
                order_to(now, -value[now]["shortpos"])
                logger.info(f"品种{item}的主力合约是{now}， 因此卖开{value[now]['shortpos']}手")

    def check_maturity_date(context, item):
        """判断持仓合约的距离交割日期是否小于20，如果是，则调整为次主力合约"""
        key = context.AllItem_hold[item]
        value = context.hold_item[item]
        if rq.instruments(key).days_to_expire(context.now) < 20:
            both = sub_dominant(context, item)
            if both[-1] != key:
                now = both[-1]
            else:
                now = both[0]
            context.AllItem_hold.update({item: now})
            value.update({now: value[key]})
            del value[key]
            context.hold_item.update({item: value})
            order_to(key, 0)
            logger.info(f"品种{item}的持仓合约{key}距离交割日期小于20，因此全部平仓")
            if value[now]["longpos"] > 0:
                order_to(now, value[now]["longpos"])
                logger.info(f"品种{item}更好为它的次主力合约{now}， 因此买开{value[now]['longpos']}手")
            elif value[now]["shortpos"] > 0:
                order_to(now, -value[now]["shortpos"])
                logger.info(f"品种{item}更好为它的次主力合约{now}， 因此卖开{value[now]['shortpos']}手")

    def check_margin_rate(context, item):
        """判断持仓合约的保证金是否高于20%，如果是，则剔除出持仓"""
        key = context.AllItem_hold[item]
        value = context.hold_item[item]
        if rq.instruments(key).margin_rate > 0.2:
            value.update({key: {"longpos": 0, "shortpos": 0}})
            context.hold_item.update({item: value})
            order_to(key, 0)
            logger.info(f"品种{item}的持仓合约是{key}，其保证金是高于20%，因此全部平仓")

    def sub_dominant(context, underlying_symbol: str):
        # 获取当前回测日期的所有期货合约
        tmp = rq.all_instruments('Future', date=context.now)[['order_book_id', 'maturity_date', 'underlying_symbol']]
        # 剔除到期日等于'0000-00-00'的合约
        tmp = tmp.query("maturity_date != '0000-00-00'")[['order_book_id', 'underlying_symbol']]
        # 保留目标品种的所有合约
        tmp = tmp.query(f"underlying_symbol == '{underlying_symbol}'")
        # 获得所有合约的交易量数据，再合并品种
        tmp_data = rq.get_price(order_book_ids=tmp['order_book_id'].tolist(), start_date=context.now,
                                end_date=context.now, fields='volume', adjust_type='pre_volume')
        if tmp_data is None:
            tmp_data = rq.get_price(order_book_ids=tmp['order_book_id'].tolist(),
                                    start_date=rq.get_previous_trading_date(context.now, 1),
                                    end_date=context.now, fields='volume', adjust_type='pre_volume')
        tmp_data = tmp_data.reset_index()[['order_book_id', 'volume']]
        # 按照品种获得每个品种交易量排序第二的合约
        book_id = tmp_data.sort_values('volume', ascending=False).iloc[0, :]['order_book_id']
        sub_book_id = tmp_data.sort_values('volume', ascending=False).iloc[1, :]['order_book_id']
        return book_id, sub_book_id

    rt = run_func(config=config, init=init, handle_bar=handle_bar, after_trading=after_trading)
    if isinstance(k, list):
        try:
            return pd.DataFrame({'k1': [k[0]], 'k2': [k[1]], 'tp': [k[-1]], 'n_std': [n_std], 'n_bias': [n_bias],
                                 'trade_freq': [trade_freq], 'symbol_freq': [symbol_freq], 'n_largest': [n_largest],
                                 'annualized_returns': [rt['sys_analyser']["summary"]['annualized_returns']],
                                 'sharpe': [rt['sys_analyser']["summary"]['sharpe']],
                                 'max_drawdown': [rt['sys_analyser']["summary"]['max_drawdown']],
                                 'start_date': [start], 'end_date': [end]})
        except TypeError:
            print('遇到价格不合法')


if __name__ == '__main__':
    start_time = time.time()
    adjust(5, [0.2, -0.2, 10], '20201231', '20230217', trade_freq=1, symbol_freq=1, n_largest=10,
           n_std=2, n_bias=5)
    print(f"总共花了{(time.time() - start_time) / 60}分钟")
