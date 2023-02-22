import numpy as np
import pandas as pd
from rqalpha_plus.apis import *
from rqalpha_plus import run_func
import rqdatac as rq
import os
import datetime
import time
import re
import warnings
warnings.filterwarnings("ignore")


def run(start: str, end: str):
    # 回测报告存放路径
    path = os.path.join(os.getcwd(), f'Overnight_{start}_{end}')
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
            "sys_progress": {
                "enabled": True,
                "show": True,
            },
            # 模拟撮合模块
            "sys_simulation": {
                "matching_type": "current_bar",
                "volume_limit": False
            },
            "sys_analyser": {
                "plot": False,
                "plot_save_file": os.path.join(path, '净值图.png'),
                "report_save_path": path,
                'plot_config': {
                    # 是否在收益图中展示买卖点
                    'open_close_points': False,
                    # 是否在收益图中展示周度指标和收益曲线
                    'weekly_indicators': False
                },
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

        # 2.尝试不同分钟使用计数器，以及不同频率更换品种
        context.count = 0
        context.trade = 0

        # 3.其他初始化
        # 存放回测日期已经上市的品种
        context.pass_items = set()
        # 夜盘品种
        context.night_item = []
        # 六个价位
        context.AllItem_Range = {}
        # 持仓信息
        context.AllItem_hold = {}
        context.hold_item = {}
        # 信号
        context.reverse_long = {}
        context.reverse_short = {}
        context.reverse_long_signal = False
        context.reverse_short_signal = False
        logger.info("RunInfo: {}".format(context.run_info))

    def handle_bar(context, bar_dict):
        # 1.区分夜盘和日盘品种
        if context.now.time() == datetime.datetime.strptime("21:05:00", "%H:%M:%S").time():
            if context.pass_items:
                context.night_item = [item for item in context.pass_items if
                                      '21' in rq.get_trading_hours(context.AllItem_hold[item], context.now)]
        # 30分钟代表每半小时进行一次交易
        if context.count % 30 == 0:
            context.count = 0
            # 在夜盘和日盘的交易时间段都更新仓位信息至context.AllItem_hold和context.hold_item
            if is_night(context) | is_day(context):
                cal_positions(context)

            # 2.夜盘和日盘计算交易
            for item in context.pass_items:
                if is_night(context):
                    if item in context.night_item:
                        if is_trading_time(context, item):
                            adjust(context, item)
                            bar_count = int((context.now - datetime.datetime.strptime("21:05:00", "%H:%M:%S")).seconds/60)
                            high = max(history_bars(item+'889', bar_count, '1m', 'high'))
                            low = min(history_bars(item+'889', bar_count, '1m', 'low'))
                            trade(context, item, high, low)
                elif is_day(context):
                    adjust(context, item)
                    if item in context.night_item:
                        bar_count = int((context.now - datetime.datetime.strptime("21:05:00", "%H:%M:%S")).seconds / 60)
                    else:
                        bar_count = int((context.now - datetime.datetime.strptime("09:00:00", "%H:%M:%S")).seconds / 60)
                    high = max(history_bars(item + '889', bar_count, '1m', 'high'))
                    low = min(history_bars(item + '889', bar_count, '1m', 'low'))
                    trade(context, item, high, low)

            for item in set(context.AllItem_hold.keys())-context.pass_items:
                key = context.AllItem_hold[item]
                if context.hold_item[item][key]['long_pos'] > 0 or context.hold_item[item][key]['short_pos'] > 0:
                    order_to(context.AllItem_hold[item], 0)
                    logger.info(f"{item}没有入选交易品种，因此全部平仓")
                del context.AllItem_hold[item]
                del context.hold_item[item]


        context.count += 1

    def after_trading(context):
        # 1.每天更新一次交易品种
        if context.trade % 1 == 0:
            context.trade = 0
            context.items = get_items(context)  # ['RB', 'JM', 'I', 'CU', 'AL', 'NI', 'TA', 'PP']
        context.trade += 1
        # 2.判断是否上市
        all_pass_items = [i.upper() for i in context.items]
        no_list = [i for i in all_pass_items if rq.futures.get_dominant(i, context.now) is None]
        context.pass_items = set(all_pass_items) - set(no_list)
        if no_list:
            logger.info("【{0}该品种{1}未上市】".format(context.now.date().strftime('%Y-%m-%d'), no_list))

        # 3.各品种等权分配资金
        context.single_value = context.portfolio.total_value / len(context.pass_items)

        # 4.获取当日持仓
        context.AllItem_Range = {}  # 存放计算后目标品种的指标值
        context.AllItem_hold = {}  # 存放当前持仓信息，key是品种，value是合约
        context.hold_item = {}  # 存放当前持仓信息，key是品种，value是合约加多空头仓位的字典

        # 当日结算仓位
        cal_positions(context)
        # 当前上市品种中没有持仓的品种，也记录其合约以及持仓
        for item in context.pass_items - set(context.hold_item.keys()):
            i_main = rq.futures.get_dominant(item, context.now).values[0]
            context.AllItem_hold[item] = i_main
            context.hold_item[item] = {i_main: {"long_pos": 0, "short_pos": 0}}
        # 5.计算六个价位
        for item in context.pass_items:
            item_bar = rq.get_price(item+'889', context.now, context.now, fields=['close', 'low', 'high'])
            high = item_bar['high'][0]
            close = item_bar['close'][0]
            low = item_bar['low'][0]
            context.AllItem_Range[item] = {"observe_the_ask_price":  high+0.35*(close-low),
                                           "observe_the_bid_price": low-0.35*(high-close),
                                           "reverse_the_ask_price": (1.07/2*(high+low))-0.07*low,
                                           "reverse_the_bid_price": (1.07/2*(high+low))-0.07*high}
            context.AllItem_Range[item].update({
                "breakout_of_the_bid_price":
                    context.AllItem_Range[item]["observe_the_ask_price"] +
                    0.25*(context.AllItem_Range[item]["observe_the_ask_price"] -
                          context.AllItem_Range[item]["observe_the_bid_price"]),
                "breakout_of_the_ask_price":
                    context.AllItem_Range[item]["observe_the_bid_price"] -
                    0.25*(context.AllItem_Range[item]["observe_the_ask_price"] -
                          context.AllItem_Range[item]["observe_the_bid_price"])})
            context.reverse_long.update({item: ""})
            context.reverse_short.update({item: ""})
        # 夜盘品种
        context.night_item = []
        context.count = 0
        context.reverse_long_signal = False
        context.reverse_short_signal = False

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
        top_book_id = group_vol.groupby(level=0).median().query('volume>10000').nlargest(10, 'volume').index.tolist()
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
                    context.hold_item[i] = {pos.order_book_id: {"long_pos": long_hold, "short_pos": short_hold}}
        except RuntimeError:
            print('遇到价格不合法')

    def is_trading_time(context, item):
        """判断是否处于该品种的夜盘交易时间"""
        if rq.get_trading_hours(item+'889', context.now) is None:
            night_time = rq.get_trading_hours(item + '889', rq.get_previous_trading_date(context.now, 1))
            night_time = night_time.split(',')[0].split('-')[-1]
        else:
            night_time = rq.get_trading_hours(item + '889', context.now).split(',')[0].split('-')[-1]
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

    def buy(now_hold_pos: dict, target_hold_pos: int, hold_book_id: str, target_book_id: str):
        if hold_book_id == target_book_id:
            if now_hold_pos[hold_book_id]["long_pos"] != target_hold_pos:
                order_to(target_book_id, target_hold_pos)
                logger.info(f"{target_book_id}买开{target_hold_pos}手")
        else:
            order_to(hold_book_id, 0)
            order_to(target_book_id, target_hold_pos)
            logger.info(f"先平{hold_book_id}，再买开{target_book_id}{target_hold_pos}手")

    def sell(now_hold_pos: dict, target_hold_pos: int, hold_book_id: str, target_book_id: str):
        if hold_book_id == target_book_id:
            if now_hold_pos[hold_book_id]["short_pos"] != target_hold_pos:
                order_to(target_book_id, -target_hold_pos)
                logger.info(f"{target_book_id}卖开{target_hold_pos}手")
        else:
            order_to(hold_book_id, 0)
            order_to(target_book_id, -target_hold_pos)
            logger.info(f"先平{hold_book_id}，再卖开{target_book_id}{target_hold_pos}手")

    def trade(context, item, high, low):
        """根据信号进行交易"""
        if rq.futures.get_dominant(item, context.now) is None:
            target_book_id = rq.futures.get_dominant(item, rq.get_previous_trading_date(context.now, 1)).values[0]
        else:
            target_book_id = rq.futures.get_dominant(item, context.now).values[0]
        now_price = history_bars(target_book_id, bar_count=1, frequency='1m', fields="close")[0]
        compare_price = history_bars(item+'889', bar_count=1, frequency='1m', fields="close")[0]
        hold_book_id = context.AllItem_hold[item]
        now_hold_pos = context.hold_item[item]
        target_hold_pos = int(context.single_value/rq.instruments(item+"889").contract_multiplier/now_price)
        try:
            # 突破上轨则做多
            if compare_price > context.AllItem_Range[item]["breakout_of_the_bid_price"]:
                buy(now_hold_pos, target_hold_pos, hold_book_id, target_book_id)
            # 突破下轨则做空
            elif compare_price < context.AllItem_Range[item]["breakout_of_the_ask_price"]:
                sell(now_hold_pos, target_hold_pos, hold_book_id, target_book_id)
            else:
                if context.reverse_long_signal and len(context.reverse_long[item]) > 1 and context.reverse_long[item][-2:] == '12':
                    buy(now_hold_pos, target_hold_pos, hold_book_id, target_book_id)
                    context.reverse_long[item] = ''
                elif context.reverse_short_signal and len(context.reverse_short[item]) > 1 and context.reverse_short[item][-2:] == '89':
                    sell(now_hold_pos, target_hold_pos, hold_book_id, target_book_id)
                    context.reverse_short[item] = ''
                else:
                    if not context.reverse_short_signal:
                        if high > context.AllItem_Range[item]["observe_the_ask_price"]:
                            context.reverse_short_signal = True
                    else:
                        if compare_price < context.AllItem_Range[item]["observe_the_ask_price"]:
                            context.reverse_short[item] += '8'
                        if compare_price < context.AllItem_Range[item]["reverse_the_ask_price"]:
                            context.reverse_short[item] += '9'
                    if not context.reverse_long_signal:
                        if low < context.AllItem_Range[item]["observe_the_bid_price"]:
                            context.reverse_long_signal = True
                    else:
                        if compare_price > context.AllItem_Range[item]["observe_the_bid_price"]:
                            context.reverse_long[item] += '1'
                        if compare_price > context.AllItem_Range[item]["reverse_the_bid_price"]:
                            context.reverse_long[item] += '2'

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
            if value[now]["long_pos"] > 0:
                order_to(now, value[now]["long_pos"])
                logger.info(f"品种{item}的主力合约是{now}， 因此买开{value[now]['long_pos']}手")
            elif value[now]["short_pos"] > 0:
                order_to(now, -value[now]["short_pos"])
                logger.info(f"品种{item}的主力合约是{now}， 因此卖开{value[now]['short_pos']}手")

    def check_maturity_date(context, item):
        """判断持仓合约的距离交割日期是否小于20，如果是，则调整为次主力合约"""
        key = context.AllItem_hold[item]
        value = context.hold_item[item]
        if rq.instruments(key).days_to_expire(context.now) < 20:
            now = sub_dominant(context, item)
            context.AllItem_hold.update({item: now})
            value.update({now: value[key]})
            del value[key]
            context.hold_item.update({item: value})
            order_to(key, 0)
            logger.info(f"品种{item}的持仓合约{key}距离交割日期小于20，因此全部平仓")
            if value[now]["long_pos"] > 0:
                order_to(now, value[now]["long_pos"])
                logger.info(f"品种{item}更好为它的次主力合约{now}， 因此买开{value[now]['long_pos']}手")
            elif value[now]["short_pos"] > 0:
                order_to(now, -value[now]["short_pos"])
                logger.info(f"品种{item}更好为它的次主力合约{now}， 因此卖开{value[now]['short_pos']}手")

    def check_margin_rate(context, item):
        """判断持仓合约的保证金是否高于20%，如果是，则剔除出持仓"""
        key = context.AllItem_hold[item]
        value = context.hold_item[item]
        if rq.instruments(key).margin_rate > 0.2:
            value.update({key: {"long_pos": 0, "short_pos": 0}})
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
        sub_book_id = tmp_data.sort_values('volume', ascending=False).iloc[1, :]['order_book_id']
        return sub_book_id


    run_func(config=config, init=init, handle_bar=handle_bar, after_trading=after_trading)


if __name__ == '__main__':
    start_time = time.time()
    run('20210501', '20210617')
    print(f"总共花了{(time.time() - start_time)/60}分钟")

