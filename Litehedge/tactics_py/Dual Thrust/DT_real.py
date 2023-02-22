from rqalpha_plus.apis import *
from rqalpha_plus import run_func
import rqdatac
import os
import re
import time
import datetime
import dateutil
import numpy as np
import pandas as pd
import DT
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")


def run_real(start: str, end: str, months: int):
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
        all_sub_item = rqdatac.all_instruments(type='Future', date=context.now)["underlying_symbol"].tolist()
        all_sub_item = set(all_sub_item) - {'IF', 'IH', 'IC', 'IM', 'TF', 'TS', 'T', 'SI'}
        all_sublist = [i + "889" for i in all_sub_item]
        for i in all_sublist:
            subscribe(i)

        # 2.参数 （窗口 上下轨系数）
        context.n = 5
        context.tp = 10
        context.freq = months
        print("更新参数")
        end_date = rqdatac.get_previous_trading_date(context.now, n=1)
        start_date = (end_date - dateutil.relativedelta.relativedelta(months=context.freq)).strftime("%Y%m%d")
        context.k1, context.k2 = get_params(start_date, end_date.strftime("%Y%m%d"))
        print(f"{start_date}-{end_date}, 滚动回测得到的最优参数为，k1: {context.k1}, k2: {context.k2}")
        # 隔夜策略暂时用不到这个参数
        context.check_time = "14:56:00"
        # 3.尝试不同分钟使用计数器
        context.count = 0
        context.c = 0  # 更新parameter的计时器

        # 4.其他初始化
        # 存放回测日期已经上市的品种
        context.pass_items = set()
        # 夜盘品种
        context.night_item = []
        context.hold_item = {}
        # 目标持仓
        context.AllItem_contract = {}
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
                                      '21' in rqdatac.get_trading_hours(context.AllItem_contract[item], context.now)]
                # logger.info("【夜盘品种{0}】".format(context.night_item))
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
                        trade(context, item)
                elif is_day(context):
                    trade(context, item)

        context.count += 1

    def after_trading(context):
        # 更新参数
        if int(rqdatac.get_next_trading_date(context.now, n=1).strftime("%m")) - int(context.now.strftime("%m")) == 1:
            context.c += 1
        elif int(rqdatac.get_next_trading_date(context.now, n=1).strftime("%m")) - int(context.now.strftime("%m")) == -11:
            context.c += 1
        if context.c == context.freq:
            if context.now.strftime("%Y%m") != end[:-2]:
                context.c = 0
                print("更新参数")
                start_date = (context.now - dateutil.relativedelta.relativedelta(months=context.freq)).strftime("%Y%m%d")
                context.k1, context.k2 = get_params(start_date, context.now.strftime("%Y%m%d"))
                print(f"{start_date}-{context.now}, 滚动回测得到的最优参数为，k1: {context.k1}, k2: {context.k2}")
        # 1.日期和时间
        context.Pre_day = rqdatac.get_previous_trading_date(context.now, n=context.n - 1)
        context.next_date = rqdatac.get_next_trading_date(context.now, n=1)
        context.items = get_items(context)

        # 2.判断是否上市
        all_pass_items = [i.upper() for i in context.items]
        no_list = [i for i in all_pass_items if rqdatac.futures.get_dominant(i, context.now) is None]
        context.pass_items = set(all_pass_items) - set(no_list)
        # logger.info(f'已上市品种{context.pass_items}')
        if no_list:
            logger.info("【{0}该品种{1}未上市】".format(context.now.date().strftime('%Y-%m-%d'), no_list))

        # 3.各品种等权分配资金
        context.single_value = context.portfolio.total_value / len(context.pass_items)

        # 4.获取当日持仓
        context.AllItem_contract = {}  # 存放计算后的目标持仓信息，key是品种，value是合约
        context.AllItem_hold = {}  # 存放当前持仓信息，key是品种，value是合约
        context.AllItem_Range = {}  # 存放计算后目标品种的range值
        context.AllItem_buy = {}  # 存放计算后目标品种的上轨
        context.AllItem_sell = {}  # 存放计算后目标品种的下轨
        context.hold_item = {}  # 存放当前持仓信息，key是品种，value是合约加多空头仓位的字典
        context.target_pos = {}  # 存放计算后的目标持仓信息，key是品种，value是合约加多空头仓位的字典

        # 当日结算仓位
        cal_positions(context)
        # 当前上市品种中没有持仓的品种，也记录其合约以及持仓
        for item in context.pass_items - set(context.hold_item.keys()):
            i_main = rqdatac.futures.get_dominant(item, context.now).values[0]
            context.AllItem_hold[item] = i_main
            context.hold_item[item] = {i_main: {"longpos": 0, "shortpos": 0}}
        # logger.info("持仓{0}".format(context.hold_item['RB']))
        # 5.计算上下轨 计算目标仓位
        for item in context.pass_items:
            # 设置品种对应的主力合约
            main = rqdatac.futures.get_dominant(item, context.now).values[0]
            context.AllItem_contract[item] = main
            item_bar = rqdatac.get_price(main, start_date=context.Pre_day, end_date=context.now,
                                         fields=['open', 'close', 'low', 'high'])
            result = dual_thrust(item_bar, context.n)

            context.AllItem_Range[item] = result["Range"].values[-1]
            # 回测中姑且在after_trading就获取下一日的开盘价，来计算上下轨，是盘中不能这么写
            today_open = rqdatac.get_price(main, start_date=context.next_date, end_date=context.next_date,
                                           fields='open')['open'].values[0]
            context.AllItem_buy[item] = today_open + (float(context.k1) * context.AllItem_Range[item])
            context.AllItem_sell[item] = today_open + (float(context.k2) * context.AllItem_Range[item])
            long_pos = int(
                context.single_value / rqdatac.instruments(item + "889").contract_multiplier / context.AllItem_buy[item])
            short_pos = int(
                context.single_value / rqdatac.instruments(item + "889").contract_multiplier / context.AllItem_sell[item])
            context.target_pos[item] = {main: {"longpos": long_pos, "shortpos": short_pos}}

        # 夜盘品种
        context.night_item = []
        context.count = 0

    def get_items(context):
        # 获取当前回测日期的所有期货合约
        data = rqdatac.all_instruments(type='Future', date=context.now)[['underlying_symbol', 'listed_date']]
        # 剔除上市时间等于'0000-00-00'， 之后再根据品种获得该品种下合约的最早的上市时间
        data = data.query("listed_date != '0000-00-00'").groupby('underlying_symbol')['listed_date'].min()
        # 将data['listed_date']转成datetime格式的数据，方便后续与context.now直接相减
        data = pd.to_datetime(data)
        # 筛选得到上市满半年的品种，半年按照180天计算
        underlying_symbol = data[(context.now - data).dt.days > 180].index.tolist()
        # 剔除股指期货{'IF', 'IH', 'IC'}和国债期货{'TF', 'TS', 'T'}
        underlying_symbol = set(underlying_symbol) - {'IF', 'IH', 'IC', 'TF', 'TS', 'T'}
        # 构造筛选后所有品种的主力连续合约
        book_id = [i + '889' for i in underlying_symbol]
        # 获得所有主力连续合约前20个交易日的交易量数据
        group_vol = rqdatac.get_price(order_book_ids=book_id, start_date=get_previous_trading_date(date=context.now, n=20),
                                      end_date=context.now, fields='volume', adjust_type='pre_volume')
        # 按照合约名称对交易量数据求均值，再筛选交易量大于10000的合约，再取前K个
        top_book_id = group_vol.groupby(level=0).median().query('volume>10000').nlargest(context.tp, 'volume').index.tolist()
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
        try:
            # 突破上轨则做多
            if now_market > context.AllItem_buy[item]:
                if hold_pos == target_pos:
                    if context.hold_item[item][hold_pos]["longpos"] != context.target_pos[item][target_pos]["longpos"]:
                        order_to(context.AllItem_contract[item], context.target_pos[item][target_pos]["longpos"])
                        logger.info(f"{context.AllItem_contract[item]}买开")
                else:
                    order_to(context.AllItem_hold[item], 0)
                    order_to(context.AllItem_contract[item], context.target_pos[item][target_pos]["longpos"])
                    logger.info(f"先平{context.AllItem_hold[item]}，再买开{context.AllItem_contract[item]}")
            # 突破下轨则做空
            elif now_market < context.AllItem_sell[item]:
                if hold_pos == target_pos:
                    if context.hold_item[item][hold_pos]["shortpos"] != context.target_pos[item][target_pos]["shortpos"]:
                        order_to(context.AllItem_contract[item], -context.target_pos[item][target_pos]["shortpos"])
                        logger.info(f"{context.AllItem_contract[item]}卖开")
                else:
                    order_to(context.AllItem_hold[item], 0)
                    order_to(context.AllItem_contract[item], -context.target_pos[item][target_pos]["shortpos"])
                    logger.info(f"先平{context.AllItem_hold[item]}，再卖开{context.AllItem_contract[item]}")
        except RuntimeError:
            print('遇到价格不合法')

    def get_params(start_date, end_date):
        this_time = time.time()
        all_ = [[k/10, 10, start_date, end_date] for k in range(1, 6)]
        dfs = Parallel(n_jobs=-1)(delayed(DT.adjust)(5, [a[0], -a[0], a[1]], a[2], a[3]) for a in tqdm(all_))
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values('sharpe', ascending=False)
        k1 = df.iloc[0, :]['k1']
        k2 = df.iloc[0, :]['k2']
        print(f'{start_date}-{end_date}滚动回测结束')
        print(f"总共花了{(time.time() - this_time)/60}分钟")
        return k1, k2

    run_func(config=config, init=init, handle_bar=handle_bar, after_trading=after_trading)


if __name__ == '__main__':
    start_time = time.time()
    run_real('20140101', '20221231', 6)
    print(f"总共花了{(time.time() - start_time)/60}分钟")







