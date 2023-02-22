from rqalpha_plus.apis import *
from rqalpha_plus import run_func
import rqdatac as rq
import os
import time
import warnings
import traceback
import numpy as np
import pandas as pd
# import talib as ta
# from tqdm import tqdm
# from joblib import Parallel, delayed
warnings.simplefilter('ignore')
rq.init('license',
        'NSmWf24LQ53L8v1TKo-xq3_glq-Mq1RIJg81t6oKZnCW07ZrZvrBEBArPoo48ozXupXChlWEd6lB3C3nEpm83mQBvj_EVg92dxEJSR4XOD8EK76_aknPz1ZO1xHMnL_eTP11I8PSGEKbqcH-TYhLOC4_MC0-6cWgSxlAFe9q39M=XEb-4930b4g05BOoCGdX0zMDV4yxKQrPjWfAjkPv-PL5AHqAVQ8NYkYV9ma5HWxyVp6aXE7s5S7bYhkFSTGM21aV6y68L39IfH9vhHr0lDJje0cVNQcn5V1TxzPRnxqZuNaM20loa02Ij-30Qbv4X08JBUaYtE3MQKrSDTOzcI8=',
        ("rqdatad-pro.ricequant.com", 16011))


def run(d: list, t: str, r: int, h: int, ts: str):
    path = os.path.join(os.getcwd(), t, f'{d[0]}_{d[1]}_R_{str(r)}_H_{str(h)}_{ts}')
    if not os.path.exists(path):
        os.makedirs(path)
    config = {
        "base": {
            "accounts": {
                "future": 10000000,
            },
            "start_date": d[0],
            "end_date": d[1],
        },
        'mod': {
            "sys_progress": {
                "show": True,
            },
            "sys_analyser": {
                # 策略基准，该基准将用于风险指标计算和收益曲线图绘制
                #   若基准为单指数/股票，此处直接设置 order_book_id，如："000300.XSHG"
                #   若基准为复合指数，则需传入 order_book_id 和权重构成的字符串，如："000300.XSHG:0.2,000905.XSHG:0.8"
                "benchmark": None,
                # 当不输出 csv/pickle/plot 等内容时，关闭该项可关闭策略运行过程中部分收集数据的逻辑，用以提升性能
                "record": True,
                # 回测结果输出的文件路径，该文件为 pickle 格式，内容为每日净值、头寸、流水及风险指标等；若不设置则不输出该文件
                "output_file": None,
                # 是否在回测结束后绘制收益曲线图
                'plot': True,
                'report_save_path': path,
                # 收益曲线图路径，若设置则将收益曲线图保存为 png 文件
                'plot_save_file': os.path.join(path, '净值图.png'),
                # 收益曲线图设置
                'plot_config': {
                    # 是否在收益图中展示买卖点
                    'open_close_points': False,
                    # 是否在收益图中展示周度指标和收益曲线
                    'weekly_indicators': False
                },
            },
            "sys_simulation": {
                "volume_limit": False,
            }
        }
    }

    def init(context):
        # 策略类型
        context.T = t
        # 排序期
        context.R = r
        # 持有期
        context.H = h
        # 期限结构的类型
        context.TS = ts
        # 期限结构的类型
        context.ts_book_id = {}
        # 调仓计时器
        context.cnt = 0
        # 调仓的目标book_id和对应的手数
        context.target_long = {}
        context.target_short = {}
        logger.info("RunInfo: {}".format(context.run_info))

    # 你选择的期货数据更新将会触发此段逻辑，例如日线或分钟线更新
    def handle_bar(context, bar_dict):
        # 开始编写你的主要的算法逻辑
        hold_list = {i.order_book_id: {'quantity': i.quantity, 'direction': i.direction} for i in get_positions()}
        for i in {**hold_list, **context.target_long, **context.target_short}.keys():
            if i in context.target_long:
                if i in hold_list:
                    if hold_list[i]['quantity'] != context.target_long[i]['amount']:
                        order_to(i, context.target_long[i]['amount'])
                else:
                    order_to(i, context.target_long[i]['amount'])
            elif i in context.target_short:
                if i in hold_list:
                    if hold_list[i]['quantity'] != abs(context.target_short[i]['amount']):
                        order_to(i, context.target_short[i]['amount'])
                else:
                    order_to(i, context.target_short[i]['amount'])
            else:
                order_to(i, 0)

    def after_trading(context):
        # 每一个持有周期进行一次计算
        if context.cnt % context.H == 0:
            if '等权做多' in context.T:
                top_book_id, bottom_book_id = main(context)
            elif '横截面动量' in context.T:
                top_book_id, bottom_book_id = main(context)
                if '多头' in context.T:
                    top_book_id, bottom_book_id = cross(context, top_book_id, bottom_book_id)[0], []
                elif '空头' in context.T:
                    top_book_id, bottom_book_id = [], cross(context, top_book_id, bottom_book_id)[1]
                elif '多空' in context.T:
                    top_book_id, bottom_book_id = cross(context, top_book_id, bottom_book_id)
            elif '期限结构' in context.T:
                top_book_id, bottom_book_id = main(context)
                if '多头' in context.T:
                    top_book_id, bottom_book_id = roll(context, top_book_id, bottom_book_id)[0], []
                elif '空头' in context.T:
                    top_book_id, bottom_book_id = [], roll(context, top_book_id, bottom_book_id)[1]
                elif '多空' in context.T:
                    top_book_id, bottom_book_id = roll(context, top_book_id, bottom_book_id)
            elif '动量期限' in context.T:
                top_book_id, bottom_book_id = main(context)
                if '多头' in context.T:
                    # print(top_book_id, bottom_book_id)
                    top_book_id, bottom_book_id = cross(context, top_book_id, bottom_book_id)[0], []
                    if top_book_id:
                        top_book_id, bottom_book_id = roll(context, top_book_id, bottom_book_id)[0], []
                    else:
                        top_book_id, bottom_book_id = [], []
                elif '空头' in context.T:
                    top_book_id, bottom_book_id = [], cross(context, top_book_id, bottom_book_id)[1]
                    if bottom_book_id:
                        top_book_id, bottom_book_id = [], roll(context, top_book_id, bottom_book_id)[1]
                    else:
                        top_book_id, bottom_book_id = [], []
                elif '多空' in context.T:
                    top_book_id, bottom_book_id = cross(context, top_book_id, bottom_book_id)
                    if top_book_id:
                        top_book_id = roll(context, top_book_id, [])[0]
                    else:
                        top_book_id = []
                    if bottom_book_id:
                        bottom_book_id = roll(context, [], bottom_book_id)[1]
                    else:
                        bottom_book_id = []
            elif '期限动量' in context.T:
                top_book_id, bottom_book_id = main(context)
                if '多头' in context.T:
                    top_book_id, bottom_book_id = roll(context, top_book_id, bottom_book_id)[0], []
                    top_book_id, bottom_book_id = cross(context, top_book_id, bottom_book_id)[0], []
                elif '空头' in context.T:
                    top_book_id, bottom_book_id = [], roll(context, top_book_id, bottom_book_id)[1]
                    top_book_id, bottom_book_id = [], cross(context, top_book_id, bottom_book_id)[1]
                elif '多空' in context.T:
                    top_book_id, bottom_book_id = roll(context, top_book_id, bottom_book_id)
                    top_book_id = cross(context, top_book_id, [])[0]
                    bottom_book_id = cross(context, [], bottom_book_id)[1]
            elif '双因子打分' in context.T:
                top_book_id, bottom_book_id = main(context)
                if '多头' in context.T:
                    top_book_id, bottom_book_id = double(context, top_book_id, bottom_book_id)[0], []
                elif '空头' in context.T:
                    top_book_id, bottom_book_id = [], double(context, top_book_id, bottom_book_id)[1]
                elif '多空' in context.T:
                    top_book_id, bottom_book_id = double(context, top_book_id, bottom_book_id)
            else:
                top_book_id, bottom_book_id = [], []
            # top_book_id, bottom_book_id = to_dominant(context, top_book_id, bottom_book_id)
            change(context, top_book_id, bottom_book_id)
        # 每一个交易日运行，针对流动性，到期日以及保证金的要求调整持仓
        adjust(context)
        # 每过一个交易日对计时器加一
        context.cnt += 1

    def main(context):
        # 获取当前回测日期的所有期货合约
        data = rq.all_instruments(type='Future', date=context.now)[['underlying_symbol', 'listed_date']]
        # 剔除上市时间等于'0000-00-00'， 之后再根据品种获得该品种下合约的最早的上市时间
        data = data.query("listed_date != '0000-00-00'").groupby('underlying_symbol')['listed_date'].min()
        # 将data['listed_date']转成datetime格式的数据，方便后续与context.now直接相减
        data = pd.to_datetime(data)
        # 筛选得到上市满半年的品种，半年按照180天计算
        underlying_symbol = data[(context.now - data).dt.days > 180].index.tolist()
        # 剔除股指期货{'IF', 'IH', 'IC'}和国债期货{'TF', 'TS', 'T'}
        underlying_symbol = set(underlying_symbol) - {'IF', 'IH', 'IC', 'TF', 'TS', 'T'}
        # 获得筛选后的所有品种在当前回测日期的各自主力合约
        book_id = [rq.futures.get_dominant(i, start_date=context.now, end_date=context.now).item() for i in underlying_symbol]
        # 获得所有主力合约前20个交易日的交易量数据
        group_vol = rq.get_price(order_book_ids=book_id, start_date=get_previous_trading_date(date=context.now, n=20),
                                 end_date=context.now, fields='volume', adjust_type='pre_volume')
        # 按照合约名称对交易量数据求均值，再筛选交易量大于10000的合约
        top_book_id = group_vol.groupby(level=0).mean().query('volume>10000').index.tolist()
        return top_book_id, []

    def cross(context, top_book_id: list, bottom_book_id: list):
        # 合并目标合约
        book_id = top_book_id + bottom_book_id
        # 计算获得目标合约涨跌幅
        df = cal_cross(context, book_id)
        return sort(context, df)

    def cal_cross(context, book_id: list):
        # 获得所有主力合约排序期的收盘价数据
        df = rq.get_price(order_book_ids=book_id, start_date=get_previous_trading_date(date=context.now, n=context.R),
                          end_date=context.now, fields='close')
        # 因为米筐没有20050104之前的数据，所以在策略开始的时候，会遇见合约收盘价的时间序列长度小于排序期的情况
        # 这时就要设置pct_change的参数为当前时间序列长度，其他情况则仍然是排序期的长度
        if df.groupby(level=0).count().mean().values[0] < context.R:
            re_call = df.groupby(level=0).count().mean().values[0]
        else:
            re_call = context.R
        # 按照合约名称求排序期内的涨跌幅，tail(1)作用就是保留唯一的一个值，即当前日期的涨跌幅数据
        df = df.groupby(level=0).pct_change(re_call - 1).groupby(level=0).tail(1)
        # 只保留[['order_book_id', 'close']]这两列，并把列名['close']改成['target']
        df = df.reset_index()[['order_book_id', 'close']].rename(columns={'close': 'target'})
        return df

    def roll(context, top_book_id: list, bottom_book_id: list):
        # 合并目标合约
        book_id = top_book_id + bottom_book_id
        # 计算获得目标合约展期收益率
        df = cal_roll(context, book_id)
        return sort(context, df)

    def cal_roll(context, book_id: list):
        # 获得所有主力合约回测当前日期的收盘价数据
        df = rq.get_price(order_book_ids=book_id, start_date=context.now, end_date=context.now, fields='close')
        # 只保留[['order_book_id', 'close']]这两列
        df = df.reset_index()[['order_book_id', 'close']]
        # 获得合约对应的品种，并合并一起
        df = df.merge(rq.all_instruments('Future', date=context.now)[['order_book_id', 'underlying_symbol']], on='order_book_id')
        if context.TS == 'TS1':
            # 'TS1'为近月和次近月
            # 得到近月合约
            df['near_book_id'] = df['underlying_symbol'].map(lambda x: rq.futures.get_contracts(x, context.now)[0])
            # 得到次近月合约
            df['far_book_id'] = df['underlying_symbol'].map(lambda x: rq.futures.get_contracts(x, context.now)[1])
            df = cal(context, df)
        elif context.TS == 'TS2':
            # 'TS2'为近月和主力
            # 得到近月合约
            df['near_book_id'] = df['underlying_symbol'].map(lambda x: rq.futures.get_contracts(x, context.now)[0])
            df['far_book_id'] = df['order_book_id']
            df = cal(context, df)
        elif context.TS == 'TS3':
            # 'TS3'为近月和最远月
            # 得到近月合约
            df['near_book_id'] = df['underlying_symbol'].map(lambda x: rq.futures.get_contracts(x, context.now)[0])
            # 得到最远月合约
            df['far_book_id'] = df['underlying_symbol'].map(lambda x: rq.futures.get_contracts(x, context.now)[-1])
            df = cal(context, df)
        elif context.TS == 'TS4':
            # 'TS4'为主力和次主力
            # 得到次主力合约
            df = cal_sub_dominant(context, df)
            # 得到到期日数据
            df['order_days'] = list(map(lambda x: x.days_to_expire(context.now), rq.instruments(df['order_book_id'].tolist())))
            df['sub_days'] = list(map(lambda x: x.days_to_expire(context.now), rq.instruments(df['sub_book_id'].tolist())))
            # 判断主力合约和次主力合约的到期日，到期日小的设为近月合约，大的设为远月合约
            df['near_book_id'] = np.where(df['order_days'] < df['sub_days'], df['order_book_id'], df['sub_book_id'])
            df['far_book_id'] = np.where(df['order_days'] < df['sub_days'], df['sub_book_id'], df['order_book_id'])
            df = cal(context, df)
        # 计算展期收益率
        df['target'] = (np.log(df['near_price']) - np.log(df['far_price'])) * 365 / (df['far_days'] - df['near_days'])
        df = df[['order_book_id', 'target']]
        return df

    def double(context, top_book_id: list, bottom_book_id: list):
        # 合并目标合约
        book_id = top_book_id + bottom_book_id
        # 合并双因子按排序打分相加后的总分
        df = cal_double(context, book_id)
        return sort(context, df)

    def cal_double(context, book_id: list):
        df1 = cal_cross(context, book_id).rename(columns={'target': 'cross_return'})
        df2 = cal_roll(context, book_id).rename(columns={'target': 'time_return'})
        # 按照’order_book_id‘合并'cross_return'因子和time_return'因子
        df = df1.merge(df2, on='order_book_id')
        # 计算排序相加得到的总分
        df['target'] = df['cross_return'].rank() + df['time_return'].rank()
        df = df[['order_book_id', 'target']]
        return df

    def cal(context, df: pd.DataFrame):
        # 获得近月合约和远月合约的收盘价
        tmp = rq.get_price(df['near_book_id'].tolist()+df['far_book_id'].tolist(), start_date=context.now,
                           end_date=context.now, fields='close').reset_index()[['order_book_id', 'close']]
        # 近月合约及其收盘价数据
        tmp_near = tmp.query(f"order_book_id in {df['near_book_id'].tolist()}").rename(columns={'order_book_id': 'near_book_id', 'close': 'near_price'})
        # 远月合约及其收盘价数据
        tmp_far = tmp.query(f"order_book_id in {df['far_book_id'].tolist()}").rename(columns={'order_book_id': 'far_book_id', 'close': 'far_price'})
        df = df.merge(tmp_near, on='near_book_id').merge(tmp_far, on='far_book_id')
        # 获得近月合约和远月合约的到期日
        df['near_days'] = list(map(lambda x: x.days_to_expire(context.now), rq.instruments(df['near_book_id'].tolist())))
        df['far_days'] = list(map(lambda x: x.days_to_expire(context.now), rq.instruments(df['far_book_id'].tolist())))
        return df

    def sort(context, df: pd.DataFrame):
        df = df.sort_values(by='target', ascending=False)
        if '动量期限' in context.T:
            if traceback.extract_stack()[-2][2] == "cross":
                # 涨跌幅的80%分位数
                top = df['target'].quantile(q=0.8)
                # 涨跌幅的20%分位数
                bottom = df['target'].quantile(q=0.2)
            elif traceback.extract_stack()[-2][2] == "roll":
                # 涨跌幅的50%分位数
                top = df['target'].quantile(q=0.5)
                # 涨跌幅的50%分位数
                bottom = df['target'].quantile(q=0.5)
            else:
                top, bottom = None, None
        elif '期限动量' in context.T:
            if traceback.extract_stack()[-2][2] == "roll":
                # 涨跌幅的80%分位数
                top = df['target'].quantile(q=0.8)
                # 涨跌幅的20%分位数
                bottom = df['target'].quantile(q=0.2)
            elif traceback.extract_stack()[-2][2] == "cross":
                # 涨跌幅的50%分位数
                top = df['target'].quantile(q=0.5)
                # 涨跌幅的50%分位数
                bottom = df['target'].quantile(q=0.5)
            else:
                top, bottom = None, None
        elif '双因子打分' in context.T:
            # 涨跌幅的75%分位数
            top = df['target'].quantile(q=0.75)
            # 涨跌幅的25%分位数
            bottom = df['target'].quantile(q=0.25)
        else:
            # 涨跌幅的80%分位数
            top = df['target'].quantile(q=0.8)
            # 涨跌幅的20%分位数
            bottom = df['target'].quantile(q=0.2)
        # 涨跌幅大于top分位数的就是累计涨幅排名前top的品种
        top_book_id = df.query(f'target > {top}')['order_book_id'].tolist()
        # 涨跌幅小于bottom分位数的就是累计涨幅排名后bottom的品种
        bottom_book_id = df.query(f'target < {bottom}')['order_book_id'].tolist()
        return top_book_id, bottom_book_id

    def change(context, top_book_id: list, bottom_book_id: list):
        filtered_book_id = list(top_book_id) + list(bottom_book_id)
        # 若当前计算得到的合约有更新，则调仓
        if (set(context.target_long.keys()) != set(top_book_id)) or (set(context.target_short.keys()) != set(bottom_book_id)):
            # 重置context.target_long, context.target_short
            context.target_long = {}
            context.target_short = {}
            # 当前投资组合总权益除以合约数，得到平均分配给每个合约的价值
            mean_value = context.portfolio.portfolio_value / len(filtered_book_id)
            # 获得所有合约
            instrument = rq.instruments(filtered_book_id)
            # 获得合约乘数
            multiplier = list(map(lambda x: x.contract_multiplier, instrument))
            # 获得合约的品种
            underlying_symbol = list(map(lambda x: x.underlying_symbol, instrument))
            if filtered_book_id:
                price = rq.get_price(filtered_book_id, start_date=context.now, end_date=context.now, fields='close').reset_index(level=1, drop=True)
            for j, m, u in zip(filtered_book_id, multiplier, underlying_symbol):
                # 获得当前合约当前交易日的收盘价
                close = price.loc[j, 'close']
                # 平均分配给每个合约的价值，除以一手合约的市值，得到该合约下单手数
                amount = int(mean_value / (m * close))
                if j in top_book_id:
                    context.target_long.update({j: {'underlying_symbol': u, 'amount': amount}})
                elif j in bottom_book_id:
                    context.target_short.update({j: {'underlying_symbol': u, 'amount': amount if '空头' in context.T else -amount}})

    def adjust(context):
        # 每天判断目标合约是不是主力合约，如果不是，则调整为主力合约
        for key, value in {**context.target_long, **context.target_short}.items():
            now = rq.futures.get_dominant(value['underlying_symbol'], start_date=context.now, end_date=context.now).item()
            if key != now:
                if key in context.target_long:
                    context.target_long.update({now: value})
                    del context.target_long[key]
                else:
                    context.target_short.update({now: value})
                    del context.target_short[key]
        # 每天判断目标合约距离交割日期是否小于20，如果是，则调整为次主力合约
        for key, value in {**context.target_long, **context.target_short}.items():
            if rq.instruments(key).days_to_expire(context.now) < 20:
                now = sub_dominant(context, [value['underlying_symbol']])['sub_book_id'].values[0]
                if key in context.target_long:
                    context.target_long.update({now: value})
                    del context.target_long[key]
                else:
                    context.target_short.update({now: value})
                    del context.target_short[key]
        # 每天判断目标合约保证金是否高于20%，如果是，则剔除出持仓
        for key, value in {**context.target_long, **context.target_short}.items():
            if rq.instruments(key).margin_rate > 0.2:
                if key in context.target_long:
                    del context.target_long[key]
                else:
                    del context.target_short[key]

    def cal_sub_dominant(context, df: pd.DataFrame):
        return df.merge(sub_dominant(context, df['underlying_symbol'].tolist()), on='underlying_symbol')

    def sub_dominant(context, underlying_symbol: list):
        # 获取当前回测日期的所有期货合约
        tmp = rq.all_instruments('Future', date=context.now)[['order_book_id', 'maturity_date', 'underlying_symbol']]
        # 剔除到期日等于'0000-00-00'的合约
        tmp = tmp.query("maturity_date != '0000-00-00'")[['order_book_id', 'underlying_symbol']]
        # 保留目标品种的所有合约
        tmp = tmp.query(f"underlying_symbol in {underlying_symbol}")
        # 获得所有合约的交易量数据，再合并品种
        tmp_data = rq.get_price(order_book_ids=tmp['order_book_id'].tolist(), start_date=context.now,
                                end_date=context.now, fields='volume', adjust_type='pre_volume').reset_index()[['order_book_id', 'volume']]
        tmp_data = tmp_data.merge(tmp, on='order_book_id')
        # 按照品种获得每个品种交易量排序第二的合约
        tmp_data = tmp_data.groupby('underlying_symbol').apply(lambda x: x.sort_values('volume', ascending=False).iloc[1, :])
        tmp_data = tmp_data.reset_index(drop=True).rename(columns={'order_book_id': 'sub_book_id'})[['sub_book_id', 'underlying_symbol']]
        return tmp_data

    def to_dominant(context, top_book_id: list, bottom_book_id: list):
        top_book_id = [rq.futures.get_dominant(i[:-3], start_date=context.now, end_date=context.now).item() for i in top_book_id]
        bottom_book_id = [rq.futures.get_dominant(i[:-3], start_date=context.now, end_date=context.now).item() for i in bottom_book_id]
        return top_book_id, bottom_book_id

    return run_func(init=init, config=config, handle_bar=handle_bar, after_trading=after_trading)


DATE_TIME = ["20050104", "20170120"]
T = ['等权做多策略',
     '横截面动量多头组合策略', '横截面动量空头组合策略', '横截面动量多空组合策略',
     '期限结构多头组合策略', '期限结构空头组合策略', '期限结构多空组合策略',
     '动量期限多头组合策略', '动量期限空头组合策略', '动量期限多空组合策略',
     '期限动量多头组合策略', '期限动量空头组合策略', '期限动量多空组合策略',
     '双因子打分多头组合策略', '双因子打分空头组合策略', '双因子打分多空组合策略']
R = [0, 5, 10, 15, 20, 25, 30, 35, 40]
H = [0, 5, 10, 15, 20, 25, 30, 35, 40]
TS = ['TS', 'TS1', 'TS2', 'TS3', 'TS4']
start_time = time.time()
result = run(DATE_TIME, T[15], R[2], H[2], TS[3])
print(time.time()-start_time)
