import time
import calendar
import datetime
import pandas as pd
from joblib import Parallel, delayed
import DT
from tqdm import tqdm
import dateutil

# def run():
#     date_list = [['20200101', '20201231'],
#                  ['20210101', '20211231'],
#                  ['20220101', '20221231']]
#     kd = {}
#     for i, j in date_list:
#         this_time = time.time()
#         dfs = Parallel(n_jobs=-2)(delayed(DT.adjust)(5, [k/10, -k/10], i, j) for k in tqdm(range(1, 6)))
#         df = pd.concat(dfs, ignore_index=True).sort_values(by='sharpe', ascending=False)
#         df.to_csv(f'{i}_{j}.csv', index=False)
#         kd.update({str(int(i[:-4])+1): {'k1': df.iloc[0, :]['k1'], 'k2': df.iloc[0, :]['k2']}})
#         print(kd)
#         print(f'{i}-{j}回测结束')
#         print(time.time() - this_time)
#     pd.DataFrame(kd).to_csv(f'{date_list[0][0]}_{date_list[-1][-1]}.csv', index=False)
#
#     DT.adjust(5, kd, date_list[1][0], date_list[-1][-1])


def get_months_range_list(start_date, end_date, freq):
    """
    获取时间参数列表
    :param startdate: 起始月初时间 --> str
    :param enddate: 结束时间 --> str
    :return: date_range_list -->list
    """
    date_range_list = []
    date_type_list = []
    start_date = datetime.datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.datetime.strptime(end_date, '%Y%m%d')
    start_date = datetime.date(start_date.year, start_date.month, 1)
    end_date = datetime.date(end_date.year, end_date.month, calendar.monthrange(end_date.year, end_date.month)[-1])
    assert start_date < end_date
    date_type_list.append(start_date - datetime.timedelta(days=1))
    while True:
        next_month = start_date + dateutil.relativedelta.relativedelta(months=freq)
        month_end = next_month - datetime.timedelta(days=1)
        if month_end <= end_date:
            date_type_list.append(month_end)
            date_range_list.append((datetime.datetime.strftime(start_date, '%Y%m%d'), datetime.datetime.strftime(month_end, '%Y%m%d')))
            start_date = next_month
        else:
            return date_range_list, date_type_list


def get_months_range_list(start_date, end_date, freq, forward):
    """
    获取时间参数列表
    :param startdate: 起始月初时间 --> str
    :param enddate: 结束时间 --> str
    :return: date_range_list -->list
    """
    date_range_list = []
    date_type_list = []
    start_date = datetime.datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.datetime.strptime(end_date, '%Y%m%d')
    start_date = datetime.date(start_date.year, start_date.month, 1)
    end_date = datetime.date(end_date.year, end_date.month, calendar.monthrange(end_date.year, end_date.month)[-1])
    assert start_date < end_date
    date_type_list.append(start_date - datetime.timedelta(days=1))
    while True:
        next_month = start_date + dateutil.relativedelta.relativedelta(months=freq)
        month_end = next_month - datetime.timedelta(days=1)
        if month_end <= end_date:
            date_type_list.append(month_end)
            date_range_list.append((datetime.datetime.strftime(start_date, '%Y%m%d'), datetime.datetime.strftime(month_end, '%Y%m%d')))
            start_date = start_date + dateutil.relativedelta.relativedelta(months=forward)
        else:
            return date_range_list, date_type_list


# def run(t: str):
#     date_list = [['20200101', '20201231'],
#                  ['20210101', '20211231'],
#                  ['20220101', '20221231']]
#     date_list = [['20200101', '20200131'],
#                  ['20200201', '20200229'],
#                  ['20200301', '20200331'],
#                  ['20200401', '20200430'],
#                  ['20200501', '20200531']]
#     # kd = {}
#     this_time = time.time()
#     all_ = [[k/10, i[0], i[-1]] for i in date_list[:-1] for k in range(1, 6)]
#     dfs = Parallel(n_jobs=-1)(delayed(DT.adjust)(5, [a[0], -a[0]], a[1], a[2]) for a in tqdm(all_))
#     df = pd.concat(dfs, ignore_index=True)
#     print(df)
#     if t == 'years':
#         sft = '%Y'
#     elif t == 'months':
#         sft = '%Y%m'
#     df[t] = eval(f"pd.to_datetime(df['start_date']).map(lambda x: x + dateutil.relativedelta.relativedelta({t}=1)).dt.strftime(sft)")
#     print(df)
#     df = df.groupby(t, as_index=False).apply(lambda x: x.sort_values(by='sharpe', ascending=False).iloc[0, :])
#     df = df[[t, 'k1', 'k2']]
#     print(df)
#     df = df.pivot_table(columns=s)
#     df.to_csv(f'{date_list[1][0]}_{date_list[-1][-1]}.csv', index=False)
#     kd = df.to_dict()
#     print(kd)
#     print(f'{date_list[0][0]}-{date_list[-2][-1]}回测结束')
#     print(time.time() - this_time)
#
#     DT.adjust(5, kd, date_list[1][0], date_list[-1][-1])


# def run(t: str, start_date: str, end_date: str, freq: int, forward: int):
#     date_list = eval(f"get_{t}_range_list(start_date, end_date, freq)[0]")
#     bins = eval(f"pd.to_datetime(get_{t}_range_list(start_date, end_date, freq)[-1])")
#     print(date_list)
#     print(bins)
#     this_time = time.time()
#     all_ = [[k/10, t, i[0], i[-1]] for i in date_list[:-1] for k in range(1, 6) for t in [10]]
#     dfs = Parallel(n_jobs=-1)(delayed(DT.adjust)(5, [a[0], -a[0], a[1]], a[2], a[3]) for a in tqdm(all_))
#     df = pd.concat(dfs, ignore_index=True)
#     print(df)
#     df['end_date'] = pd.to_datetime(df['end_date'])
#     df['date_range'] = pd.cut(df['end_date'], bins=bins)
#     df = df.sort_values(['date_range', 'sharpe'], ascending=[True, False])
#     df['k1'] = df.groupby('date_range', as_index=False)['k1'].transform('first')
#     df['k2'] = df.groupby('date_range', as_index=False)['k2'].transform('first')
#     df['tp'] = df.groupby('date_range', as_index=False)['tp'].transform('first')
#     if t == 'years':
#         sft = '%Y'
#     elif t == 'months':
#         sft = '%Y%m'
#     df[t] = eval(f"df['end_date'].map(lambda x: x + dateutil.relativedelta.relativedelta({t}=1)).dt.strftime(sft)")
#     print(df)
#     df = df[[t, 'k1', 'k2', 'tp']]
#     print(df)
#     df = df.pivot_table(columns=t)
#     kd = {}
#     for k, v in df.to_dict().items():
#         for j in range(1, freq+1):
#             new_k = (datetime.date(int(k[:-2]), int(k[4:]), 1) + dateutil.relativedelta.relativedelta(months=j-1)).strftime('%Y%m')
#             kd.update({new_k: v})
#     pd.DataFrame(kd).to_csv(f'{date_list[1][0]}_{date_list[-1][-1]}.csv')
#     print(kd)
#     print(f'{date_list[0][0]}-{date_list[-2][-1]}回测结束')
#     print(time.time() - this_time)
#
#     DT.adjust(5, kd, date_list[1][0], date_list[-1][-1])

# def run(t: str, start_date: str, end_date: str, freq: int, forward: int):
#     date_list = eval(f"get_{t}_range_list(start_date, end_date, freq, forward)[0]")
#     bins = eval(f"pd.to_datetime(get_{t}_range_list(start_date, end_date, freq, forward)[-1])")
#     print(date_list)
#     print(bins)
#     this_time = time.time()
#     all_ = [[k/10, t, i[0], i[-1]] for i in date_list[:-1] for k in range(1, 6) for t in [10]]
#     dfs = Parallel(n_jobs=-1)(delayed(DT.adjust)(5, [a[0], -a[0], a[1]], a[2], a[3]) for a in tqdm(all_))
#     df = pd.concat(dfs, ignore_index=True)
#     print(df)
#     df['end_date'] = pd.to_datetime(df['end_date'])
#     df['date_range'] = pd.cut(df['end_date'], bins=bins)
#     df = df.sort_values(['date_range', 'sharpe'], ascending=[True, False])
#     df['k1'] = df.groupby('date_range', as_index=False)['k1'].transform('first')
#     df['k2'] = df.groupby('date_range', as_index=False)['k2'].transform('first')
#     df['tp'] = df.groupby('date_range', as_index=False)['tp'].transform('first')
#     if t == 'years':
#         sft = '%Y'
#     elif t == 'months':
#         sft = '%Y%m'
#     df[t] = eval(f"df['end_date'].map(lambda x: x + dateutil.relativedelta.relativedelta({t}=1)).dt.strftime(sft)")
#     print(df)
#     df = df[[t, 'k1', 'k2', 'tp']]
#     print(df)
#     df = df.pivot_table(columns=t)
#     kd = df.to_dict()
#     for k, v in kd.items():
#         for j in range(1, forward):
#             new_k = (datetime.date(int(k[:-2]), int(k[4:]), 1) + dateutil.relativedelta.relativedelta(months=j)).strftime('%Y%m')
#             kd.update({new_k: v})
#     pd.DataFrame(kd).to_csv(f'{date_list[1][0]}_{date_list[-1][-1]}.csv')
#     print(kd)
#     print(f'{date_list[0][0]}-{date_list[-2][-1]}回测结束')
#     print(time.time() - this_time)
#
#     DT.adjust(5, kd, '20210101', date_list[-1][-1])
#
#
# if __name__ == "__main__":
#     start_time = time.time()
#     run('months', '20200701', '20210630', 6, 1)
#     print(f"总共花了{(time.time() - start_time)/60}分钟")


def run():
    date_list = [['20201231', '20230217']]
    all_ = [[i, k] for i in date_list for k in [20, 30]]
    return Parallel(n_jobs=-1)(delayed(DT.adjust)(5, [0.2, -0.2, a[1]], a[0][0], a[0][1], 2, 5) for a in tqdm(all_))


if __name__ == "__main__":
    start_time = time.time()
    print(pd.concat(
        run(), ignore_index=True).
          sort_values('annualized_returns', ascending=False)
          [['k1', 'k2', 'tp', 'n_std', 'n_bias', 'annualized_returns', 'sharpe', 'max_drawdown']])
    print(f"总共花了{(time.time() - start_time)/60}分钟")

