from joblib import Parallel, delayed
import R_Breaker_faster
import pandas as pd
import time


def run(start: str, end: str):
    all_ = [[5 * i, j] for i in range(1, 7) for j in range(1, 4) if 5 * i != 5 or j != 2]
    return Parallel(n_jobs=-1)(delayed(R_Breaker_faster.run)(start, end, 25, 1, 0.05, 0.01, 10, j, i) for i, j in all_)


if __name__ == '__main__':
    start_time = time.time()
    df = pd.concat(run('20210101', '20230217'), ignore_index=True)
    df = df.sort_values('annualized_returns', ascending=False)
    df = df[['n_std', 'n_bias', 'annualized_returns', 'sharpe', 'max_drawdown']]
    df.to_csv('params.csv', index=False)
    print(df)
    print(f"总共花了{(time.time() - start_time) / 60: .2f}分钟")
