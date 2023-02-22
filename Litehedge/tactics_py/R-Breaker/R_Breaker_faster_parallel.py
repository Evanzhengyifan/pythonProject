from joblib import Parallel, delayed
import R_Breaker_faster
import time


def run(start: str, end: str):
    all_ = [5, 10, 15, 20, 25, 30]
    Parallel(n_jobs=-1)(delayed(R_Breaker_faster.run)(start, end, i) for i in all_)


if __name__ == '__main__':
    start_time = time.time()
    run('20210101', '20230217')
    print(f"总共花了{(time.time() - start_time) / 60: .2f}分钟")
