from utils import *
import pandas as pd

def chiMerge(df, attr, label, max_intervals):
    # 初始化，记录区间左右端点，总样本数量，负样本数量；合并区间时更新
    intervals = df[[label, attr]].groupby(attr).agg(['count', 'sum'])
    intervals.columns = ['total', 'bad']
    intervals_info = intervals.to_dict('index')
    intervals_info = {x:{'bound':[x,x], 'num':[float(v['total']), float(v['bad'])]} for x, v in intervals_info.items()}
    intervals_chi = {x: None for x in intervals_info}
    # while loop
    while len(intervals_info) > max_intervals:
        # calculate chi statistic
        sorted_keys = sorted(intervals_info.keys())
        min_chi, min_chi_idx = float('inf'), -1
        for i in range(len(sorted_keys)-1):
            group1, group2 = sorted_keys[i], sorted_keys[i+1]
            if intervals_chi[group1] is None:
                N1, B1 = intervals_info[group1]['num']
                N2, B2 = intervals_info[group2]['num']
                N, B = N1+N2, B1+B2
                if B == 0 or B == N:
                    min_chi_idx = i
                    break
                chi = N*N*N/N1/N2/B/(N-B)*(B1-N1*B/N)*(B1-N1*B/N)
                intervals_chi[group1] = chi
            else:
                chi = intervals_chi[group1]
            if chi < min_chi:
                min_chi = chi
                min_chi_idx = i

        # merge
        group1, group2 = sorted_keys[min_chi_idx], sorted_keys[min_chi_idx+1]
        intervals_info[group1]['num'] = [intervals_info[group1]['num'][0] + intervals_info[group2]['num'][0],
                                         intervals_info[group1]['num'][1] + intervals_info[group2]['num'][1]]
        intervals_info[group1]['bound'][1] = intervals_info[group2]['bound'][1]
        intervals_chi[group1] = None
        if min_chi_idx > 1:
            intervals_chi[sorted_keys[i-1]] = None
        del intervals_info[group2]
    return intervals_info

# def calc_ci(rate_df):
#     tmp = 1.96*np.sqrt((rate_df['rate']*(1-rate_df['rate'])/rate_df['total']))
#     rate_df['rate_lower'] = np.clip(rate_df['rate']-tmp, 0, 1)
#     rate_df['rate_upper'] = np.clip(rate_df['rate']+tmp, 0, 1)
#     rate_df.loc[rate_df['1']==0, 'rate_upper'] = 1 - np.power(0.05, 1/rate_df['total'])

def highFreq(data, f, label='label', tol=0.01, plot=True, verbose=True, default_value=999999):
    if plot:
        data[f].hist(bins=50)
        plt.show()
    tmp = data.groupby(f).size()
    if 0 < tol < 1:
        idx = list(tmp.index[tmp > (data[f].notnull().sum()*tol)])
    else:
        idx = list(tmp.index[tmp > tol])
    data['%s#encode'%f] = data[f].copy()
    data.loc[(~data[f].isin(idx)) &(data[f].notnull()), '%s#encode'%f] = default_value
    table, iv = count(data, '%s#encode'%f, 10**10)
    #calc_ci(table)
    table = table.sort_values('rate')
    if verbose:
        print('iv=%.4f'%iv)
        print(table)
    return list(table.index)

if __name__ == '__main__':
    df_trans = pd.read_csv('./data/train_transaction.csv')
    df_trans.rename({'isFraud':'label'}, inplace=True, axis=1)
    idx = highFreq(df_trans, 'addr1')
    idx = [x for x in idx if not np.isnan(x)]
    print(idx)
    df_trans['test'] = df_trans['addr1'].copy()
    df_trans.loc[(~df_trans['addr1'].isin(idx)) & (df_trans['addr1'].notnull()), 'test'] = idx[-1]
    df_trans['test'] = df_trans['test'].replace({idx[i]: (i + 1) for i in range(len(idx))})
    print(sorted(list(df_trans['test'].unique())))
    print(count(df_trans, 'test', 1000))
    x = chiMerge(df_trans.sample(10000), 'test', 'label', 10)