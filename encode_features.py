from merge_utils import *


class FeatureEncode:
    def __init__(self, label):
        self.label = label
        self.code_map_dict = dict()

    def replace_encode(self, data, f, cands):
        data['%s#encode' % f] = (data[f].isin(cands) * data[f]).replace(
            {cands[i]: (i + 1) for i in range(len(cands))})

    def merge_encode(self, data, f, tol, max_intervals, include_nan=False):
        if f not in self.code_map_dict:
            merge_value_lst = [x for x in highFreq(data, f, self.label, tol, plot=False)
                               if x != 999999]
            merge_value = {}
            for i in range(len(merge_value_lst)):
                if str(merge_value_lst[i]) == 'nan':
                    merge_value['nan'] = i
                else:
                    merge_value[merge_value_lst[i]] = i
            merge_idx = data[f].isin(merge_value) | data[f].isnull()
            data_tmp = data[[self.label, f]].copy()
            if include_nan:
                data_tmp[f].fillna('nan', inplace=True)
            data_tmp.loc[merge_idx, f] = data_tmp.loc[merge_idx, f].map(merge_value)
            intervals_info = chiMerge(data_tmp[merge_idx], f, self.label, max_intervals)
            count_table = data[[self.label, f]].groupby(f).agg(['count', 'sum'])[self.label]
            count_table['rate'] = count_table['sum'] / count_table['count']
            code_map = {}
            if include_nan and 'nan' in merge_value:
                idx = merge_value['nan']
                for x in intervals_info:
                    if idx >= intervals_info[x]['bound'][0] and idx <= intervals_info[x]['bound'][1]:
                        code_map['nan'] = x
                        break
            for k, v in count_table.iterrows():
                if str(k) == 'nan':
                    k = 'nan'
                code = np.nan
                if k in merge_value:
                    idx = merge_value[k]
                    for x in intervals_info:
                        if idx >= intervals_info[x]['bound'][0] and idx <= intervals_info[x]['bound'][1]:
                            code = x
                            break
                else:
                    gap = {x: abs(v['rate'] - intervals_info[x]['num'][1] / intervals_info[x]['num'][0])
                           for x in intervals_info}
                    code = sorted(gap, key=lambda x: gap[x])[0] if gap else np.nan
                code_map[k] = code
            self.code_map_dict[f] = code_map
        if include_nan:
            res = data[f].fillna('nan').map(self.code_map_dict[f])
        else:
            res = data[f].map(self.code_map_dict[f])
        return res

    def main_trans(self, data):
        n = data.shape[0]
        for col in ['ProductCD', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain'] + \
                   ['card%i' % i for i in range(1, 7)] + \
                   ['M%i' % i for i in range(1, 10)]:
            data['%s#encode' % col] = self.merge_encode(data, col, 3000, 10)

    def main_id(self, data):
        for col in ['DeviceType', 'DeviceInfo'] + ['id_%i' % i for i in range(12, 39)]:
            data['%s#encode' % col] = self.merge_encode(data, col, 3000, 10, True)


if __name__ == '__main__':
    df_trans = pd.read_csv('./data/train_transaction.csv')
    df_trans.rename({'isFraud':'label'}, inplace=True, axis=1)
    E = FeatureEncode('label')
    df_trans['addr1#encode'] = E.merge_encode(df_trans, 'addr1', 5000, 10)
    print(count(df_trans, 'addr1#encode', 200))