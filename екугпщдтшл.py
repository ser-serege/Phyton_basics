x = int(input())
y = int(input())
z = int(input())

p = (x + y + z) / 2
s = (p*(p-x)*(p-y)*(p-z))**(1/2)

print('{0:.6f}'.format(s))


def grpb(X, col_to_group, what):
   
    aggregations = {
            'mean' :'mean', 
            'sum' : 'sum', 
            'size' :'size', 
            'min' : 'min', 
            'max' : 'max', 
            'median' : 'median', 
            'std' : 'std',
            'var' : 'var', 
            'skew' : 'skew',
            'sem' : 'sem',
            'prod' : 'prod',
            'mad' : 'mad',
            'nunique' : 'nunique',
            'sred' : lambda x: max(x) - np.mean(x), 
            'sred_np_log' : lambda x: np.mean(np.log(x)),
            'quantile' : lambda x: x.quantile(0.05)
            # 'cummax': 'cummax',
            # 'cumsum': 'cumsum',
            # 'cumprod': 'cumprod', 
            # 'shift' : 'shift(-1)',
            # 'diff' : 'diff(1)',
            # 'first' : 'first',
            # 'last' : 'last',
            # 'pct_change' : 'pct_change',
            # 'nth' : 'nth' ,
            # 'rank' : 'rank'
            # .filter(lambda x: len(x) >= 26000)
                    }
            
    grouped = X.fillna(1).groupby(col_to_group)[what].agg(aggregations)
    grouped.columns = [[str(col_to_group) + '_' + x  for x in grouped.columns]]
    grouped = grouped.reset_index()
    return grouped

a = ['a', 'b', 'c', 'd', 'e']

for i in a:
    globals()[i] = grpb(X, i, 'f')   

a
	
