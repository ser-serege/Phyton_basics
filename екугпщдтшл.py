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
	

	
def get_ROC_curve(predicted, true):

    from ipywidgets import interact
    from bokeh.plotting import figure
    from bokeh.io import push_notebook, show, output_notebook
    output_notebook()
    from sklearn import metrics
    from sklearn.metrics import accuracy_score
    import matplotlib as plt
    
    
    # get number of positives and negatives:  
    # количество значений
    total_values = len(predicted)
    total_positive = len(np.where(true > 0)[0]);
    total_negative = total_values - total_positive;
    
    # сортируем значения по вероятность от меньшего к большему:
    sorted_index = np.argsort(predicted);
    sorted_predicted = predicted[sorted_index];
    sorted_true  = true[sorted_index];
    
    # создаем переменные true positive rate и false positive rate размера как values
    TPR = np.zeros([total_values, 1]);
    FPR = np.zeros([total_values, 1]);
 
    for i in range(total_values):
        # threshold = s_values[e]
        # Positive when bigger:
        positive = np.sum(sorted_true[i:]);
        TPR[i] = positive / total_positive;
        
        # number of false positives is the remaining samples above the
        # threshold divided by all negative samples:
        FPR[i] = (len(sorted_true[i:]) - positive) / total_negative;
 
    FPR = FPR.flatten()
    TPR = TPR.flatten()
    auc = metrics.auc(FPR,TPR)
    p = figure(title="ROC Curve - Train data")
    r = p.line(list(FPR), list(TPR), color='red', legend = 'AUC = '+ str(round(auc,3)), line_width=2)
    s = p.line([0,1],[0,1], color= '#d15555',line_dash='dotdash',line_width=2)
    show(p)
    
    return TPR, FPR;

TPR, FPR = get_ROC_curve(result, y)
