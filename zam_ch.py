
for i in range(10, 100):
    b = i // 10
    l = i % 10
    if b * l * 2 == i:
        print(i)

        
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", 100)
pd.set_option('display.max_rows', 30)
pd.options.display.float_format = '{:,.2f}'.format
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
pd.options.display.max_columns =100
import pyodbc
import matplotlib.pyplot as plt
from matplotlib.pylab import rc, plot
import seaborn as sns ; sns.set(style='white')
%matplotlib inline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest,  f_regression, f_classif, chi2
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix , roc_auc_score, roc_curve, auc, precision_score, recall_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.utils import resample
import random
from sklearn import decomposition

fin = pd.read_csv('urlica.csv', encoding = 'cp1251')

fin.head()

def plot_roc_curve(fprs, tprs):
    
    # Initialize useful lists + the plot axes.
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(8, 5))
    
    # Plot ROC for each K-Fold + compute AUC scores.
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,  label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        
    # Plot the luck line.
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    
    # Plot the mean ROC.
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
    
    # Plot the standard deviation around the mean ROC.
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    
    # Fine tune and show the plot.
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    plt.show()
    return (f, ax)
	
def dawnsampling(df, target):  

    # пропуски NaN заполним нулями, т.к. это пусте значения
    df.fillna(0, inplace=True)

    df_majority = df[df[target] ==0]
    df_minority = df[df[target]==1]

    df_majority_downsampled = resample(df_majority, replace=False, n_samples= len(df_minority)*3 , random_state=42) 
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    #print(df_downsampled.target.value_counts()) 
    return df_downsampled

def choose_model(train, test, model = 'RF' ):
    
    if model == 'LR':
        rfc = LogisticRegression()
        std = StandardScaler()
        train = pd.DataFrame(std.fit_transform(train))
        test = pd.DataFrame(std.transform(test))
        train.columns = X.columns
        test.columns = X.columns
    
    elif model == 'RF':
        rfc = RandomForestClassifier()
        #n_jobs=-1, random_state=4, max_depth=max_depth, n_estimators=n_estimators
    
    elif model == 'XGB':
        rfc = XGBClassifier()

    elif model == 'CatB':
        rfc = CatBoostClassifier()
        #iterations=1000, learning_rate=0.01, depth=3 ,random_seed=42 ,scale_pos_weight = 1
        
    return train, test, rfc

	
def cv(X, y, splits = 3, clf = rfc):
    
    cv = StratifiedKFold( n_splits = splits )
    results = pd.DataFrame(columns = ['training_score', 'test_score'])
    fprs, tprs, scores = [], [], []

    for train_index, test_index in cv.split(X, y):
        clf.fit(X.iloc[train_index],y.iloc[train_index])
        _, _, auc_score_train = compute_roc_auc(train_index, rfc)
        fpr, tpr, auc_score = compute_roc_auc(test_index, rfc)
        scores.append((auc_score_train, auc_score))
        fprs.append(fpr)
        tprs.append(tpr)
    
    CV = pd.DataFrame(scores, columns=['AUC Train', 'AUC Test'])
    CV['AUC_diff'] = CV['AUC Train'] - CV['AUC Test']
    plot_roc_curve(fprs, tprs);
    print(CV)
    print('Standart_dev train = ', round(CV['AUC Train'].std(), 3))
    print('Standart_dev test  = ', round(CV['AUC Test'].std(), 3))
    print('Standart_dev diff  = ', round(CV['AUC_diff'].std(), 3))
    print('Number of samples in each fold ', 'train = ', len(train_index), ', test = ', len(test_index))
    print( 'Share of test = ', int(len(test_index) / len(train_index) * 100), '%')
    
    
    #return CV


def compute_roc_auc(index, rfc):
    y_predict = rfc.predict_proba(X.iloc[index])[:,1]
    fpr, tpr, thresholds = roc_curve(y.iloc[index], y_predict)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score

def plot_roc_curve(fprs, tprs):
    
    # Initialize useful lists + the plot axes.
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(8, 5))
    
    # Plot ROC for each K-Fold + compute AUC scores.
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,  label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        
    # Plot the luck line.
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    
    # Plot the mean ROC.
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
    
    # Plot the standard deviation around the mean ROC.
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    
    # Fine tune and show the plot.
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    plt.show()
    return (f, ax)
	
cv(train, y_train, splits = 5, clf = rfc)

# Функция по построению зависимость скора от размера выборки
def dep_on_len_df(df, target, max_depth, n_estimators):
    
    gini_train, gini_test, len_df, k = [], [], [], np.linspace(0.1, 0.9, 10)
 
    for i in k:
        
        y = df[target]
        X = df.drop(target, axis =1)

        train, test, y_train, y_test = train_test_split(X, y, test_size = i,  stratify = y)
        rfc = RandomForestClassifier(n_jobs=-1, random_state=72, max_depth=max_depth, n_estimators=n_estimators)
        rfc.fit(train, y_train)

        y_pred_train = rfc.predict_proba(train)
        y_pred_test = rfc.predict_proba(test)

        Gini_train = 2 * roc_auc_score(y_train, y_pred_train[:,1]) - 1
        Gini_test = 2 * roc_auc_score(y_test, y_pred_test[:,1]) - 1
        
        gini_train.append(Gini_train)
        gini_test.append(Gini_test)
        len_df.append(i)

    # вернем предсказания, метрики, важность фичей, и саму модель
    return  gini_train, gini_test, len_df #tot_pred, metrics, importances_rfc, rfc, CV

gini_train, gini_test, len_df = dep_on_len_df(fin , 'ottok', 3, 1600);
plt.plot(len_df, gini_train, linestyle='--', lw=2, color='r',  alpha=.8)
plt.plot(len_df, gini_test, linestyle='-', lw=2, color='b',   alpha=.8);

def dep_on_num_of_features(df, target, max_depth, n_estimators ):
    
    gini_train, gini_test, len_df, k = [], [], [], [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
 
    f = []
    
    y = df[target]
    X = df.drop(target, axis =1)

    train, test, y_train, y_test = train_test_split(X, y, test_size = 33,  stratify = y)
    rfc = RandomForestClassifier(n_jobs=-1, random_state=72, max_depth=max_depth, n_estimators=n_estimators)
    rfc.fit(train, y_train)
    
    importances_rfc = pd.DataFrame(abs(rfc.feature_importances_.T), train.columns).reset_index()
    importances_rfc.columns='feature', 'score'
    importances_rfc = importances_rfc.sort_values(by='score', ascending =False)
    
    for i in k:
        
        y = df[target]
        X = df.drop(target, axis =1)

        features = importances_rfc[: int(len(importances_rfc) * i)]['feature'].tolist()
        X = X[features]
        
        train, test, y_train, y_test = train_test_split(X, y, test_size = 30,  stratify = y, random_state = 7)
        
        rfc = RandomForestClassifier(n_jobs=-1, random_state=72, max_depth=max_depth, n_estimators=n_estimators)
        rfc.fit(train, y_train)

        y_pred_train = rfc.predict_proba(train)
        y_pred_test = rfc.predict_proba(test)

        Gini_train = 2 * roc_auc_score(y_train, y_pred_train[:,1]) - 1
        Gini_test = 2 * roc_auc_score(y_test, y_pred_test[:,1]) - 1
        
        gini_train.append(Gini_train)
        gini_test.append(Gini_test)
        len_df.append(i)
        f.append(features)

    # вернем предсказания, метрики, важность фичей, и саму модель
    return  gini_train, gini_test, len_df, f #tot_pred, metrics, importances_rfc, rfc, CV

gini_train, gini_test, len_df, f  = dep_on_num_of_features(fin , 'ottok', 4, 600);
plt.plot(len_df, gini_train, linestyle='--', lw=2, color='r',  alpha=.8)
plt.plot(len_df, gini_test, linestyle='-', lw=2, color='b',   alpha=.8);


ax = plt.subplots(figsize=(8, 5))


ax = plot(len_df, gini_train, linestyle='--', lw=2, color='r',  alpha=.8, label='aasdasd')
ax = plot(len_df, gini_test, linestyle='-', lw=2, color='b',   alpha=.8)

#ax.set_xlabel('False Positive Rate')
#ax.set_ylabel('True Positive Rate')
#ax.set_title('Receiver operating characteristic')
#ax.legend(loc="lower right")
plt.show()


def fit_predict(train, y_train, test, model):
    model.fit(train, y_train)
    y_pred_train     = model.predict_proba(train)
    y_pred_test      = model.predict_proba(test)
    y_pred_train_bin = model.predict(train)
    y_pred_test_bin  = model.predict(test)
    return y_pred_train, y_pred_test, y_pred_train_bin, y_pred_test_bin, model

def choose_model(train, test, model = 'RF', params=None ):
    # example of parametres for random forest par = {'max_depth' : 3, 'criterion' : 'enthropy'}
    
    if model == 'LR':
        rfc = LogisticRegression(params)
        std = StandardScaler()
        train = pd.DataFrame(std.fit_transform(train))
        test = pd.DataFrame(std.transform(test))
        train.columns = train.columns
        test.columns = train.columns
    
    elif model == 'RF':
        rfc = RandomForestClassifier(params)
        #n_jobs=-1, random_state=4, max_depth=max_depth, n_estimators=n_estimators
    
    elif model == 'XGB':
        rfc = XGBClassifier(params)

    elif model == 'CatB':
        rfc = CatBoostClassifier(params)
        #iterations=1000, learning_rate=0.01, depth=3 ,random_seed=42 ,scale_pos_weight = 1
        
    return train, test, rfc

def metrics_calc(y_train, y_test, y_pred_test, y_pred_train, y_pred_test_bin, y_pred_train_bin):
    metrics = {
        'Gini_test'      : round (2 * roc_auc_score(y_test, y_pred_test[:,1])   - 1, 2),
        'Gini_train'     : round (2 * roc_auc_score(y_train, y_pred_train[:,1]) - 1, 2),
        'Precision_test' : round (precision_score  (y_test, y_pred_test_bin )      , 2),
        'Precision_train': round (precision_score  (y_train, y_pred_train_bin )    , 2),
        'Recall_test'    : round (recall_score     (y_test, y_pred_test_bin)       , 2),
        'Recall_train'   : round (recall_score     (y_train, y_pred_train_bin)     , 2),
        'Accuracy_test'  : round (accuracy_score   (y_test, y_pred_test_bin)       , 2),
        'Accuracy_train' : round (accuracy_score   (y_train, y_pred_train_bin)     , 2),
        'F1_score_test'  : round (f1_score(y_test, y_pred_test_bin)                , 2),
        'F1_score_train' : round (f1_score(y_train, y_pred_train_bin)              , 2) 
                }
    return metrics
	
	
def best_model(df, target, model_1, model_2, model_3, blending, stacking):
    
    y = df[target]
    X = df.drop(target, axis =1)
    train, test, y_train, y_test = train_test_split(X, y, test_size = 33)
    
    index_metrics = ['Gini_test', 'Gini_train', 'Precision_test', 'Precision_train', 'Recall_test',
    'Recall_train', 'Accuracy_test', 'Accuracy_train', 'F1_score_test', 'F1_score_train']
    
    all_metrics = pd.DataFrame(index = index_metrics)
    
    # Тренируем 1 модель
    if model_1:
        train, test, rfc_1 = choose_model(train, test, model = model_1)
        y_pred_train_1, y_pred_test_1, y_pred_train_bin_1, y_pred_test_bin_1, m_1  = fit_predict(train, y_train, test, rfc_1)
        metrics_1 = metrics_calc(y_train, y_test, y_pred_test_1, y_pred_train_1, y_pred_test_bin_1, y_pred_train_bin_1)
        all_metrics[model_1] = metrics_1.values()
    else: metrics_1 = 'Model_1 was not build'
    
    # Тренируем 2 модель
    if model_2:
        train, test, rfc_2 = choose_model(train, test, model = model_2)
        y_pred_train_2, y_pred_test_2, y_pred_train_bin_2, y_pred_test_bin_2, m_2  = fit_predict(train, y_train, test, rfc_2)
        metrics_2 = metrics_calc(y_train, y_test, y_pred_test_2, y_pred_train_2, y_pred_test_bin_2, y_pred_train_bin_2)
        all_metrics[model_2] = metrics_2.values()
    else: metrics_2 = 'Model_2 was not build'

    if model_3:
        train, test, rfc_3 = choose_model(train, test, model = model_3)
        y_pred_train_3, y_pred_test_3, y_pred_train_bin_3, y_pred_test_bin_3, m_3  = fit_predict(train, y_train, test, rfc_3)
        metrics_3 = metrics_calc(y_train, y_test, y_pred_test_3, y_pred_train_3, y_pred_test_bin_3, y_pred_train_bin_3)
        all_metrics[model_3] = metrics_3.values()
    else: metrics_3 = 'Model_3 was not build'
        
    if blending:
        y_pred_train_blended     = ( y_pred_train_1[:,1] + y_pred_train_2[:,1] + y_pred_train_3[:,1] ) / 3
        y_pred_test_blended      = ( y_pred_test_1[:,1]  + y_pred_test_2[:,1]  + y_pred_test_3[:,1] )  / 3
        
        
        y_pred_train_bin_blended = np.where(y_pred_train_blended > 0.5, 1, 0 )
        y_pred_test_bin_blended  = np.where(y_pred_test_blended  > 0.5, 1, 0 )
        metrics_blended = metrics_calc(y_train, y_test, y_pred_test_blended, y_pred_train_blended, 
                                 y_pred_test_bin_blended, y_pred_train_bin_blended)
        all_metrics['blending'] = metrics_blended.values()
    else: metrics_blended = 'Model_blended was not build'
        
        
    if stacking:
        # Добавить веса моделей в зависимости от метрик
        df_stacking_train = pd.DataFrame([y_pred_train_1[:,1], y_pred_train_2[:,1], y_pred_train_3[:,1]]).T
        df_stacking_test  = pd.DataFrame([y_pred_test_1[:,1], y_pred_test_2[:,1], y_pred_test_3[:,1]]).T
        lr = LogisticRegression()
        lr.fit(df_stacking_train, y_train)
        y_pred_train_stacking     = lr.predict_proba(df_stacking_train)
        y_pred_test_stacking      = lr.predict_proba(df_stacking_test)
        y_pred_train_bin_stacking = lr.predict(df_stacking_train)
        y_pred_test_bin_stacking  = lr.predict(df_stacking_test)
        metrics_stacking = metrics_calc(y_train, y_test, y_pred_test_stacking, y_pred_train_stacking, 
                                 y_pred_test_bin_stacking, y_pred_train_bin_stacking)
        all_metrics['stacking'] = metrics_stacking.values()
    else: metrics_stacking = 'Model_stacking was not build'
        
    
    return all_metrics
	
	
me = best_model(fin, 'ottok', model_1 = 'RF', model_2 = 'LR', model_3 = 'CatB', blending=False, stacking=True)



----------------------------------------
----------------------------------------
----------------------------------------
----------------------------------------
----------------------------------------

import pandas_profiling

fin[['Unnamed_0','CM_nb_m3_5_is<1', 'CM_nb_m3_6_is<1', 'LE_salary_m_5_is<1', 'LE_salary_m_6_is<1']].profile_report(style={'full_width':True})

# Найдем все не уникальные значения в столбцах------------------------------------------------
def un_unique(df):
    to_drop_nunique = []
    for i in (df.columns):
        if df[i].nunique() == 1:
            to_drop_nunique.append(i)
    df.drop(to_drop_nunique, axis =1) 
    return  df, to_drop_nunique

# Удалим явно кореллированные переменные
def drop_corr_col(df, treshold , persent ):
    # n выбирается на какой выборке строить корреляции в процентах
    a = [random.choice(df.index) for i in range(int(len(df)*persent))]
    tmp = df.iloc[a]
    tmp.fillna(0, inplace=True)
    to_drop_after_corr = pd.DataFrame(tmp.corr().abs())
    upper = to_drop_after_corr.where(np.triu(np.ones(to_drop_after_corr.shape), k =1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > treshold)]
    df_ = df.drop(to_drop, axis = 1)
    del tmp, a, upper, df
    return df_, to_drop, to_drop_after_corr

def check_col_names(df):
    pass

# посмотреть на фичи, где от 2 до 5 значений. Являются ли они категориальными?------------------------------------------------
def check_for_cut_feat(df, nmin=2, nmax=5, retype=False): 
    num_unique=[]
    columns_list=[]
    for column in df.columns:
        num_unique.append(df[column].nunique())
        columns_list.append(column)
    cols_df = pd.DataFrame(num_unique,columns_list).reset_index()
    cols_df.columns = 'feature', 'num'

    cat_features = list(cols_df[(cols_df['num'] <= nmax) & (cols_df['num'] > nmin)]['feature'])
    print(len(cat_features))
    
    if retype:
        df_ = df.copy()
        for num_to_cat in cat_features:
            df_[num_to_cat] = df_[num_to_cat].astype(object)
    
    return df_, cat_features


# Выеделим категориальные переменные и числовые------------------------------------------------
def cut_num_feat(df): 
    cat_feat = list(df.dtypes[df.dtypes == object].index)
    num_feat = [f for f in df if f not in cat_feat]
    print('cat_feat =', len(cat_feat))
    print('num_feat =',len(num_feat))
    return cat_feat, num_feat


def null_means(df, treshold_to_delete, treshhold_to_analyze ):
    # delete with more than  @treshold_to_delete percents of empty values
    tmp = pd.DataFrame(df.isnull().sum().sort_values()).reset_index()
    tmp.columns = 'feature', 'score'
    to_drop = tmp[tmp['score'] > len(df) * treshold_to_delete]['feature'].tolist() 
    df_new = df.drop(to_drop, axis=1)
    # cols to analyze why there are a lot of empty values
    tmp = pd.DataFrame(df_new.isnull().sum().sort_values()).reset_index()
    tmp.columns = 'feature', 'score'
    cols_to_analyze = tmp[tmp['score'] > len(df) * treshhold_to_analyze]['feature'].tolist() 
    
    print('dropped with 70% of missings ', len(to_drop), 'columns')
    return df_new, to_drop, cols_to_analyze

#x_train, to_drop, cols_to_analyze = null_means(df, 0.6)



# Избавиться от выбрасов
def outlyers(df, columns, nmin= 0.05, nmax = 0.95, treshold = 0.001, del_or_min_max=True):
    col_name = []
    num_outlyers_min = []
    num_outlyers_max = []
    
    print('Number of values before cut = ', len(df))

    for column in df[columns].columns.tolist():
        quant_min = df[column].quantile(nmin)
        quant_max = df[column].quantile(nmax)
        
        if len(df[column][df[column] < quant_min]) < treshold * len(df) or \
                                                               len(df[column][df[column] > quant_max]) < treshold * len(df):
            if del_or_min_max:
                df[column][df[column] < quant_min ] = quant_min
                df[column][df[column] > quant_max ] = quant_max
            df = df[df[column].between(quant_min, quant_max)]
            
        col_name.append(column)
        num_outlyers_min.append(len(df[column][df[column] < quant_min]))
        num_outlyers_max.append(len(df[column][df[column] > quant_max]))
        
    outlyers_table = pd.DataFrame(index = col_name)
    outlyers_table['num_outlyers_min'] = num_outlyers_min
    outlyers_table['num_outlyers_max'] = num_outlyers_max
    outlyers_table['sum_outlyers'] = num_outlyers_min + num_outlyers_max
    
    print('Number of values after cut = ', len(df))
    
    out = outlyers_table[outlyers_table['sum_outlyers'] > outlyers_table['sum_outlyers'].quantile(0.75)].index.tolist()
    
    for i in out:
        sns.boxplot(x = fin[i])
        plt.show()        
        
    return df, outlyers_table



def identify_outlyers(df, min, max):
    feature=[]
    length=[]
    non_stable = pd.DataFrame()
    for i in df.columns:
        feature.append(i)
        length.append(len(df[~df[i].between(df[i].quantile(min), df[i].quantile(max))]))
        df = df[df[i].between(df[i].quantile(min), df[i].quantile(max))]
    non_stable['feature'] = feature # features which are out of the 
    non_stable['score'] = length
    print(len(df))
    return non_stable, df

df1, to_drop, to_drop_after_corr = drop_corr_col(fin.drop(['Unnamed_0', 'ottok'] , axis=1),  treshold = 0.6, persent = 1)
df, outlyers_table = outlyers(fin, fin.columns, del_or_min_max=True )
