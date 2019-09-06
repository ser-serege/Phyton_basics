a = int(input())
b = int(input())
c = int(input())
d = int(input())

if 0 < a < 9 and 0 < b < 9 and 0 < c < 9 and 0 < d < 9:

    if a == c and b + 1 == d:
        print('YES')
    elif a == c and b - 1 == d:
        print('YES')
    elif a + 1 == c and b - 1 == d:
        print('YES')
    elif a + 1 == c and b + 1 == d:
        print('YES')
    elif a - 1 == c and b + 1 == d:
        print('YES')
    elif a - 1 == c and b - 1 == d:
        print('YES')
    elif a + 1 == c and b == d:
        print('YES')
    elif a - 1 == c and b == d:
        print('YES')
    else:
        print('NO')
else:
    print('Числа должны быть от 1 до 8')

    
    
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.utils import resample
import random
from sklearn import decomposition

df = pd.read_csv('X_data.csv', delimiter=';', parse_dates=['Unnamed: 0'])
df.rename_axis({'Unnamed: 0':'time'}, axis='columns', inplace=True)
y = pd.read_csv('Y_train.csv', delimiter=';', header=None, names =['time', 'target'], parse_dates=[0])

#df['time'].dt.date

df['n'] = df['time'].apply(lambda x: x.strftime("%Y-%m-%d %H"))
y['n'] = y['time'].apply(lambda x: x.strftime("%Y-%m-%d %H"))

#df['time'][1].strftime("%Y-%m-%d %H")

aa = df.columns.tolist()
aa.remove('time')
#aa.remove('date')
aa.remove('n')
aa.remove('T_data_1_1')

y['d'] = y['time'].dt.date

Y = y.groupby('d')['target'].agg({'min', 'max', 'std', 'size'}).reset_index()
Y['std'].std()

d = df.groupby('n')['T_data_1_1'].agg({'mean', 'median', 'std', 'min', 'max'}).reset_index()
d['diff_min_max'] = d['max'] - d['min']
a, b = [d.columns[0]] , [str('T_data_1_1') + x for x in d.columns[1:].tolist()]
[a.append(y) for y in b]
d.columns = a


for i in aa:
    z = df.groupby('n')[i].agg({'mean', 'median', 'std', 'min', 'max'}).reset_index()
    z['diff_min_max'] = z['max'] - z['min']
    a, b = [z.columns[0]] , [str(i) + str('_') + x for x in z.columns[1:].tolist()]
    [a.append(y) for y in b]
    z.columns = a
    
    d = d.merge(z, on='n')

train, test, y_train, y_test = train_test_split(zz.drop('target', axis=1), zz['target'], test_size = 0.33)
rfc = CatBoostRegressor()
rfc.fit(train, y_train)

y_pred_train = rfc.predict(train)
y_pred_test = rfc.predict(test)

# RandomForest
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_train, y_pred_train))
print(mean_absolute_error(y_test, y_pred_test))

from sklearn.preprocessing import PolynomialFeatures
polynomial_features= PolynomialFeatures(degree=2)
x_poly = pd.DataFrame(polynomial_features.fit_transform(train))
x_poly_test = pd.DataFrame(polynomial_features.fit_transform(test))

x_poly.columns = polynomial_features.get_feature_names()
x_poly_test.columns = polynomial_features.get_feature_names()


train, test, y_train, y_test = train_test_split(zz.drop('target', axis=1), zz['target'], test_size = 0.33)
rfc = RandomForestRegressor()
rfc.fit(x_poly, y_train)

y_pred_train = rfc.predict(x_poly)
y_pred_test = rfc.predict(x_poly_test)

# XGB Polynomial 
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_train, y_pred_train))
print(mean_absolute_error(y_test, y_pred_test))

importances_rfc = pd.DataFrame(abs(rfc.feature_importances_.T), polynomial_features.get_feature_names()).reset_index()
importances_rfc.columns='feature', 'score'
importances_rfc = importances_rfc.sort_values(by='score', ascending =False)
importances_rfc

cols = importances_rfc[importances_rfc['score']> 0.0005]['feature'].tolist()

def drop_corr_col(df, treshold , persent ):
    # n выбирается на какой выборке строить корреляции в процентах
    a = [random.choice(df.index) for i in range(int(len(df)*persent))]
    tmp = df.iloc[a]
    tmp.fillna(0, inplace=True)
    to_drop_after_corr = pd.DataFrame(tmp.corr())
    upper = tmp.where(np.triu(np.ones(tmp.shape), k = 1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column]> treshold)]
    df_ = df.drop(to_drop, axis =1)
    return to_drop_after_corr, df_, to_drop

to_drop_after_corr, df_, to_drop = drop_corr_col(ddd, 0.9955, 0.8)


from sklearn.decomposition import PCA

pca = PCA(n_components=5)
pca.fit(ddd)
X_pca = pca.transform(ddd)

X_pca_test = pca.transform(x_poly_test[cols])

ddd = train.reset_index().join(pd.DataFrame(X_pca))
