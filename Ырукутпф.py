a = [int(i) for i in input().split()]
x = int(input())
pos = 0
while pos < len(a) and a[pos] >= x:
    pos += 1
print(pos + 1)


def crotab(df, col):
    p = pd.crosstab(df['inn_roo'], df['rn'], values=df[col], aggfunc='mean')
    
#     try:
#         p['1_2'] = p[1] / p[2]
#     except: pass
    
#     try:
#         p['2_3'] = p[2] / p[3]
#     except: pass
    
    try:
        p['3_4'] = p[3] / p[4]
    except: pass
    
    try:
        p['4_5'] = p[4] / p[5]
    except: pass
    
    try:
        p['5_6'] = p[5] / p[6]
    except: pass
    
    try:
        p['6_7'] = p[6] / p[7]
    except: pass
    
#     try:
#         p['1_is<1'] = np.where(p['1_2']< 1, 1,0)
#     except: pass
    
#     try:
#         p['2_is<1'] = np.where(p['2_3']< 1, 1,0)
#     except: pass
    
    try:
        p['3_is<1'] = np.where(p['3_4']< 1, 1,0)
    except: pass
    
    try:
        p['4_is<1'] = np.where(p['4_5']< 1, 1,0)
    except: pass
    
    try:
        p['5_is<1'] = np.where(p['5_6']< 1, 1,0)
    except: pass
    
    try:
        p['6_is<1'] = np.where(p['6_7']< 1, 1,0)
    except: pass
    
    p.columns = [str(col) + '_' + str(x)  for x in list(p.columns)]
    p = p.iloc[:,13:20]
    return p

def merg(df1, df2):
    df1 = df1.join(df2)
    
сategorical_feature_mask = df.dtypes==object
# filter categorical columns using mask and turn it into a list
categorical_cols = df.columns[categorical_feature_mask].tolist()
num_cols = df.columns[~categorical_feature_mask].tolist()

l = crotab(df, 'CM_nb_m3')
DF = pd.DataFrame(index = l.index)

for i in num_cols:
    l = crotab(df, i)
    DF = DF.join(l)
DF.head()


grouped = df.groupby('inn_roo')['CM_trans_nb_top_10%'].agg (['mean', 'median', 'max', 'min', 'size'])
grouped.columns = [[str('CM_trans_nb_top_10%') + '_' + x  for x in grouped.columns]]

for i in for_group:
    grouped2 = df.groupby('inn_roo')[i].agg (['mean', 'median', 'max', 'min', 'size'])
    grouped2.columns = [[str(i) + '_' + x  for x in grouped2.columns]]
    grouped = grouped.join(grouped2)
    
# Напишем функцию для построения модели и подсчета статистик
def mod(df, target, max_depth, n_estimators):
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score

    y = df[target]
    X = df.drop(target, axis =1)

    train, test, y_train, y_test = train_test_split(X, y, test_size =0.33,  stratify = y)

#     std = StandardScaler()
#     train = pd.DataFrame(std.fit_transform(train))
#     test = pd.DataFrame(std.transform(test))
#     train.columns = X.columns
#     test.columns = X.columns
    
    #rfc = LogisticRegression()
    #rfc = RandomForestClassifier(n_jobs=-1, random_state=42, max_depth=max_depth, n_estimators=n_estimators)
    #rfc = GradientBoostingClassifier(learning_rate = 0.01, n_estimators= n_estimators)
    rfc = CatBoostClassifier(iterations=1000, learning_rate=0.01, depth=3 ,random_seed=42 ,scale_pos_weight = 1)
                                #loss_function= 'Logloss', eval_metric='AUC')
    #rfc = DecisionTreeClassifier()
    rfc.fit(train, y_train)

    y_pred_train = rfc.predict_proba(train)
    y_pred_test = rfc.predict_proba(test)
    
    tot_pred = rfc.predict_proba(X)[:,1]
    
    tot = rfc.predict(X)

    metrics = {
        'Gini_test' : 2 * roc_auc_score(y_test, y_pred_test[:,1]) - 1 ,
        'Gini_train': 2 * roc_auc_score(y_train, y_pred_train[:,1]) - 1,
        'Precision' : precision_score(y, tot ) ,
        'Recall' :  recall_score(y, tot),
        'Accuracy': accuracy_score(y, tot)
        }
    

    importances_rfc = pd.DataFrame(abs(rfc.feature_importances_.T), train.columns).reset_index()
    importances_rfc.columns='feature', 'score'
    importances_rfc = importances_rfc.sort_values(by='score', ascending =False)

#     importances_rfc = pd.DataFrame(abs(rfc.coef_.T), train.columns).reset_index()
#     importances_rfc.columns='feature', 'score'
#     importances_rfc = importances_rfc.sort_values(by='score', ascending =False)

    # вернем предсказания, метрики, важность фичей, и саму модель
    return tot_pred, metrics, importances_rfc, rfc
