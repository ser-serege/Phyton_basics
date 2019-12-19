from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd
import sys
from IPython.core.display import HTML
    
def plot_pandas_style(styler):

    html = '\n'.join([line.lstrip() for line in styler.render().split('\n')])
    return HTML(html)

def highlight_max(s,color='yellow'):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: {}'.format(color) if v else '' for v in is_max]

def decile_labels(agg1,label,color='skyblue'):
    agg_dummy = pd.DataFrame(OrderedDict((('TOTAL',0),('TARGET',0),('NONTARGET',0),('PCT_TAR',0),('CUM_TAR',0),('CUM_NONTAR',0),('DIST_TAR',0),('DIST_NONTAR',0),('SPREAD',0))),index=[0])
    agg1 = agg1.append(agg_dummy).sort_index()
    agg1.index.name = label
    agg1 = agg1.style.apply(highlight_max, color = 'yellow', subset=['SPREAD'])
    agg1.bar(subset=['TARGET'], color='{}'.format(color))
    agg1.bar(subset=['TOTAL'], color='{}'.format(color))
    agg1.bar(subset=['PCT_TAR'], color='{}'.format(color))
    return(agg1)

def deciling(data,decile_by,target,nontarget):
    inputs = list(decile_by)
    inputs.extend((target,nontarget))
    decile = data[inputs]
    grouped = decile.groupby(decile_by)
    agg1 = pd.DataFrame({},index=[])
    agg1['TOTAL'] = grouped.sum()[nontarget] + grouped.sum()[target]
    agg1['TARGET'] = grouped.sum()[target]
    agg1['NONTARGET'] = grouped.sum()[nontarget]
    agg1['PCT_TAR'] = grouped.mean()[target]*100
    agg1['CUM_TAR'] = grouped.sum()[target].cumsum()
    agg1['CUM_NONTAR'] = grouped.sum()[nontarget].cumsum()
    agg1['DIST_TAR'] = agg1['CUM_TAR']/agg1['TARGET'].sum()*100
    agg1['DIST_NONTAR'] = agg1['CUM_NONTAR']/agg1['NONTARGET'].sum()*100
    agg1['SPREAD'] = (agg1['DIST_TAR'] - agg1['DIST_NONTAR'])
    agg1 = decile_labels(agg1,'DECILE',color='skyblue')
    return(plot_pandas_style(agg1))

def deciling_table(data,decile_by,target,nontarget):
    inputs = list(decile_by)
    inputs.extend((target,nontarget))
    decile = data[inputs]
    grouped = decile.groupby(decile_by)
    agg1 = pd.DataFrame({},index=[])
    agg1['TOTAL'] = grouped.sum()[nontarget] + grouped.sum()[target]
    agg1['TARGET'] = grouped.sum()[target]
    agg1['NONTARGET'] = grouped.sum()[nontarget]
    agg1['PCT_TAR'] = grouped.mean()[target]*100
    agg1['CUM_TAR'] = grouped.sum()[target].cumsum()
    agg1['CUM_NONTAR'] = grouped.sum()[nontarget].cumsum()
    agg1['DIST_TAR'] = agg1['CUM_TAR']/agg1['TARGET'].sum()*100
    agg1['DIST_NONTAR'] = agg1['CUM_NONTAR']/agg1['NONTARGET'].sum()*100
    agg1['SPREAD'] = (agg1['DIST_TAR'] - agg1['DIST_NONTAR'])
    #agg1 = decile_labels(agg1,'DECILE',color='skyblue')
    return agg1

def scoring(features,clf,target):
    score = pd.DataFrame(clf.predict_proba(features)[:,1], columns = ['SCORE'])
    score['DECILE'] = pd.qcut(score['SCORE'].rank(method = 'first'),10,labels=range(10,0,-1))
    score['DECILE'] = score['DECILE'].astype(float)
    score['TARGET'] = target
    score['NONTARGET'] = 1 - target
    return(score)


def plots(agg1,target,type):

    plt.figure(1,figsize=(20, 5))

    plt.subplot(131)
    plt.plot(agg1['DECILE'],agg1['ACTUAL'],label='Actual')
    plt.plot(agg1['DECILE'],agg1['PRED'],label='Pred')
    plt.xticks(range(10,110,10))
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.title('Actual vs Predicted', fontsize=20)
    plt.xlabel("Population %",fontsize=15)
    plt.ylabel(str(target) + " " + str(type) + " %",fontsize=15)

    plt.subplot(132)
    X = agg1['DECILE'].tolist()
    X.append(0)
    Y = agg1['DIST_TAR'].tolist()
    Y.append(0)
    plt.plot(sorted(X),sorted(Y))
    plt.plot([0, 100], [0, 100],'r--')
    plt.xticks(range(0,110,10))
    plt.yticks(range(0,110,10))
    plt.grid(True)
    plt.title('Gains Chart', fontsize=20)
    plt.xlabel("Population %",fontsize=15)
    plt.ylabel(str(target) + str(" DISTRIBUTION") + " %",fontsize=15)
    plt.annotate(round(agg1[agg1['DECILE'] == 30].DIST_TAR.item(),2),xy=[30,30], 
            xytext=(25, agg1[agg1['DECILE'] == 30].DIST_TAR.item() + 5),fontsize = 13)
    plt.annotate(round(agg1[agg1['DECILE'] == 50].DIST_TAR.item(),2),xy=[50,50], 
            xytext=(45, agg1[agg1['DECILE'] == 50].DIST_TAR.item() + 5),fontsize = 13)

    plt.subplot(133)
    plt.plot(agg1['DECILE'],agg1['LIFT'])
    plt.xticks(range(10,110,10))
    plt.grid(True)
    plt.title('Lift Chart', fontsize=20)
    plt.xlabel("Population %",fontsize=15)
    plt.ylabel("Lift",fontsize=15)

    plt.tight_layout()

def gains(data,decile_by,target,score):
    inputs = list(decile_by)
    inputs.extend((target,score))
    decile = data[inputs]
    grouped = decile.groupby(decile_by)
    agg1 = pd.DataFrame({},index=[])
    agg1['ACTUAL'] = grouped.mean()[target]*100
    agg1['PRED'] = grouped.mean()[score]*100
    agg1['DIST_TAR'] = grouped.sum()[target].cumsum()/grouped.sum()[target].sum()*100
    agg1.index.name = 'DECILE'
    agg1 = agg1.reset_index()
    agg1['DECILE'] = agg1['DECILE']*10
    agg1['LIFT'] = agg1['DIST_TAR']/agg1['DECILE']
    plots(agg1,target,'Distribution')

def gains_table(data,decile_by,target,score):
    inputs = list(decile_by)
    inputs.extend((target,score))
    decile = data[inputs]
    grouped = decile.groupby(decile_by)
    agg1 = pd.DataFrame({},index=[])
    agg1['ACTUAL'] = grouped.mean()[target]*100
    agg1['PRED'] = grouped.mean()[score]*100
    agg1['DIST_TAR'] = grouped.sum()[target].cumsum()/grouped.sum()[target].sum()*100
    agg1['DIST_SCORE'] = grouped.sum()[score].cumsum()/grouped.sum()[score].sum()*100
    agg1.index.name = 'DECILE'
    agg1 = agg1.reset_index()
    agg1['DECILE'] = agg1['DECILE']*10
    agg1['LIFT'] = agg1['DIST_TAR']/agg1['DECILE']
    agg1['LIFT2'] = agg1['DIST_SCORE']/agg1['DECILE']
    return agg1


def plots(agg1,target,type):

    plt.figure(1,figsize=(20, 5))

    plt.subplot(131)
    plt.plot(agg1['DECILE'],agg1['ACTUAL'],label='Actual')
    plt.plot(agg1['DECILE'],agg1['PRED'],label='Pred')
    plt.xticks(range(10,110,10))
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.title('Actual vs Predicted', fontsize=20)
    plt.xlabel("Population %",fontsize=15)
    plt.ylabel(str(target) + " " + str(type) + " %",fontsize=15)

    plt.subplot(132)
    X = agg1['DECILE'].tolist()
    X.append(0)
    Y = agg1['DIST_TAR'].tolist()
    Y.append(0)
    plt.plot(sorted(X),sorted(Y))
    plt.plot([0, 100], [0, 100],'r--')
    plt.xticks(range(0,110,10))
    plt.yticks(range(0,110,10))
    plt.grid(True)
    plt.title('Gains Chart', fontsize=20)
    plt.xlabel("Population %",fontsize=15)
    plt.ylabel(str(target) + str(" DISTRIBUTION") + " %",fontsize=15)
    plt.annotate(round(agg1[agg1['DECILE'] == 30].DIST_TAR.item(),2),xy=[30,30], 
            xytext=(25, agg1[agg1['DECILE'] == 30].DIST_TAR.item() + 5),fontsize = 13)
    plt.annotate(round(agg1[agg1['DECILE'] == 50].DIST_TAR.item(),2),xy=[50,50], 
            xytext=(45, agg1[agg1['DECILE'] == 50].DIST_TAR.item() + 5),fontsize = 13)

    plt.subplot(133)
    plt.plot(agg1['DECILE'],agg1['LIFT'])
    plt.xticks(range(10,110,10))
    plt.grid(True)
    plt.title('Lift Chart', fontsize=20)
    plt.xlabel("Population %",fontsize=15)
    plt.ylabel("Lift",fontsize=15)

    plt.tight_layout()

def gains(data,decile_by,target,score):
    inputs = list(decile_by)
    inputs.extend((target,score))
    decile = data[inputs]
    grouped = decile.groupby(decile_by)
    agg1 = pd.DataFrame({},index=[])
    agg1['ACTUAL'] = grouped.mean()[target]*100
    agg1['PRED'] = grouped.mean()[score]*100
    agg1['DIST_TAR'] = grouped.sum()[target].cumsum()/grouped.sum()[target].sum()*100
    agg1.index.name = 'DECILE'
    agg1 = agg1.reset_index()
    agg1['DECILE'] = agg1['DECILE']*10
    agg1['LIFT'] = agg1['DIST_TAR']/agg1['DECILE']
    plots(agg1,target,'Distribution')

def gains_table(data,decile_by,target,score):
    inputs = list(decile_by)
    inputs.extend((target,score))
    decile = data[inputs]
    grouped = decile.groupby(decile_by)
    agg1 = pd.DataFrame({},index=[])
    agg1['ACTUAL'] = grouped.mean()[target]*100
    agg1['PRED'] = grouped.mean()[score]*100
    agg1['DIST_TAR'] = grouped.sum()[target].cumsum()/grouped.sum()[target].sum()*100
    agg1['DIST_SCORE'] = grouped.sum()[score].cumsum()/grouped.sum()[score].sum()*100
    agg1.index.name = 'DECILE'
    agg1 = agg1.reset_index()
    agg1['DECILE'] = agg1['DECILE']*10
    agg1['LIFT'] = agg1['DIST_TAR']/agg1['DECILE']
    agg1['LIFT2'] = agg1['DIST_SCORE']/agg1['DECILE']
    return agg1



def plot_graph(fpr, tpr, fpr1, tpr1, model, model1, gini, gini1, df):# , DECILE, DIST_TAR, DIST_SCORE, LIFT, LIFT2

    plt.figure(1,figsize=(20, 5))
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1)

    plt.subplot(131)
    plt.plot(fpr, tpr ,label=model)
    plt.plot(fpr1, tpr1 ,label=model1)
    plt.plot([0, 1], [0, 1], color='gray',  linestyle='--')
    plt.grid(True)
    plt.title('ROC AUC ', fontsize=20)
    plt.xlabel("Population %",fontsize=15)
    plt.ylabel("Lift",fontsize=15)
    plt.legend(fontsize=12, loc = 'center right')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.annotate('GINI_' + str(model) + '= ' + str(round(gini,2)),
                 xy=(0.6,  0.03),
                 xytext=(0.6,  0.03),fontsize = 14,
                bbox=bbox_props)

    plt.annotate('GINI_' + str(model1) + '= '+ str(round(gini1,2)),
                 xy=(0.6,  0.12),
                 xytext=(0.6,  0.12),fontsize = 14,
                bbox=bbox_props)


    ##########-----------------------------##########--------------------##########--------------------##########

    plt.subplot(132)
    plt.plot(df['DECILE'], df.LIFT, label=model)
    plt.plot(df['DECILE'], df.LIFT2, label=model1)
    plt.xticks(range(10, 110, 10))
    plt.grid(True)
    plt.title('Lift', fontsize= 20)
    plt.xlabel("Population %", fontsize= 15)
    plt.ylabel("Lift", fontsize= 15)


    x = 10
    y = round(df[df['DECILE'] == 10].LIFT.item(), 2)
    plt.plot(x, y, 'o', color= 'blue')

    x1 = 20
    y1 = round(df[df['DECILE'] == 20].LIFT.item(), 2)
    plt.plot(x1, y1, 'o', color= 'blue')

    x5 = 30
    y5 = round(df[df['DECILE'] == 30].LIFT.item(), 2)
    plt.plot(x5, y5, 'o', color= 'blue')


    x2 = 10
    y2 = round(df[df['DECILE'] == 10].LIFT2.item(), 2)
    plt.plot(x2, y2, 'o', color='orange')

    x3 = 20
    y3 = round(df[df['DECILE'] == 20].LIFT2.item(), 2)
    plt.plot(x3, y3, 'o', color='orange')

    x4 = 30
    y4 = round(df[df['DECILE'] == 30].LIFT2.item(), 2)
    plt.plot(x4, y4, 'o', color='orange')

    plt.annotate(round(df[df['DECILE'] == 10].LIFT.item(),1),
                 xy=(12, 5),
                 xytext=(12, df[df['DECILE'] == 10].LIFT.item() + 0),fontsize = 14,
                bbox=bbox_props)

    plt.annotate(round(df[df['DECILE'] == 20].LIFT.item(),1),
                 xy=(12, 5),
                 xytext=(22, df[df['DECILE'] == 20].LIFT.item() + 0.1),fontsize = 14,
                bbox=bbox_props)

    plt.annotate(round(df[df['DECILE'] == 30].LIFT.item(),1),
                 xy=(12, 5),
                 xytext=(32, df[df['DECILE'] == 30].LIFT.item() + 0.1),fontsize = 14,
                bbox=bbox_props)

    plt.annotate(round(df[df['DECILE'] == 10].LIFT2.item(),1),
                 xy=(12, 5),
                 xytext=(12, df[df['DECILE'] == 10].LIFT2.item() + 0.1),fontsize = 14,
                bbox=bbox_props)

    plt.annotate(round(df[df['DECILE'] == 20].LIFT2.item(),1),
                 xy=(12, 5),
                 xytext=(22, df[df['DECILE'] == 20].LIFT2.item() + 0.1),fontsize = 14,
                bbox=bbox_props)

    plt.annotate(round(df[df['DECILE'] == 30].LIFT2.item(),1),
                 xy=(12, 5),
                 xytext=(32, df[df['DECILE'] == 30].LIFT2.item() + 0.3),fontsize = 14,
                bbox=bbox_props)

    # lift top10 top20 top30
#     plt.annotate('Lift Top10 =' + str(round(df[df['DECILE'] == 10].LIFT2.item() /  df[df['DECILE'] == 10]\
#                                                                                                      .LIFT.item(),2)),
#                  xy=(12, 5),
#                  xytext=(75, 2.5),fontsize = 14,
#                 bbox=bbox_props)
#     plt.annotate('Lift Top20 =' + str(round(df[df['DECILE'] == 20].LIFT2.item() /  df[df['DECILE'] == 20]\
#                                                                                                      .LIFT.item(),2)),
#                  xy=(12, 5),
#                  xytext=(75, 2),fontsize = 14,
#                 bbox=bbox_props)

#     plt.annotate('Lift Top30 =' + str(round(df[df['DECILE'] == 30].LIFT2.item() /  df[df['DECILE'] == 30]\
#                                                                                                      .LIFT.item(),2)),
#                  xy=(12, 5),
#                  xytext=(75, 1.5),fontsize = 14,
#                 bbox=bbox_props)


    plt.legend(fontsize=12, loc='center right')

    ##########-----------------------------##########--------------------##########--------------------##########

    plt.subplot(133)
    X = df.DECILE.tolist()
    X.append(0)
    Y = df.DIST_TAR.tolist()
    Y.append(0)
    Z = df.DIST_SCORE.tolist()
    Z.append(0)
    plt.plot(sorted(X),sorted(Y) , label = model)
    plt.plot(sorted(X),sorted(Z), label = model1)
    plt.plot([0, 100], [0, 100],'r--')
    plt.xticks(range(0,110,10))
    plt.yticks(range(0,110,10))
    plt.grid(True)
    plt.title('GAINS', fontsize=20)
    plt.xlabel("Population %",fontsize=15)
    plt.ylabel(str("Target distribution") + " %",fontsize=15)

    x = 30
    y = round(df[df['DECILE'] == 30].DIST_TAR.item(),2)
    plt.plot(x, y, 'o', color='blue')

    x1 = 50
    y1 = round(df[df['DECILE'] == 50].DIST_TAR.item(),2)
    plt.plot(x1, y1, 'o', color='blue')


    x2 = 30
    y2 = round(df[df['DECILE'] == 30].DIST_SCORE.item(),2)
    plt.plot(x2, y2, 'o', color='orange')

    x3 = 50
    y3 = round(df[df['DECILE'] == 50].DIST_SCORE.item(),2)
    plt.plot(x3, y3, 'o', color='orange')

    plt.annotate(round(df[df['DECILE'] == 30].DIST_TAR.item(),1), xy=[5,5], 
                xytext=(30, df[df['DECILE'] == 30].DIST_TAR.item() - 10),fontsize = 14,
                bbox=bbox_props)

    plt.annotate(round(df[df['DECILE'] == 50].DIST_TAR.item(),1),xy=[5,5], 
                xytext=(50, df[df['DECILE'] == 50].DIST_TAR.item() - 11),fontsize = 14,
                bbox=bbox_props)

    plt.annotate(round(df[df['DECILE'] == 30].DIST_SCORE.item(),1), xy=[5,5], 
                xytext=(30, df[df['DECILE'] == 30].DIST_SCORE.item() + 5),fontsize = 14,
                bbox=bbox_props)

    plt.annotate(round(df[df['DECILE'] == 50].DIST_SCORE.item(),1),xy=[5,5], 
                xytext=(50, df[df['DECILE'] == 50].DIST_SCORE.item() + 5),fontsize = 14,
                bbox=bbox_props
                )

    plt.legend(fontsize=12, loc='center right')

    plt.tight_layout()
    plt.savefig('foo.png')