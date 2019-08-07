x = int(input())
y = int(input())
z = int(input())

p = (x + y + z) / 2
s = (p*(p-x)*(p-y)*(p-z))**(1/2)

print('{0:.6f}'.format(s))



from sklearn.preprocessing import Imputer
application[numerical_list] = Imputer(strategy='median').fit_transform(application[numerical_list])

del application_train; gc.collect()
application = pd.get_dummies(application, drop_first=True)
print(application.shape)

X = application.drop(['SK_ID_CURR', 'TARGET'], axis=1)
y = application.TARGET
feature_name = X.columns.tolist()




def feature_selection(df, target):
	null_means
	un_unique
	drop_corr_col
	identify_outlyers
	pca
	selectkbest
	RF
	vote
	crossvalimportance
	ols_regression
	

def null_means(df, treshhold):
    tmp = pd.DataFrame(df.isnull().sum().sort_values()).reset_index()
    tmp.columns = 'feature', 'score'
    to_drop = tmp[tmp['score'] > len(df) * 0.7]['feature'].tolist() 
    df_new = df.drop(to_drop, axis=1)
    
    tmp = pd.DataFrame(df_new.isnull().sum().sort_values()).reset_index()
    tmp.columns = 'feature', 'score'
    cols_to_analyze = tmp[tmp['score'] > len(df) * treshhold]['feature'].tolist() 
    
    print('dropped with 70% of missings ', len(to_drop), 'columns')
    return df_new, to_drop, cols_to_analyze
x_train, to_drop, cols_to_analyze = null_means(df, 0.6)



def un_unique(df):
    to_drop_nunique = []
    for i in (df.columns):
        if df[i].nunique() == 1:
            to_drop_nunique.append(i)
    df.drop(to_drop_nunique, axis =1) 
    return  df, to_drop_nunique
	
	

def drop_corr_col(df, treshold , persent ):
    # n выбирается на какой выборке строить корреляции в процентах
    a = [random.choice(df.index) for i in xrange(int(len(df)*persent))]
    tmp = df.iloc[a]
    tmp.fillna(0, inplace=True)
    to_drop_after_corr = pd.DataFrame(tmp.corr())
    upper = tmp.where(np.triu(np.ones(tmp.shape), k =1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column]> treshold)]
    df.drop(to_drop, axis =1)
    return df, to_drop

	
# Избавиться от выбрасов, а если они есть, но обучаться на них 
def outlyers(df, min, max):
	for i in df.columns:
		df = df[df[i].between(df[i].quantile(min), df[i].quantile(max))]
	return df

	
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


# Отбор фичей, которые объясняют всю дисперсию данных с помощью принципа главных компонент.
from sklearn import decomposition

def pca_decomposition(df):
    from sklearn import decomposition
    std = StandardScaler()
    #y = df[target]
    #c = df.drop([target], axis =1).fillna(0)
    X = std.fit_transform(df)

    pca = decomposition.PCA().fit(X)

    plt.figure(figsize=(10,7))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
    plt.xlabel('Number of components')
    plt.ylabel('Total explained variance')
    #plt.xlim(0,200)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.axvline(20, c='b')
    plt.axhline(0.97, c='r')
    plt.show()
pca_decomposition(x_train)


# Выбрать K лучших фичей
def select_kbest_reg(df, target, k_best, classif= True):
    t_target = df[target]
    train = df.drop([target], axis =1)
    std = StandardScaler()
    df2 = std.fit_transform(train)

    if classif:
        feat_selector = SelectKBest(f_classif, k=k_best)  #f_classif f_regression
    else:
        df2 = abs(df2)
        feat_selector = SelectKBest(chi2, k=k_best)
        
    _ = feat_selector.fit(df2, t_target)

    feat_scores = pd.DataFrame()
    feat_scores["F Score"] = feat_selector.scores_
    feat_scores["P Value"] = feat_selector.pvalues_
    feat_scores["Support"] = feat_selector.get_support()
    feat_scores["features"] = train.columns

    feat_scores.sort_values(by = 'P Value', ascending = True)
    feat_scores = feat_scores[feat_scores['Support'] == True].sort_values(by = 'P Value')
    best_k_features = list(feat_scores.head(k_best)['features'])
      
    selected = df[best_k_features]
    selected['fl_vyd_pil'] = t_target
    return feat_scores , best_k_features, selected
	
	

# Построим random forest для исключения не важных признаков------------------------------------------------
def RF(df):
    y = df['fl_vyd_pil']
    X = df.drop(['fl_vyd_pil'], axis =1)

    # Делим на трейн тест
    train, test, y_train, y_test = train_test_split(X, y, test_size =0.43,  stratify = y)

    # Строим модель
    rfc = RandomForestClassifier( random_state=42)
    rfc.fit(train, y_train)
    
    y_pred_train = rfc.predict_proba(train)
    y_pred_test = rfc.predict_proba(test)

    gini_test =  2 * roc_auc_score(y_test, y_pred_test[:,1]) - 1 
    
    importances = pd.DataFrame(rfc.feature_importances_, X.columns).reset_index()
    importances.columns='feature', 'score'
    zeroes_features = importances[importances['score'] == 0]['feature']
    
    weak_features = importances[importances['score'] <= 0.00001]['feature']
    
    df_new = df.drop(zeroes_features, axis =1 )
    
    return zeroes_features, df_new, gini_test	



from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler

# Нормализация данных
X_norm = MinMaxScaler().fit_transform(X)

# Выбираем 100 индивидуальных лучших признаков
chi_selector = SelectKBest(chi2, k=100)
chi_selector.fit(X_norm, y)
chi_support = chi_selector.get_support()
chi_feature = X.loc[:,chi_support].columns.tolist()
print(str(len(chi_feature)), 'selected features')

# Выбираем фичи с LogisticRegression
embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l1"), '1.25*median')
embeded_lr_selector.fit(X_norm, y)
embeded_lr_support = embeded_lr_selector.get_support()
embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
print(str(len(embeded_lr_feature)), 'selected features')

# Выбираем фичи с RandomForestClassifier
embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), threshold='1.25*median')
embeded_rf_selector.fit(X, y)
embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
print(str(len(embeded_rf_feature)), 'selected features')

# Выбираем фичи с GradientBoostingClassifier
embeded_gbc_selector = SelectFromModel(GradientBoostingClassifier(n_estimators= 255, max_depth=10), threshold='1.25*median') 
embeded_xgb_selector.fit(X, y)
embeded_xgb_support = embeded_xgb_selector.get_support()
embeded_xgb_feature = X.loc[:,embeded_xgb_support].columns.tolist()
print(str(len(embeded_xgb_feature)), 'selected features')


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=100, step=10, verbose=5)
rfe_selector.fit(X_norm, y)
rfe_support = rfe_selector.get_support()
rfe_feature = X.loc[:,rfe_support].columns.tolist()
print(str(len(rfe_feature)), 'selected features')



pd.set_option('display.max_rows', None)
# put all selection together
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embeded_lr_support,
                                    'Random Forest':embeded_rf_support, 'LightGBM':embeded_lgb_support})
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
feature_selection_df.head(100)


def crossvalimportance(df, target, top_best):
    
    from sklearn.model_selection import KFold, StratifiedKFold

    X = df.drop(target,axis=1)#.values
    y = df[target]#.values

    rfc = RandomForestClassifier()
    cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    im = []
    un_im = []
    scores = []

    for train_index, test_index in cv.split(X,y):
        #print("Train Index: ", train_index,)
        #print("Test Index: ", test_index)

        X_train, X_test, y_train, y_test = X[X.index.isin(train_index)], X[X.index.isin(test_index)],  y[y.index.isin(train_index)], y[y.index.isin(test_index)]
        rfc.fit(X_train, y_train)
        scores.append(rfc.score(X_test, y_test))
        importances_rfc = pd.DataFrame(abs(rfc.feature_importances_.T), X_train.columns).reset_index()
        importances_rfc.columns='feature', 'score'
        importances_rfc = importances_rfc.sort_values(by='score', ascending =False)
        im.append(importances_rfc.head(top_best)['feature'].tolist())
        un_im.append(importances_rfc[importances_rfc['score']==0]['feature'].tolist())


    def listmerge3(lstlst):
        all=[]
        for lst in lstlst:
            all.extend(lst)
        return all

    A = list(set( listmerge3(im) ))
    B = list(set( listmerge3(un_im) ))
    print(len(A))
    print(len(B))
    print(scores)
    print(np.std(scores))
    
    return A, B
	
	
	def identify_outlyers(df, min, max):
    feature=[]
    length=[]
    non_stable = pd.DataFrame()
    for i in df.columns:
        feature.append(i)
        length.append(len(df[~df[i].between(df[i].quantile(min), df[i].quantile(max))]))
        df = df[df[i].between(df[i].quantile(min), df[i].quantile(max))]
    non_stable['feature'] = feature
    non_stable['score'] = length
    print(len(df))
    return non_stable, df
	
	
	############################# stepwise selection  on R language
	
	#Function to calculate ROC index
ROC <- function(indicator, probabilities, step)
{

	#Vector of cutoff
	cutoff <- step*c(0:(1/step));

	#The number of cutoffs
	num <- length(cutoff);

	#The result vectors
	y <- matrix(0, num, 1);
	x <- matrix(0, num, 1);

	#Loop where we consider all the cutoffs
	for(i in c(1:num))
	{
		#The value of cutoff
		cut <- cutoff[(num - i + 1)];

		#Forecast of indicator
		forecast <- as.numeric(probabilities >= cut);

		index_1 <- which(indicator == 1);
		index_0 <- which(indicator == 0);

		y[i] <- sum(forecast[index_1])/length(forecast[index_1]);
		x[i] <- sum(forecast[index_0])/length(forecast[index_0]);
	}
	
	#ROC index
	ROC_index <- 0;
	for (i in c(2:num))
	{
		ROC_index <- ROC_index + ( x[i] - x[i - 1] ) * ( y[i] + y[i-1] ) / 2;
	}

	result <- list(xy = cbind(x, y), ROC_index = ROC_index);
	result;
}

#Function for forward stepwise selection
stepwiseBIC <- function(target, Data_matr, log_file)
{
	library(foreach)
	library(doSNOW)

	cl<-makeCluster(16) 
	registerDoSNOW(cl)

	#Write that we have finished
	cat("\n", file = log_file, append=FALSE);
	cat("Model selection has started", file = log_file, append=TRUE, sep="\n");

	#The number of explanetory variables
	num_expl <- length(Data_matr[1, ]);

	#Available variables
	av_variables <- c(1:num_expl);
	
	#Current model
	curr_model <- c();

	#Sample size
	sample_size <- length(Data_matr[, 1]);

	#Initial value of BIC
	BIC <- 100000000000000000000000000.0;

	#Indicators for leaving the loop
	ind_add_break <- 0;
	ind_rem_break <- 0;

	#Variables for logging
	num_regressions <- 0;
	num_steps_add <- 0;
	num_steps_remove <- 0;

	#Loop where we consider all the numbers of variables
	while (length(curr_model) <= 49)
	{
		#The number of parameters not used in the model
		num_left_par <- length(av_variables);

		#Temprorary BIC and number for the best variable among the rest of them
		BIC_add <- 100000000000000000000000000000000.0;
		best_add <- 0;

		#Loop where we consider all the left variables and find the variable that improves the model more than others
		if (num_left_par > 0)
		{
			num_steps_add <- num_steps_add + 1;
		
			BIC_vect <- foreach (j = c(1:num_left_par)) %dopar%
			{			
				#Include jth variable in the list of used variables
				Data_estimation <- Data_matr[, c(curr_model, av_variables[j])];

				#Create formula
				f <- target ~ Data_estimation;
	
				#Estimate the model
				mylogit <- glm(f, family = "binomial");
			
				#Summary of the logistic regression
				s <- summary(mylogit);
			
				#The number of parameters (we add 2 because of intercept and 1 additional parameter)
				nu_pa <- 2 + length(curr_model);

				#Calculate BIC
				BIC_new <- s$aic + nu_pa*(log(sample_size) - 2);

				#Return BIC
				BIC_new;

			}

			#Write that we have run one more regression
			num_regressions <- num_regressions + 1;
			cat("\n", file = log_file, append=TRUE);
			cat(format(Sys.time(), "[%Y-%m-%d %H:%M:%S] "), file = log_file, append=TRUE);
			
			cat("Regressions = ", file = log_file, append=TRUE);
			cat(toString(num_regressions), file = log_file, append=TRUE);
			
			cat("; Add steps = ", file = log_file, append=TRUE);
			cat(toString(num_steps_add), file = log_file, append=TRUE);
			
			cat("; Remove steps = ", file = log_file, append=TRUE);
			cat(toString(num_steps_remove), file = log_file, append=TRUE);
				
			cat("; Number in model = ", file = log_file, append=TRUE);
			cat(toString(length(curr_model)), file = log_file, append=TRUE);
			
			#Getting index of minimal BIC from vector
			BIC_min_ind <- which(BIC_vect[] == min(as.numeric(BIC_vect)))[1];
			
			if (as.numeric(BIC_vect[BIC_min_ind]) < BIC_add)
			{
					BIC_add <- as.numeric(BIC_vect[BIC_min_ind]);
					best_add <- BIC_min_ind;
			}

		}
		else
		{
			break;
		}

		#Check that the best variable we found in the loop improves the model
		if (BIC_add >= BIC)
		{	
			ind_add_break <- 1;
		}

		if (BIC_add < BIC)
		{
			BIC <- BIC_add;
			curr_model <- c(curr_model, av_variables[best_add]);
			av_variables <- av_variables[av_variables != av_variables[best_add]];
			ind_add_break <- 0;
			ind_rem_break <- 0;
		}
		
		#Calculating and printing ROC index in the log file
		Data_estimation <- Data_matr[, c(curr_model)];
		f <- target ~ Data_estimation;
		mylogit <- glm(f, family = "binomial");
		probability <- predict(mylogit, type = "response");
		roc <- round(ROC(target, probability, 0.001)$ROC_index, digits = 4);
		cat("; ROC index = ", file = log_file, append=TRUE);
		cat(toString(roc), file = log_file, append=TRUE);

		#Loop where we search for the variable whose removal improves the model most of all
		BIC_rem <- 10000000000000000000000.0;
		best_rem <- 0;
		num_in_model <- length(curr_model);
		if (num_in_model > 1)
		{
			num_steps_remove <- num_steps_remove + 1;

			BIC_vect <- foreach (k = c(1:num_in_model)) %dopar%
			{
				#Create formula
				f <- target ~ Data_matr[, curr_model[curr_model != curr_model[k]]];

				#Estimate the model
				mylogit <- glm(f, family = "binomial");

				#Summary of the logistic regression
				s <- summary(mylogit);

				#The number of parameters (we excluded 1 variable and + 1 for intercept)
				nu_pa <- length(curr_model);

				#Calculate BIC
				BIC_new <- s$aic + (nu_pa)*(log(sample_size) - 2);

				#Return BIC
				BIC_new;
				
			}

			num_regressions <- num_regressions + 1;
			cat("\n", file = log_file, append=TRUE);
			cat(format(Sys.time(), "[%Y-%m-%d %H:%M:%S] "), file = log_file, append=TRUE);
				
			cat("Regressions = ", file = log_file, append=TRUE);
			cat(toString(num_regressions), file = log_file, append=TRUE);

			cat("; Add steps = ", file = log_file, append=TRUE);
			cat(toString(num_steps_add), file = log_file, append=TRUE);

			cat("; Remove steps = ", file = log_file, append=TRUE);
			cat(toString(num_steps_remove), file = log_file, append=TRUE);

			cat("; Number in model = ", file = log_file, append=TRUE);
			cat(toString(length(curr_model)), file = log_file, append=TRUE);

			#Getting index of minimal BIC from vector
			best_rem <- which(BIC_vect[] == min(as.numeric(BIC_vect)))[1];
			BIC_rem <- min(as.numeric(BIC_vect));

			#Check that the removal of the found variable improves the model
			if (BIC_rem >= BIC)
			{
				ind_rem_break <- 1;
			}

			if (BIC_rem < BIC)
			{
				BIC <- BIC_rem;
				av_variables <- c(av_variables, curr_model[best_rem]);
				curr_model <- curr_model[curr_model != curr_model[best_rem]];
				ind_rem_break <- 0;
				ind_add_break <- 0;
						
				#Calculating and printing ROC index in the log file
				Data_estimation <- Data_matr[, c(curr_model)];
				f <- target ~ Data_estimation;
				mylogit <- glm(f, family = "binomial");
				probability <- predict(mylogit, type = "response");
				roc <- round(ROC(target, probability, 0.001)$ROC_index, digits = 4);
				cat("; ROC index = ", file = log_file, append=TRUE);
				cat(toString(roc), file = log_file, append=TRUE);

			}
		}
		else
		{
			ind_rem_break <- 1;
		}

		#The number of parameters (including constant)
		num_par <- 1 + length(curr_model);

		if (num_par == 1)
		{
			ind_rem_insign <- 1;
		}

		#Check that it is time to leave the loop
		if ((ind_rem_break == 1) & (ind_add_break == 1))
		{
			break;
		}
	}

	#Write that we are finished
	cat("\n", file = log_file, append=TRUE);
	cat("Model selection is finished", file = log_file, append=TRUE, sep="\n");

	#Stopping parallel cluster
	stopCluster(cl);
	
	#Return current model
	curr_model;
}

#Get data
data <-  knime.in;
log_file <- "D:/USER_FILES/KNIME_LOGS/inv_vay2.txt";
target_ind <- strtoi(data$trg_flg);
num_target = which(names(data) == "trg_flg");

data_modeling <- data[-num_target];
dat_for_logit <- data.matrix(data_modeling);
trg_flg <- data[, num_target];

#f <- target_ind ~ dat_for_logit;
#mylogit <- glm(f, family = "binomial");
#summary(mylogit)
library(iterators)
	library(snow)
#Estimate the model
model <- stepwiseBIC(target_ind, dat_for_logit, log_file);
#knime.out  <- cbind(data[, num_target], data_modeling[, model]);
knime.out  <- cbind(trg_flg, data_modeling[, model]);
------------------------------------------------------


----------------------------------------------------------------




## Cross validation

clf = LogisticRegression(penalty="l1" ,C = 0.05 ,n_jobs = 24, random_state=42, max_iter=50)

X = x_train[numerical_final]
y = y_train

cv = StratifiedKFold(y,n_folds= 3)
results = pd.DataFrame(columns=['training_score', 'test_score'])
fprs, tprs, scores = [], [], []


for train_index, test_index in cv:
    clf.fit(X.iloc[train_index],y.iloc[train_index])
    _, _, auc_score_train = compute_roc_auc(train_index)
    fpr, tpr, auc_score = compute_roc_auc(test_index)
    scores.append((auc_score_train, auc_score))
    fprs.append(fpr)
    tprs.append(tpr)

plot_roc_curve(fprs, tprs);
pd.DataFrame(scores, columns=['AUC Train', 'AUC Test'])




def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    
    # Apply average function to all target data
    prior = target.mean()
    
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
	
	
	
start_time = timeit.default_timer()

target = y_train

category_mean =[]

for i in categorial_final:
    trn_ = x_train[i]
    tst_ = x_test[i]
    a,b = target_encode(trn_series=trn_, 
                  tst_series=tst_, 
                  target=target)
    x_train[a.name] = a
    x_test[b.name] = b
    category_mean.append(b.name)

print( "time: " + str(round(timeit.default_timer() - start_time, 2)))
print("Target encoded: " + str(len(category_mean)) + " variables")


## for ROC_AUC

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def plot_roc_curve(fprs, tprs):
    
    # Initialize useful lists + the plot axes.
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(14,10))
    
    # Plot ROC for each K-Fold + compute AUC scores.
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        
    # Plot the luck line.
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)
    
    # Plot the mean ROC.
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    
    # Plot the standard deviation around the mean ROC.
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    
    # Fine tune and show the plot.
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    plt.show()
    return (f, ax)

def compute_roc_auc(index):
    y_predict = clf.predict_proba(X.iloc[index])[:,1]
    fpr, tpr, thresholds = roc_curve(y.iloc[index], y_predict)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score
	
	

def for_analysis(newdf,clf,categorial_final, numerical_final, name):

    print 'Creating model for', name
    
    x_train, x_test, y_train, y_test = train_test_split(newdf.drop(["trg"], axis =1), newdf["trg"],test_size=0.25,
                                                        random_state=42)

    ## Mean target encoding
    target = y_train

    category_mean =[]

    for i in categorial_final:
        trn_ = x_train[i]
        tst_ = x_test[i]
        a,b = target_encode(trn_series=trn_, 
                      tst_series=tst_, 
                      target=target)
        x_train[a.name] = a
        x_test[b.name] = b
        category_mean.append(b.name)
    print("Target encoding: Done")
        
    ## NA filling     
    x_train[numerical_final] = x_train[numerical_final].fillna(x_train[numerical_final].mean())

    x_test[numerical_final] = x_test[numerical_final].fillna(x_train[numerical_final].mean())
    print('NA filling: Done')
    ## for categorial with target encoding - min

    x_train[category_mean] = x_train[category_mean].fillna(x_train[category_mean].min())
    x_test[category_mean] = x_test[category_mean].fillna(x_train[category_mean].min())
    
    X = x_train[category_mean]
    y = y_train
    
    clf.fit(X,y)
    print("Model_category_mean: Done")

    probs = clf.predict_proba(x_test[category_mean])
    preds = probs[:,1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)

    probs_ = clf.predict_proba(x_train[category_mean])
    preds_ = probs_[:,1]
    fpr1, tpr1, threshold1 = roc_curve(y_train, preds_)
    roc_auc_ = auc(fpr1, tpr1)
    print 'TRAIN_category_mean:', roc_auc_," TEST_category_mean:",roc_auc
    
    feat_importances = pd.Series(clf.feature_importances_, index = x_train[category_mean].columns)

    ## first 15 for model
    cat_for_model = [i for i in feat_importances.nlargest(15).index]
    print "Top 15 category vars:"
    print feat_importances.nlargest(15)
    
    X = x_train[numerical_final]
    y = y_train
    
    clf.fit(X,y)
    print("Model_numerical: Done")

    probs = clf.predict_proba(x_test[numerical_final])
    preds = probs[:,1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)

    probs_ = clf.predict_proba(x_train[numerical_final])
    preds_ = probs_[:,1]
    fpr1, tpr1, threshold1 = roc_curve(y_train, preds_)
    roc_auc_ = auc(fpr1, tpr1)
    print 'TRAIN_numerical:', roc_auc_," TEST_numerical:",roc_auc
    
    feat_importances = pd.Series(clf.feature_importances_, index = x_train[numerical_final].columns)

    ## first 30 for model
    num_for_model = [i for i in feat_importances.nlargest(30).index]
    print "Top 30 numerical_final vars:"
    print feat_importances.nlargest(30)
    
    final = cat_for_model+ num_for_model
    
    X = x_train[final]
    y = y_train
    
    clf.fit(X,y)
    print("Model_final: Done")

    probs = clf.predict_proba(x_test[final])
    preds = probs[:,1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)

    probs_ = clf.predict_proba(x_train[final])
    preds_ = probs_[:,1]
    fpr1, tpr1, threshold1 = roc_curve(y_train, preds_)
    roc_auc_ = auc(fpr1, tpr1)
    print 'TRAIN_final:', roc_auc_," TEST_final:",roc_auc
    
    feat_importances = pd.Series(clf.feature_importances_, index = x_train[final].columns)

    ## first 25 for model
    final_all = [i for i in feat_importances.nlargest(25).index]
    print "Top 25 final vars:"
    print feat_importances.nlargest(25)
    
    clf.fit(x_train[final_all],y_train)
    prediction=clf.predict(x_test[final_all],)
    print('The accuracy of the clf is',metrics.accuracy_score(prediction,y_test),)
    return clf.predict_proba(x_test[final_all])[:,1]
	
	
	
	for i,y in zip(prods_top.tsp_name_m.unique(),range(len(prods_top.tsp_name_m.unique()))):
    
    c = pd.DataFrame(prods_top[prods_top['tsp_name_m']==i].target_.value_counts(normalize = True)).T
    for_prods_magaz = for_prods_magaz.append(c)
    for_prods_magaz['magaz'][y] = i
	
	
	
	------------------------------------------------------------------
	
	------------------------------------------------------------------
	
	import pandas as pd
import numpy as np
pd.options.display.max_columns =100
import pyodbc
pd.options.display.float_format = '{:.2f}'.format
import matplotlib.pyplot as plt
from matplotlib.pylab import rc, plot
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix , roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest,  f_regression, f_classif, chi2
from sklearn.utils import resample
import random
import warnings
warnings.filterwarnings("ignore")

              ############################################################################################
              #                                     Отбор фичей                                          #
              ############################################################################################


# Найдем все не уникальные значения в столбцах------------------------------------------------
def un_unique(df):
    to_drop_nunique = []
    for i in (df.columns):
        if df[i].nunique() == 1:
            to_drop_nunique.append(i)
    df.drop(to_drop_nunique, axis =1) 
    return  df, to_drop_nunique


# Удалить дубликаты колонок------------------------------------------------
def del_duplicate_col(df):    
    k=[]
    for i in xrange(len(df.columns)): 
        if '(#1)' in df.columns[i]: 
            k.append(df.columns[i])
        if '_y' in df.columns[i]: 
            k.append(df.columns[i])
        if '.1' in df.columns[i]: 
            k.append(df.columns[i])
        if '0_y' in df.columns[i]: 
            k.append(df.columns[i])
        if '0_x' in df.columns[i]: 
            k.append(df.columns[i])
        if 'Unnamed: 0' in df.columns[i]: 
            k.append(df.columns[i])
        if 'product_name' in df.columns[i]: 
            k.append(df.columns[i])
        if 'MONTH_END' in df.columns[i]: 
            k.append(df.columns[i])
        if 'customer_id' in df.columns[i]: 
            k.append(df.columns[i])
        if 'wave_nm' in df.columns[i]: 
            k.append(df.columns[i])
        if 'CSTM_GENDER_M' in df.columns[i]: 
            k.append(df.columns[i])
    df.drop(k, axis=1, inplace = True)
    return df, k


#возьмем часть данных и построим матрицу корреляций------------------------------------------------
def drop_corr_col(df, treshold , persent ):
    # n выбирается на какой выборке строить корреляции в процентах
    a = [random.choice(df.index) for i in xrange(int(len(df)*persent))]
    tmp = df.iloc[a]
    tmp.fillna(0, inplace=True)
    to_drop_after_corr = pd.DataFrame(tmp.corr())
    upper = tmp.where(np.triu(np.ones(tmp.shape), k =1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column]> treshold)]
    df.drop(to_drop, axis =1)
    return df, to_drop



# Для обучения и валидации сделаем даунсэмплинг классов------------------------------------------------
def dawnsampling(df, target):  

    # пропуски NaN заполним нулями, т.к. это пусте значения
    df.fillna(0, inplace=True)
    
    df_majority = df[df[target] ==0]
    df_minority = df[df[target]==1]

    df_majority_downsampled = resample(df_majority, replace=False, n_samples= len(df_minority)*3 , random_state=42) 
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    #print(df_downsampled.target.value_counts()) 
    return df_downsampled


# посмотреть на фичи, где от 2 до 5 значений. Являются ли они категориальными?------------------------------------------------
def check_for_cut_feat(df, nmin, nmax): 
    a=[]
    b=[]
    for i in df.columns:
        a.append(df[i].nunique())
        b.append(i)
    c = pd.DataFrame(a,b).reset_index()
    c.columns = 'feature', 'num'

    cat_features = list(c[(c['num'] <= nmax) & (c['num'] > nmin)]['feature'])
    print(len(cat_features))
    return cat_features



# Выеделим категориальные переменные и числовые------------------------------------------------
def cut_num_feat(df): 
    cat_feat = list(df.dtypes[df.dtypes == object].index)
    num_feat = [f for f in df if f not in cat_feat]
    print('cat_feat =', len(cat_feat))
    print('num_feat =',len(num_feat))
    return cat_feat, num_feat



# Выбрать K лучших фичей
def select_kbest_reg(df, target, k_best, classif= True):
    t_target = df[target]
    train = df.drop([target], axis =1)
    std = StandardScaler()
    df2 = std.fit_transform(train)

    if classif:
        feat_selector = SelectKBest(f_classif, k=k_best)  #f_classif f_regression
    else:
        df2 = abs(df2)
        feat_selector = SelectKBest(chi2, k=k_best)
        
    _ = feat_selector.fit(df2, t_target)

    feat_scores = pd.DataFrame()
    feat_scores["F Score"] = feat_selector.scores_
    feat_scores["P Value"] = feat_selector.pvalues_
    feat_scores["Support"] = feat_selector.get_support()
    feat_scores["features"] = train.columns

    feat_scores.sort_values(by = 'P Value', ascending = True)
    feat_scores = feat_scores[feat_scores['Support'] == True].sort_values(by = 'P Value')
    best_k_features = list(feat_scores.head(k_best)['features'])
      
    selected = df[best_k_features]
    selected['fl_vyd_pil'] = t_target
    return feat_scores , best_k_features, selected


def get_dummies(df, cat_feat, nunique_cat):
    df[cat_feat] 
    a = []
    for k in df[cat_feat].columns:
        df[k] = df[k].astype('object')
        if df[k].nunique() <= nunique_cat:
            a.append(k)
    
    cut = pd.get_dummies(df[a])
    dummied = pd.concat([df.drop(cat_feat, axis = 1), cut]) #, ignore_index=True)
    return dummied



# Построим random forest для исключения не важных признаков------------------------------------------------
def RF(df):
    y = df['fl_vyd_pil']
    X = df.drop(['fl_vyd_pil'], axis =1)

    # Делим на трейн тест
    train, test, y_train, y_test = train_test_split(X, y, test_size =0.43,  stratify = y)

    # Строим модель
    rfc = RandomForestClassifier( random_state=42)
    rfc.fit(train, y_train)
    
    y_pred_train = rfc.predict_proba(train)
    y_pred_test = rfc.predict_proba(test)

    gini_test =  2 * roc_auc_score(y_test, y_pred_test[:,1]) - 1 
    
    importances = pd.DataFrame(rfc.feature_importances_, X.columns).reset_index()
    importances.columns='feature', 'score'
    zeroes_features = importances[importances['score'] == 0]['feature']
    
    weak_features = importances[importances['score'] <= 0.00001]['feature']
    
    df_new = df.drop(zeroes_features, axis =1 )
    
    return zeroes_features, df_new, gini_test


def GradientBoosting(df):

    y = df['fl_vyd_pil']
    X = df.drop(['fl_vyd_pil'], axis =1)

    train, test, y_train, y_test = train_test_split(X, y, test_size =0.33,  stratify = y)

    gbc = GradientBoostingClassifier()
    gbc.fit(train, y_train)

    y_pred_train = gbc.predict_proba(train)
    y_pred_test = gbc.predict_proba(test)

    metrics = {
        'Gini_test' : 2 * roc_auc_score(y_test, y_pred_test[:,1]) - 1 ,
        'Gini_train': 2 * roc_auc_score(y_train, y_pred_train[:,1]) - 1
              }

    importances_gbc = pd.DataFrame(abs(gbc.feature_importances_.T), X.columns).reset_index()
    importances_gbc.columns='feature', 'score'
    importances_gbc = importances_lr.sort_values(by='score', ascending =False)

    return  train, test, y_train, y_test, y_pred_train, y_pred_test, metrics,  importances_gbc, gbc

def Logistic_Regression(df, penalty , C):
    std = StandardScaler()
    y = df['fl_vyd_pil']
    c = df.drop(['fl_vyd_pil'], axis =1)

    X = std.fit_transform(c)

    train, test, y_train, y_test = train_test_split(X, y, test_size =0.33,  stratify = y)

    lr = LogisticRegression(penalty=penalty, C=C)
    lr.fit(train, y_train)

    y_pred_train = lr.predict_proba(train)
    y_pred_test = lr.predict_proba(test)

    metrics = {
        'Gini_test' : 2 * roc_auc_score(y_test, y_pred_test[:,1]) - 1 ,
        'Gini_train': 2 * roc_auc_score(y_train, y_pred_train[:,1]) - 1
              }

    importances_lr = pd.DataFrame(abs(lr.coef_.T), c.columns).reset_index()
    importances_lr.columns='feature', 'score'
    importances_lr = importances_lr.sort_values(by='score', ascending =False)

    return  train, test, y_train, y_test, y_pred_train, y_pred_test, metrics, \
                                                                      importances_lr, lr, lr.intercept_ , lr.coef_
            
def RandomForest(df):

    y = df['fl_vyd_pil']
    X = df.drop(['fl_vyd_pil'], axis =1)

    train, test, y_train, y_test = train_test_split(X, y, test_size =0.33,  stratify = y)

    rfc = RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=5, max_features=0.7)
    rfc.fit(train, y_train)

    y_pred_train = rfc.predict_proba(train)
    y_pred_test = rfc.predict_proba(test)

    metrics = {
        'Gini_test' : 2 * roc_auc_score(y_test, y_pred_test[:,1]) - 1 ,
        'Gini_train': 2 * roc_auc_score(y_train, y_pred_train[:,1]) - 1
              }

    importances_rfc = pd.DataFrame(abs(rfc.feature_importances_.T), X.columns).reset_index()
    importances_rfc.columns='feature', 'score'
    importances_rfc = importances_rfc.sort_values(by='score', ascending =False)

    return  train, test, y_train, y_test, y_pred_train, y_pred_test, metrics, importances_rfc, rfc

def plot_confusion_matrix(model, df, yy_test, 
                          classes, normalize=False,
                          title=' Confusion   matrix ',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the  confusion   matrix .
    Normalization can be applied by setting `normalize=True`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.pylab import rc, plot
    import itertools

    cm  =  confusion_matrix(yy_test, model.predict(df))    
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized  confusion   matrix ")
    else:
        print(' Confusion   matrix , without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
   
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    font = {'size' : 15}
    plt.rc('font', **font)
    

    
def report(model, X_train, y_train, X_test, y_test):

    report1 =  classification_report(y_train, model.predict(train), target_names=['Non-take', 'take'])
    print('__________________________________________________________________________________________')
    print('                                   Classification Report                                  ')
    print('__________________________________________________________________________________________')
    print('                                          TRAIN                                           ')
    print(report1)

    report2 =  classification_report(y_test, model.predict(test), target_names=['Non-take', 'take'])
    print('__________________________________________________________________________________________')
    print('                                           TEST                                           ')
    print(report2)
    print('__________________________________________________________________________________________')
    print('______________________________________FEATURE IMPORTANCE__________________________________')
    
    
def roc_auc_plot(model, df, yy_test ):# method I: plt
    import matplotlib.pyplot as plt
    
    probs = model.predict_proba(df)
    preds = probs[:,1]
    fpr, tpr, threshold = roc_curve(yy_test, preds)
    roc_auc = auc(fpr, tpr)
    
    plt.title('ROC')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

from collections import OrderedDict
import pandas as pd
import sys
%matplotlib inline
    
def plot_pandas_style(styler):
    from IPython.core.display import HTML
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

def scoring(features,clf,target):
    score = pd.DataFrame(clf.predict_proba(features)[:,1], columns = ['SCORE'])
    score['DECILE'] = pd.qcut(score['SCORE'].rank(method = 'first'),10,labels=range(10,0,-1))
    score['DECILE'] = score['DECILE'].astype(float)
    score['TARGET'] = target
    score['NONTARGET'] = 1 - target
    return(score)

from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt

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
    
    

def plot_graph(fpr, tpr, fpr1, tpr1, 
               model, model1, 
                gini, gini1,  
                df, name):# , DECILE, DIST_TAR, DIST_SCORE, LIFT, LIFT2

    plt.figure(1,figsize=(20, 5))
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1)
 # ROC AUC
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
 
    z = [round(gini,2), round(gini1,2)]
    zx = ['GINI_' + str(model) + '=' , 'GINI_' + str(model1) + '=']
    
    plt.annotate(pd.DataFrame(z,zx, columns = ['GINI']),
                 xy=(0.6,  0.03),
                 xytext=(0.6,  0.03),fontsize = 14,
                bbox=bbox_props)
    
#     plt.annotate('GINI_' + str(model) + '= ' + str(round(gini_lr,2)),
#                  xy=(0.6,  0.03),
#                  xytext=(0.6,  0.03),fontsize = 14,
#                 bbox=bbox_props)
 
#     plt.annotate('GINI_' + str(model1) + '= '+ str(round(gini_clf,2)),
#                  xy=(0.6,  0.12),
#                  xytext=(0.6,  0.12),fontsize = 14,
#                 bbox=bbox_props)
 
 
    ##########-----------------------------##########--------------------##########--------------------##########
 # LIFT
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
                 xy=(12, df[df['DECILE'] == 10].LIFT.item()),
                 xytext=(12, df[df['DECILE'] == 10].LIFT.item() ),fontsize = 14,
                bbox=bbox_props)
 
    plt.annotate(round(df[df['DECILE'] == 20].LIFT.item(),1),
                 xy=(12, df[df['DECILE'] == 20].LIFT.item()),
                 xytext=(22, df[df['DECILE'] == 20].LIFT.item() ),fontsize = 14,
                bbox=bbox_props)
 
    plt.annotate(round(df[df['DECILE'] == 30].LIFT.item(),1),
                 xy=(12,  df[df['DECILE'] == 30].LIFT.item()),
                 xytext=(32, df[df['DECILE'] == 30].LIFT.item() ),fontsize = 14,
                bbox=bbox_props)
 
    plt.annotate(round(df[df['DECILE'] == 10].LIFT2.item(),1),
                 xy=(12, df[df['DECILE'] == 10].LIFT2.item()),
                 xytext=(12, df[df['DECILE'] == 10].LIFT2.item() ),fontsize = 14,
                bbox=bbox_props)
 
    plt.annotate(round(df[df['DECILE'] == 20].LIFT2.item(),1),
                 xy=(12, df[df['DECILE'] == 20].LIFT2.item()),
                 xytext=(22, df[df['DECILE'] == 20].LIFT2.item() ),fontsize = 14,
                bbox=bbox_props)
 
    plt.annotate(round(df[df['DECILE'] == 30].LIFT2.item(),1),
                 xy=(12, df[df['DECILE'] == 30].LIFT2.item()),
                 xytext=(32, df[df['DECILE'] == 30].LIFT2.item() ),fontsize = 14,
                bbox=bbox_props)
 
    # lift top10 top20 top30
#     plt.annotate('Lift Top10 =' + str(round(df[df['DECILE'] == 10].LIFT2.item() /  df[df['DECILE'] == 10]\
#                                                                                                      .LIFT.item(),2)),
#                  xy=(75, 1.5),
#                  xytext=(75, 1.5),fontsize = 14,
#                 bbox=bbox_props)
    
    top10 = round(df[df['DECILE'] == 10].LIFT2.item() /  df[df['DECILE'] == 10].LIFT.item(),2)
    top20 = round(df[df['DECILE'] == 20].LIFT2.item() /  df[df['DECILE'] == 20].LIFT.item(),2)
    top30 = round(df[df['DECILE'] == 30].LIFT2.item() /  df[df['DECILE'] == 30].LIFT.item(),2)
    
    a = ['LIFT_10=', 'LIFT_20=', 'LIFT_30=']
    b = [top10, top20, top30]
    
    plt.annotate( pd.DataFrame(b,a, columns=['m2m'], )
                 ,
                 xy=(75, 1),
                 xytext=(75, 1),fontsize = 14,
                bbox=bbox_props)
 
     #plt.annotate('Lift Top30 =' + str(round(df[df['DECILE'] == 30].LIFT2.item() /  df[df['DECILE'] == 30]\
#                                                                                                      .LIFT.item(),2)),
#                  xy=(75, 1),
#                  xytext=(75, 1),fontsize = 14,
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
    plt.title('Gains', fontsize=20)
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
                xytext=(30, df[df['DECILE'] == 30].DIST_TAR.item() - 10),
                fontsize = 14,
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
    plt.savefig( str(name) + '.png')
    
#import matplotlib.pyplot as plt; plt.rcdefaults()
# data to plot
def lift_bar(df, LIFT1, LIFT2):
    plt.figure(1,figsize=(1, 1))
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1)
    n_groups = 10
    lift1 = agg[LIFT1]
    lift2 = agg[LIFT2]

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.4
    #plt.grid(True)

    rects1 = plt.bar(index, lift1, bar_width,
    alpha=opacity,
    color='green',
    label='lift1')

    rects2 = plt.bar(index +0.35 , lift2, width= bar_width , #+0.2 ,
    alpha=opacity,
    color='red',
    label='lift2')

    #plt.text(0, 4, r'$\delta=100\ $')

    plt.xlabel('--')
    plt.ylabel('--')
    plt.title('Lift')
    plt.xticks(index + bar_width, ('A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'D1', 'D2', 'E1', 'E2'))
    plt.legend()

    #plt.tight_layout()
    plt.show()


# def roc_auc_plot(model, df, yy_test ):# method I: plt
#     import matplotlib.pyplot as plt
    
#     probs = model.predict_proba(df)
#     preds = probs[:,1]
#     fpr, tpr, threshold = roc_curve(yy_test, preds)
#     roc_auc = auc(fpr, tpr)
    
#     plt.title('ROC')
#     plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#     plt.legend(loc = 'lower right')
#     plt.plot([0, 1], [0, 1],'r--')
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     plt.ylabel('True Positive Rate')
#     plt.xlabel('False Positive Rate')
#     plt.show()
    
    

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
    #agg1 = decile_labels(agg1,'DECILE',color='skyblue')
    
    return agg1

def gains(data,decile_by,target,score):
    inputs = list(decile_by)
    inputs.extend((target,score))
    decile = data[inputs]
    grouped = decile.groupby(decile_by)
    agg1 = pd.DataFrame({},index=[])
    agg1['ACTUAL'] = grouped.mean()[target]*100
    agg1['PRED'] = grouped.mean()[score]*100
    agg1['1'] = grouped.sum()[target].cumsum()
    agg1['2'] = grouped.sum()[target].sum()
    agg1['DIST_TAR'] = grouped.sum()[target].cumsum()/grouped.sum()[target].sum()*100
    agg1.index.name = 'DECILE'
    agg1 = agg1.reset_index()
    agg1['DECILE'] = agg1['DECILE']*10
    agg1['LIFT'] = agg1['DIST_TAR']/agg1['DECILE']
    return agg1

def scoring(features,model,target):
    score = pd.DataFrame(model.predict_proba(features)[:,1], columns = ['SCORE'])
    score['DECILE'] = pd.qcut(score['SCORE'].rank(method = 'first'),10,labels=range(10,0,-1))
    score['DECILE'] = score['DECILE'].astype(float)
    score['TARGET'] = target
    score['NONTARGET'] = 1 - target
    return(score)

-----------------------

---------------------------
def plot_confusion_matrix(model, df, yy_test, 
                          classes, normalize=False,
                          title=' Confusion   matrix ',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the  confusion   matrix .
    Normalization can be applied by setting `normalize=True`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.pylab import rc, plot
    import itertools

    cm  =  confusion_matrix(yy_test, model.predict(df))    
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized  confusion   matrix ")
    else:
        print(' Confusion   matrix , without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
   
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    font = {'size' : 15}
    plt.rc('font', **font)
    

    
def report(model, X_train, y_train, X_test, y_test):

    report1 =  classification_report(y_train1, gbc.predict(train1), target_names=['Non-take', 'take'])
    print('__________________________________________________________________________________________')
    print('                                   Classification Report                                  ')
    print('__________________________________________________________________________________________')
    print('                                          TRAIN                                           ')
    print(report1)

    report2 =  classification_report(y_test1, gbc.predict(test1), target_names=['Non-take', 'take'])
    print('__________________________________________________________________________________________')
    print('                                           TEST                                           ')
    print(report2)
    print('__________________________________________________________________________________________')
    
    
def roc_auc_plot(model, df, yy_test ):# method I: plt
    import matplotlib.pyplot as plt
    
    probs = model.predict_proba(df)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(yy_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    
    plt.title('ROC')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
	
	
def plot_lift(model, df, Y):
	zero = pd.DataFrame(model.predict_proba(df)[:,-1], Y).reset_index().sort_values(by=0, ascending = False).reset_index().reset_index()

	part1, part2, part3, part4, part5, part6, part7, part8, part9, part10 = np.array_split(list(zero['level_0']), 10)
	a = part1, part2, part3, part4, part5, part6, part7, part8, part9, part10

	b =[]
	for i in a:
		b.append(zero.loc[i]['fl_vyd_pil'].sum() / len(zero.loc[i]['fl_vyd_pil']))
		
	plt.plot(b,  marker='o')
	plt.ylabel('True Positive Rate')
	plt.xlabel('Volume')
	for i, txt in enumerate(b):
		plt.annotate(round(txt,2), (i, txt))
	plt.show()
	
	
	
	
	
	


	
