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
