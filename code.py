import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC,SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score,f1_score
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import numpy as np

import matplotlib.pyplot as plt

import io
data = pd.read_csv('feature_X.csv')
data.dropna(inplace=True)


X=data[[ 'inter-stroke time', 'stroke duration',
       'start $x$', 'start $y$', 'stop $x$', 'stop $y$',
       'direct end-to-end distance', ' mean resultant lenght',
       'up/down/left/right flag', 'direction of end-to-end line',
       '20\%-perc. pairwise velocity', '50\%-perc. pairwise velocity',
       '80\%-perc. pairwise velocity', '20\%-perc. pairwise acc',
       '50\%-perc. pairwise acc', '80\%-perc. pairwise acc',
       'median velocity at last 3 pts',
       'largest deviation from end-to-end line',
       '20\%-perc. dev. from end-to-end line',
       '50\%-perc. dev. from end-to-end line',
       '80\%-perc. dev. from end-to-end line', 'average direction',
       'length of trajectory',
       'ratio end-to-end dist and length of trajectory', 'average velocity',
       'median acceleration at first 5 points', 'mid-stroke pressure',
       'mid-stroke area covered', 'mid-stroke finger orientation',
       'change of finger orientation', 'phone orientation']]

y=data[['user id']]

print(X.shape)


#translating each feature within a give range- preprocessing

scalar_1= MinMaxScaler()
scalar_1.fit(X= X)

X=scalar_1.fit_transform(X)

X = pd.DataFrame(X)



def model(X,y, folds=5, is_stratified=True, is_smote=True):
	

	#predictions = np.zeros(test.shape[0])
	clf = RandomForestClassifier(
	n_estimators=100,
	# criterion='gini',
	max_depth=5,
	# min_samples_split=2,
	# min_samples_leaf=50,
	# min_weight_fraction_leaf=0.0,
	# max_features='auto',
	# max_leaf_nodes=None,
	# min_impurity_decrease=0.0,
	# min_impurity_split=None,
	# bootstrap=True,
	# oob_score=False,
	n_jobs=-1,
	random_state=0,
	# verbose=0,
	# warm_start=False,
	class_weight='balanced'
	)

	if is_stratified:
		folds = StratifiedKFold(n_splits=folds, random_state=1283, shuffle=True)
	else:
		folds = KFold(n_splits=folds, random_state=1283, shuffle=True)


	fprs, tprs, scores = [],[],[]
	auc_score = []
	avg_recall_neg = []
	avg_precision_neg = []
	avg_recall_pos = []
	avg_precision_pos = []
	neg_pred_value = []

	feat_importance_df = pd.DataFrame()
	
for n_fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    print(n_fold, train_idx,val_idx)
	clf.fit(X[train_idx], y[train_idx])
	predict_val = clf.predict_proba(X[val_idx])[:,1]
	# print(predict_val)
	fpr, tpr, thresholds = roc_curve(y[val_idx], predict_val)
	auc_ = roc_auc_score(y[val_idx],predict_val)
	auc_score.append(auc_)
	cm = confusion_matrix(y[val_idx],clf.predict(X[val_idx]))
	TN, FP, FN, TP  = cm.ravel()

		precision_neg = TN/(TN+FN)
		recall_neg = TN/(TN+FP)
		precision_pos = TP/(TP+FP)
		recall_pos = TP/(TP+FN)

		avg_recall_neg.append(recall_neg)
		avg_precision_neg.append(precision_neg)
		avg_recall_pos.append(recall_pos)
		avg_precision_pos.append(precision_pos)
		
		# print(f'Precision = {precision}')
		# print(f'Recall = {recall}')
		print(cm)
		# print(classification_report(y[val_idx],clf.predict(X[val_idx])))
		print(f'AUC score for fold {n_fold+1} = {auc_}')
		print('--------------------------------------------------------------------')

		temp = pd.Series(clf.feature_importances_, index=features).reset_index().rename(columns={'index':'feature',0:'importance'}).sort_values(by='importance',ascending=False)
		feat_importance_df = feat_importance_df.append(temp,ignore_index=True)

		# predictions += clf.predict(test[features]) / folds.getn_splits
	
	predictions = clf.predict(test[features])
	predictions_df = test[['customer']].copy()
	predictions_df['is_churn'] = predictions
	predictions_df.to_csv('predictions.csv', index=False)

	print(predictions_df['is_churn'].value_counts())
	print(predictions_df['is_churn'].value_counts(normalize=True))
	display_importance(feat_importance_df)	

	# print(f'\nMean Precision for positive classes = {np.mean(avg_precision_pos)}')
	# print(f'Mean Recall for positive classes = {np.mean(avg_recall_pos)}')
	# print(f'Mean Precision for negative classes = {np.mean(avg_precision_neg)}')
	# print(f'Mean Recall for negative classes = {np.mean(avg_recall_neg)}')
	print(f'\nMean AUC score = {np.mean(auc_score)}')


#for i in range(20887):
  #for j in list(set(train_class)):
train_set,test_set,train_class,test_class=train_test_split(X,y,test_size=0.2, random_state=42)
    #train_set= train_set[i:i+2]
    #train_class=train_class[i:i+2]
    #test_set=test_set[i:i+2]
   # test_class=test_class[i:i+2]
for j in list(set(train_class)):
    new_train_class=np.array([1 if k==j else 0 for k in train_class ]).reshape(-1,1)
     
    new_test_class=np.array([1 if k==j else 0 for k in test_class]).reshape(-1,1)

from scipy.optimize import brentq
from scipy.interpolate import interp1d
rbf_svc_1=SVC(kernel='rbf',  gamma=1,random_state=10,C=10,tol=0.00001,class_weight='balanced').fit(train_set, new_train_class)





y_trainPred= rbf_svc_1.decision_function(test_set)

#y_scores_1 = cross_val_score(rbf_svc_2, X, y_1, cv=5)



#False Positive fpr and True Positve tpr
#fpr, tpr, thresholds= roc_curve(new_test_class , y_trainPred)
#frr=np.ones(len(tpr))-tpr
#far=fpr

#y_trainPred=rbf_svc_1.decision_function(test_set)

fpr, tpr, thresholds= roc_curve(new_test_class , y_trainPred)
print(thresholds)
eer = np.round(brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.),2)
thresh = interp1d(fpr, thresholds)(eer)
#print(thresh)
print("Error Rate", eer)
from sklearn.metrics import roc_auc_score

#roc_auc_score(test_class, y_trainPred)
#frr=1-tpr


#EER = np.round(far[np.nanargmin(np.absolute((far - frr)))],2)
#EER2 = np.round(frr[np.nanargmin(np.absolute((far - frr)))],2)

#print((EER+EER2)/2)
roc_auc=auc(fpr, tpr)
plt.figure()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('SVM Classifier ROC')
plt.plot(fpr, tpr, color='blue', lw=2, label='SVM ROC area = %0.2f)' % roc_auc)
plt.legend(loc="lower right")
plt.show()

#print(y_scores_1.mean())


