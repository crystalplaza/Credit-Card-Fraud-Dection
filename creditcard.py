#!/usr/bin/env python
# coding: utf-8

# ## Credit Card Fraud Detection 
# Three predictive models to see how accurate they are in detecting whether a transaction is legit or fraud.
# - Support vector machine
# - Random Forest
# - Logistic Regression
# Two Approaches are applied to handle the inbalanced dataset
# - Undersampling
# - Oversampling, to be specific, SMOTE

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, accuracy_score


# In[3]:


creditCard = pd.read_csv('gs://project271/creditcard.csv', sep=",")


# In[4]:


creditCard.shape


# In[5]:


creditCard['Amount'].mean()


# In[6]:


creditCard.head()


# In[7]:


creditCard[creditCard.Class == 1].shape[0]/creditCard.shape[0]


# In[8]:


creditCard[creditCard.Class == 1].shape


# In[9]:


print(creditCard["Class"].value_counts())
#plt.style.use('dark_background')
sns.countplot("Class",data=creditCard,color="lightsteelblue")
plt.xlabel("Class", fontSize = 12, fontWeight = 'bold')
plt.ylabel("Count", fontSize = 12, fontWeight = 'bold')
plt.title("Countplot for Non-fraud and Fraud \n 0:normal, 1: fraud", fontSize = 15, fontWeight = 'bold')
plt.show()


# In[10]:


## fraud data visualization
creditCard_fraud = creditCard[creditCard.Class == 1]
plt.figure(figsize=(6,6))
plt.scatter(creditCard_fraud['Time'], creditCard_fraud['Amount'],color = "lightsteelblue") # Display fraud amounts according to their time
plt.title('Scratter plot amount fraud', fontSize = 18, fontWeight = "bold")
plt.xlabel('Time',fontSize = 15, fontWeight = "bold")
plt.ylabel('Amount',fontSize = 15, fontWeight = "bold")
plt.xlim([0,175000])
plt.ylim([0,2500])
plt.show()


# In[11]:


plt.boxplot(creditCard['Amount'])


# In[12]:


plt.hist(x = creditCard['Amount'],color = "lightsteelblue")


# In[13]:


#### check if remove the extreme value in outlier will improve the accuracy
creditCard[creditCard['Amount']> 10000]


# Note that there are several extreme data existing in the dataset, when choosing scaler method, we will choose robust scaler to transform the amount and time

# In[14]:


scaler = RobustScaler()
creditCard["scaledAmount"] = scaler.fit_transform(creditCard['Amount'].values.reshape(-1,1))
creditCard["scaledTime"] = scaler.fit_transform(creditCard['Time'].values.reshape(-1,1))  


# In[15]:


creditCard["scaledAmount"].head()


# In[16]:


creditCard["scaledAmount"].max()


# In[17]:


plt.boxplot(creditCard["scaledAmount"])


# In[18]:


creditCard[creditCard["scaledAmount"]>150]


# robustscaler is robust to outliers, after scaling the data, the extrem data in the original dataset still exist in the transformed data.

# In[19]:


creditCard_corr = creditCard.corr()
## heatmap
plt.figure(figsize=(15,10))
sns.heatmap(creditCard_corr, cmap="YlGnBu") # Displaying the Heatmap
sns.set(font_scale=2,style='white')

plt.title('Heatmap correlation', fontWeight = "bold")
plt.show()


# In[20]:


## check null value
creditCard.isnull().values.any()


# In[21]:


## drop original amount and time variables
creditCard = creditCard.drop(["Amount", "Time"], axis = 1)


# It is suggested on kaggle website, it is not wise to test using samling data. We should split data before sampling data.
# 

# In[22]:


creditCard.columns


# In[23]:


## splitting data before sampling
x = creditCard.drop("Class", axis = 1)
y = creditCard["Class"]

skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
for trainIndex, testIndex in skf.split(x, y):
    print("TRAIN:", trainIndex, "TEST:", testIndex)
    xTrainBefore, xTestBefore = x.iloc[trainIndex], x.iloc[testIndex]
    yTrainBefore, yTestBefore = y.iloc[trainIndex], y.iloc[testIndex]


# We want the same proportion of fraud transation and normal transaction in test set and train set. 

# In[24]:


yTestBefore.shape


# In[25]:


xTrainBeforeVal = xTrainBefore.values
xTestBeforeVal = xTestBefore.values
yTrainBeforeVal = yTrainBefore.values
yTestBeforeVal = yTestBefore.values

trainLabel, trainCounts = np.unique(yTrainBeforeVal, return_counts=True)
testLabel, testCounts = np.unique(yTestBeforeVal, return_counts=True)

print('Label Distributions: \n')
print(trainCounts/ len(yTrainBeforeVal))
print(testCounts/ len(yTestBeforeVal))

trainPortion = trainCounts/ len(yTrainBeforeVal)
print(trainPortion)
testPortion = testCounts/ len(yTestBeforeVal)
X = np.arange(2)
##  show in barchart
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, trainPortion, color = 'lightsteelblue', width = 0.25)
ax.bar(X + 0.25, testPortion, color = 'orange', width = 0.25)
ax.legend(labels=['trainPercentage', 'testPercentage'])
ax.set_ylabel("Percenatage")
ax.set_title("non-fraud percentage")


# Fraud transcation and non fraud transaction are almost the same proportion in training dataset and testing dataset.

# In[26]:


## select fraud credit card
fraud = creditCard[creditCard['Class'] == 1]
fraud.shape
#creditCard = creditCard.sample(frac = 1)


# In[27]:


##shuffle the data and pick the first nonfraud
creditCard = creditCard.sample(frac = 1)
normal = creditCard[creditCard["Class"]== 0]
normal.shape


# In[28]:


normal = normal[:492]
normal.shape


# In[29]:


creditCardUndersample = pd.concat([normal, fraud])
creditCardUndersample.shape


# In[30]:


creditCardUndersample.head()


# Due to concatnation, the first half is fraud, and the next half is normal data, so it's better to shuffle the data.

# In[31]:


creditCardUndersample = creditCardUndersample.sample(frac = 1)


# In[32]:


creditCardUndersample["Class"].head()


# OK, now it is mixed fraud and non-fradu together

# In[33]:


## plot the propotion of data in the bar plot
size = creditCardUndersample.groupby('Class')["Class"].size()
plt.figure(figsize=(6,6))
plt.bar(list(creditCardUndersample["Class"].unique()),creditCardUndersample.groupby('Class')["Class"].size(),color = "lightsteelblue")
plt.text(0, size[0]+2,str(size[0]), color='blue', fontweight='bold',fontSize = 12 )
plt.text(1.0, size[1]+2,str(size[1]), color='blue', fontweight='bold', fontSize = 12)
plt.ylabel("Count", fontsize = 15,fontweight = 'bold')
plt.title("Fraud Counts vs non-Fraud Counts", fontsize = 15, fontweight = 'bold')


# In[34]:


## let's see how the new data look like
## scatter plot
#creditCardUndersample.columns

fig, ax = plt.subplots()
color = ["orange", 'lightsteelblue']
#plt.scatter(creditCardUndersample['scaledTime'],creditCardUndersample['scaledAmount'], c = creditCardUndersample['Class'])
plt.scatter(fraud['scaledTime'] , fraud['scaledAmount'],c= color[0])
plt.scatter(normal['scaledTime'], normal['scaledAmount'],c = color[1])
ax.legend(labels = ["fraud", "non-fraud"], fontsize = 10)
ax.set_title("Scatter Plot After Undersampling", fontsize = 12,fontweight = 'bold')
ax.set_xlabel("Scaled Time", fontsize = 12,fontweight = 'bold')
ax.set_ylabel("Scaled Amount", fontsize = 12,fontweight = 'bold')


# In[35]:


data = [fraud['scaledAmount'],normal['scaledAmount']]
fig, ax = plt.subplots()
ax.boxplot(data)
ax.set_title("Fraud and non-Fraud boxplot")


# In[36]:


## heatmap
corrUnder = creditCardUndersample.corr()
#plt.figure(figsize=(15,10))
#ax = plt.axes(figsize=(15,10))
#sns.heatmap(corrUnder)
#sns.set(font_scale=0.1,style='white')
#ax.set_title("Heat Map", fontsize = 20, fontweight = 'bold')

## heatmap
plt.figure(figsize=(18,22))
sns.heatmap(corrUnder, cmap="YlGnBu",annot=True,fmt=".1f",annot_kws={'size':8},) # Displaying the Heatmap
sns.set(font_scale=2,style='white')

plt.title('Heatmap Correlation Undersampling ',fontWeight = "bold")
plt.show()


# After checking heatmap,delete V1, V3, V10, V11, V14, V16 and v18

# In[37]:


col_dict = {'V1':1, 'V2':2, 'V3':3,'V4':4, 'V5':5,'V6':6, 'V7':7, 'V8':8, 'V9':9, 'V10':10, 'V11':11, 'V12':12, 
            'V13':13, 'V14':14, 'V15':15, 'V16':16, 'V17': 17, 'V18':18, 'V19':19, 'V20':20, 'V21':21,'V22':22,
           'V23':23, 'V24':24, 'V25':25, 'V26':26,'V27':27, 'V28':28,'scaledAmount':29}
plt.figure(figsize = (20,30))
for variable, i in col_dict.items():
    plt.subplot(6,5,i)
    plt.boxplot(creditCardUndersample[variable])
plt.show()


# In[38]:


## split the data,test case 80%
creditCardUndersampleX = creditCardUndersample.drop("Class",axis = 1)
creditCardUndersampleY = creditCardUndersample['Class']
xTrain,xTest,yTrain, yTest = train_test_split(creditCardUndersampleX, creditCardUndersampleY, test_size=0.2, random_state=0)  


# In[39]:


## support vector machine
clf = svm.SVC(kernel='linear', C=1).fit(xTrain, yTrain)
trainingScore = cross_val_score(clf, xTrain, yTrain)  ## default is 5 fold
trainingScore


# In[40]:


#trainingScore.mean()  95% confidence interval
print("Accuracy: %0.2f (+/- %0.2f)" % (trainingScore.mean(), trainingScore.std() * 2))


# In[41]:


# support vector machine different C = 0.001
clf1 = svm.SVC(kernel='linear', C = 0.001).fit(xTrain, yTrain)
trainingScore1 = cross_val_score(clf1, xTrain, yTrain)  ## default is 5 fold
trainingScore1


# In[42]:


#trainingScore.mean()  95% confidence interval
print("Accuracy: %0.2f (+/- %0.2f)" % (trainingScore1.mean(), trainingScore1.std() * 2))


# In[43]:


# support vector machine different C = 0.01
clf2 = svm.SVC(kernel='linear', C = 0.01).fit(xTrain, yTrain)
trainingScore2 = cross_val_score(clf2, xTrain, yTrain)  ## default is 5 fold
trainingScore2


# In[44]:


#trainingScore.mean()  95% confidence interval
print("Accuracy: %0.2f (+/- %0.2f)" % (trainingScore2.mean(), trainingScore2.std() * 2))


# In[45]:


## try bigger C
clf3 = svm.SVC(kernel='linear', C = 100).fit(xTrain, yTrain)
trainingScore3 = cross_val_score(clf3, xTrain, yTrain)  ## default is 5 fold
trainingScore3


# In[46]:


#trainingScore.mean()  95% confidence interval
print("Accuracy: %0.2f (+/- %0.2f)" % (trainingScore3.mean(), trainingScore3.std() * 2))


# In[47]:


## try bigger C = 10
clf4 = svm.SVC(kernel='linear', C = 10).fit(xTrain, yTrain)
trainingScore4 = cross_val_score(clf4, xTrain, yTrain)  ## default is 5 fold
trainingScore4


# In[48]:


#trainingScore.mean()  95% confidence interval
print("Accuracy: %0.2f (+/- %0.2f)" % (trainingScore4.mean(), trainingScore4.std() * 2))


# In[49]:


## try bigger C = 5
clf5= svm.SVC(kernel='linear', C = 5).fit(xTrain, yTrain)
trainingScore5 = cross_val_score(clf5, xTrain, yTrain)  ## default is 5 fold
trainingScore5


# In[50]:


#trainingScore.mean()  95% confidence interval
print("Accuracy: %0.2f (+/- %0.2f)" % (trainingScore5.mean(), trainingScore5.std() * 2))


# In[51]:


## try bigger C = 2
clf6= svm.SVC(kernel='linear', C = 2).fit(xTrain, yTrain)
trainingScore6 = cross_val_score(clf6, xTrain, yTrain)  ## default is 5 fold
trainingScore6


# In[52]:


#trainingScore.mean()  95% confidence interval
print("Accuracy: %0.2f (+/- %0.2f)" % (trainingScore6.mean(), trainingScore6.std() * 2))


# With linear kernel, the best C would be C= 2 with bigest mean, and smallest variance

# In[53]:


## try RBF kernel
clf7= svm.SVC(kernel='rbf', C = 1).fit(xTrain, yTrain)
trainingScore7 = cross_val_score(clf7, xTrain, yTrain)  ## default is 5 fold
trainingScore7


# In[54]:


#trainingScore.mean()  95% confidence interval
print("Accuracy: %0.2f (+/- %0.2f)" % (trainingScore7.mean(), trainingScore7.std() * 2))


# In[55]:


## try RBF kernel, with smaller c = 0.001
clf8= svm.SVC(kernel='rbf', C = 0.001).fit(xTrain, yTrain)
trainingScore8 = cross_val_score(clf8, xTrain, yTrain)  ## default is 5 fold
trainingScore8


# Appearntly, with smaller C, the misclassification becomes bigger.
# 

# In[56]:


## try RBF kernel, with bigger c = 100
clf9= svm.SVC(kernel='rbf', C = 100).fit(xTrain, yTrain)
trainingScore9 = cross_val_score(clf9, xTrain, yTrain)  ## default is 5 fold
trainingScore9


# In[57]:


#trainingScore.mean()  95% confidence interval
print("Accuracy: %0.2f (+/- %0.2f)" % (trainingScore9.mean(), trainingScore9.std() * 2))


# In[58]:


## try RBF kernel, with bigger c = 10000
clf10= svm.SVC(kernel='rbf', C = 10000).fit(xTrain, yTrain)
trainingScore10 = cross_val_score(clf10, xTrain, yTrain)  ## default is 5 fold
trainingScore10


# with bigger c = 10000, the accuracy becomes lower
# 

# In[59]:


## try RBF kernel, with bigger c = 10
clf11= svm.SVC(kernel='rbf', C = 10).fit(xTrain, yTrain)
trainingScore11 = cross_val_score(clf11, xTrain, yTrain)  ## default is 5 fold
trainingScore11


# In[60]:


#trainingScore.mean()  95% confidence interval
print("Accuracy: %0.2f (+/- %0.2f)" % (trainingScore11.mean(), trainingScore11.std() * 2))


# In[61]:


## grid search
parameters = {'kernel':('linear','rbf'), 'C':[1, 10, 100]}
svc = svm.SVC()
clf10 = GridSearchCV(svc, parameters)
clf10.fit(xTrain, yTrain)
GridSearchCV(svc,
             param_grid={'C': [0.001,0.1,1, 10, 100, 1000], 'kernel': ('linear','rbf')},cv=5,scoring='accuracy')
sorted(clf10.cv_results_.keys())


# In[62]:


print(clf10.best_params_)
print()
print('Training accuracy')
print(clf10.best_score_)
print(clf10.best_estimator_)


# In[63]:


svcPred = cross_val_predict(svc, xTrain, yTrain, cv=5,
                             method="decision_function")
print(roc_auc_score(yTrain, svcPred))


# In[64]:


### ROC curve
svc_fpr, svc_tpr, svc_threshold = roc_curve(yTrain, svcPred)
plt.figure(figsize = (6,6))
#plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(yTrain, svcPred)))
plt.plot(svc_fpr, svc_tpr, label='SVM AUC: {:.3f}'.format(roc_auc_score(yTrain, svcPred)))
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.title("svm roc curve")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()


# In[65]:


predictTest = clf10.predict(xTest)


# In[66]:


precision, recall, thresholds = precision_recall_curve(yTest, predictTest )

# Plot ROC curve
plt.plot(precision, recall)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve for SVM')
plt.show()


# In[67]:


## predict with orignal data
predictTest = clf10.predict(xTestBefore)


# In[68]:


#print("accuracy: {}".format(round(accuracy_score(yTestBefore, predictTest),4)))
print("precision: {}".format(round(precision_score(yTestBefore, predictTest),4)))
print("recall: {}".format(round(recall_score(yTestBefore, predictTest),4)))


# In[69]:


from sklearn.metrics import average_precision_score
y_score = clf10.decision_function(xTestBefore)
average_precision = average_precision_score(yTestBefore, y_score)
print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))


# In[70]:


## eleminate some features according to heatmap
creditCardUndersampleXDrop = creditCardUndersample.drop(["Class",'V1','V3','V10','V11','V14','V16', 'V18'],axis = 1)
creditCardUndersampleY = creditCardUndersample['Class']
xTrainDrop,xTestDrop,yTrainDrop, yTestDrop = train_test_split(creditCardUndersampleXDrop, creditCardUndersampleY, test_size=0.2, random_state=0)


# In[71]:


## support vector machine
clfDrop1 = svm.SVC(kernel='linear', C=10).fit(xTrainDrop, yTrainDrop)
trainingScoreDrop1 = cross_val_score(clfDrop1, xTrainDrop, yTrainDrop)  ## default is 5 fold
trainingScoreDrop1


# In[72]:


## grid search
parameters = {'kernel':('linear','rbf'), 'C':[0.001,0.1,1, 10, 100]}
svc = svm.SVC()
clfGridSearch = GridSearchCV(svc, parameters)
clfGridSearch.fit(xTrainDrop, yTrainDrop)
GridSearchCV(svc,
             param_grid={'C': [0.001,0.1,1, 10, 100, 1000], 'kernel': ('linear','rbf')},cv=5,scoring='accuracy')
sorted(clfGridSearch.cv_results_.keys())


# In[73]:


print(clfGridSearch.best_params_)
print()
print('Training accuracy')
print(clfGridSearch.best_score_)
print(clfGridSearch.best_estimator_)


# In[74]:


## using RFE to do feature selection,automatic select features
svc = svm.SVC(kernel = 'linear')
rfeSVM = RFECV(estimator = svc)
rfeSVM = rfeSVM.fit(xTrainDrop, yTrainDrop)
pipelineSVM = Pipeline(steps=[('s',rfeSVM),('m',svc)])
# evaluate model
n_scoresSVM= cross_val_score(pipelineSVM, xTrainDrop, yTrainDrop, scoring='accuracy', cv= 5, n_jobs=-1, error_score='raise')
# report performance
print("Accuracy: %0.3f (+/- %0.3f)" % (n_scoresSVM.mean(), n_scoresSVM.std()))


# The accuracy did not improve

# In[75]:


rfeSVM.ranking_


# In[76]:


## using RFE to do feature selection,automatic select features
svc = svm.SVC(kernel = 'linear')
rfeSVM1 = RFECV(estimator = svc)
rfeSVM1 = rfeSVM.fit(xTrain, yTrain)
pipelineSVM1 = Pipeline(steps=[('s',rfeSVM1),('m',svc)])
# evaluate model
n_scoresSVM1 = cross_val_score(pipelineSVM1, xTrain, yTrain, scoring='accuracy', cv= 5, n_jobs=-1, error_score='raise')
# report performance
print("Accuracy: %0.2f (+/- %0.2f)" % (n_scoresSVM1.mean(), n_scoresSVM1.std()))


# In[77]:


rfeSVM1.ranking_


# In[78]:


rfeSVM.n_features_


# By RFE feature selection, 10 features are selected.

# In[79]:


xTrain.columns


# In[80]:


xTraindrop2 = xTrain.drop(['V1','V2','V3','V5', 'V6', 'V7','V8','V9', 'V14','V15', 'V17', 'V18', 'V19', 'V20', 'V21',
       'V22', 'V24', 'V26','V27', 'V28'], axis=1)


# In[ ]:





# In[ ]:





# In[81]:


## random forest
clfRF = RandomForestClassifier()
clfRF.fit(xTrain, yTrain)
trainingScoreRF= cross_val_score(clfRF, xTrain, yTrain,scoring = "accuracy")  ## default is 5 fold
trainingScoreRF


# In[82]:


## random forest
clfRF = RandomForestClassifier()
clfRF.fit(xTrain, yTrain)
trainingScoreRF= cross_val_score(clfRF, xTrain, yTrain,scoring = "accuracy")  ## default is 5 fold
trainingScoreRF


# In[83]:


## prediction score
RFPred = cross_val_predict(clfRF, xTrain, yTrain, cv=5)
print(roc_auc_score(yTrain, RFPred))


# In[84]:


### ROC curve for random forest
rf_fpr, rf_tpr, rf_threshold = roc_curve(yTrain, RFPred)
plt.figure(figsize = (6,6))
plt.plot(rf_fpr, rf_tpr, label='RF AUC: {:.3f}'.format(roc_auc_score(yTrain, RFPred)))
plt.title("Random Forest ROC Curve")
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()


# In[106]:


clfRF.feature_importances_


# In[107]:


creditCardUndersampleX.columns


# In[108]:


plt.figure(figsize = (8,8))
feat_importances = pd.Series(clfRF.feature_importances_, index=xTrain.columns)
feat_importances.nlargest(28).plot(kind='barh',color = "lightsteelblue")
plt.title("Feature Importance of Random Forest",fontsize = 10, fontWeight = 'bold')


# In[109]:


## grid search 
param_grid = {"max_depth": [3,5, None],
              "n_estimators":[3,5,10],
              "max_features": [5,6,7,8]}

# Creating the classifier
model = RandomForestClassifier(max_features=3, max_depth=2 ,n_estimators=10, random_state=3, criterion='entropy', n_jobs=1, verbose=1 )
grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='recall')
grid_search.fit(xTrain, yTrain)


# In[110]:


print(grid_search.best_score_)
print(grid_search.best_params_)


# In[111]:


## fit the model
rfGridSearch = RandomForestClassifier(max_features = 6, n_estimators = 5)
rfGridSearch.fit(xTrain, yTrain)


# In[112]:


print("Training score data: ")
print(rfGridSearch.score(xTrain, yTrain))


# In[113]:


## logistic regression assumption remove highly correlated data
xTrainRemove = xTrain.drop(["V1",'V3','V9','V12','V16'], axis = 1)


# In[114]:


## using RFE to do feature selection, keep 20 features
logreg = LogisticRegression()
rfe = RFE(logreg, n_features_to_select=  20)
rfe = rfe.fit(xTrainRemove, yTrain)
print(rfe.support_)
print(rfe.ranking_)


# In[115]:


pipeline = Pipeline(steps=[('s',rfe),('m',logreg)])
# evaluate model
#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline, xTrain, yTrain, scoring='accuracy', cv= 5, n_jobs=-1, error_score='raise')
# report performance
print("Accuracy: %0.3f (+/- %0.3f)" % (n_scores.mean(), n_scores.std()))


# In[116]:


## using RFE to do feature selection, keep 15 features
logreg = LogisticRegression()
rfe1 = RFE(logreg, n_features_to_select  =15)
rfe1 = rfe1.fit(xTrainRemove, yTrain)
print(rfe1.support_)
print(rfe1.ranking_)


# In[117]:


pipeline1 = Pipeline(steps=[('s',rfe1),('m',logreg)])
# evaluate model
n_scores1 = cross_val_score(pipeline1, xTrain, yTrain, scoring='accuracy', cv= 5, n_jobs=-1, error_score='raise')
# report performance
print("Accuracy: %0.3f (+/- %0.3f)" % (n_scores1.mean(), n_scores1.std()))


# In[118]:


## using RFE to do feature selection,automatic select features
logreg = LogisticRegression()
rfe2 = RFECV(estimator=LogisticRegression())
rfe2 = rfe2.fit(xTrain, yTrain)
pipeline2 = Pipeline(steps=[('s',rfe2),('m',logreg)])
# evaluate model
n_scores2= cross_val_score(pipeline2, xTrain, yTrain, scoring='accuracy', cv= 5, n_jobs=-1, error_score='raise')
# report performance
print("Accuracy: %0.3f (+/- %0.3f)" % (n_scores2.mean(), n_scores2.std()))


# In[119]:


rfe2.ranking_


# In[120]:


xTrainRemove.columns


# In[121]:


features = ['V2', 'V4', 'V5', 'V6','V8', 'V10', 'V11', 'V13', 'V14','V18','V21', 'V22', 'V26','V27','scaledTime']


# In[122]:


xTrainRemoveRanking = xTrainRemove[features]


# In[123]:


LogisticPred = cross_val_predict(rfe2, xTrainRemoveRanking, yTrain, cv=5)
print(roc_auc_score(yTrain, LogisticPred))


# In[ ]:





# In[126]:


from collections import Counter
counter = Counter(ysm_train)
xTrainBefore.shape


# In[124]:


sm = SMOTE(random_state=2)
xsm_train, ysm_train = sm.fit_sample(xTrainBefore, yTrainBefore.ravel())


# In[ ]:


## split the data,test case 80%
#xTrainSm,xTestSm,yTrainSm, yTestSm = train_test_split(xsm_trainDrop, ysm_train, test_size=0.2, random_state=0)  


# In[84]:



from collections import Counter
counter = Counter(ysm_train)
xTrainBefore.shape


# In[85]:


print(counter)


# In[86]:


# bar plot of examples by class label
X = np.arange(1)
fig = plt.figure(figsize =(4,4))
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, counter[0], color = 'lightsteelblue', width = 0.05)
ax.bar(X + 0.25, counter[1], color = 'orange', width = 0.05)
ax.set_title("Counts After SMOTE",fontweight = 'bold',fontSize = 12)
ax.text(0-0.008,counter[0]+1,counter[0],color='blue', fontweight='bold', fontSize = 10)
ax.text(0.25-0.008,counter[1]+1,counter[1],color='blue', fontweight='bold', fontSize = 10)
ax.set_ylabel("Counts",fontweight = 'bold',fontSize = 12)
ax.legend(labels = ["fraud", "non-fraud"], fontsize = 10)


# In[87]:


corrOver = xsm_train.corr()

## heatmap
plt.figure(figsize=(18,22))
sns.heatmap(corrOver, cmap="YlGnBu",annot=True,fmt=".1f",annot_kws={'size':8},) # Displaying the Heatmap
sns.set(font_scale=2,style='white')

plt.title('Heatmap correlation',fontWeight = "bold")
plt.show()


# - v16,17, and v18 are highly correlated,so we will delete v16,and v18 by keeping v17.
# - V1 is highly correlated with v3, v5, v7
# - v2 is highly correlated with v3 and v7
# - v3 is highly correlated with v1, v2, v5,v7, v10
# - v7 and v10 are highly correlated
# - v11 and v12 are highly correlated
# - so delete v3, v7, v11, v16, v18(V1, V3, V10,V11,V14,V16,V18)

# In[88]:


creditCard.columns


# In[89]:


xsm_trainDrop = xsm_train.drop(['V3','V7','V11','V16','V18'], axis = 1)


# In[90]:


xsm_trainDrop.shape


# In[91]:


xsm_trainDrop.columns


# In[92]:


## do feature selection


# In[93]:


## split the data,test case 80%
xTrainSm,xTestSm,yTrainSm, yTestSm = train_test_split(xsm_trainDrop, ysm_train, test_size=0.2, random_state=0)  


# In[94]:


##clfSm = svm.SVC(kernel='linear', C=1).fit(xTrainSm, yTrainSm)


# In[95]:


parameters = {
    'C': [ 0.1, 1, 10, 100]
             }
lr = LogisticRegression()
clf = GridSearchCV(lr, parameters, cv=5, verbose=5, n_jobs=3)
clf.fit(xTrainSm, yTrainSm.ravel())


# In[96]:


clf.best_estimator_


# In[97]:


prediction = best_est.predict(xTrainSm)


# In[98]:


lr1 = LogisticRegression(C= 100, verbose=5)
lr1.fit(xTrainSm, yTrainSm)


# In[107]:


y_pred_sample_score = lr1.decision_function(xTestSm)


fpr, tpr, thresholds = roc_curve(yTestSm, y_pred_sample_score)

roc_auc = auc(fpr,tpr)

# Plot ROC
plt.title('ROC')
plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[100]:


roc_auc


# In[108]:


fpr


# In[101]:


predictTest = lr1.predict(xTestSm)
precision, recall, thresholds = precision_recall_curve(yTestSm, predictTest )

# Plot ROC curve
plt.plot(precision, recall)
plt.legend(loc='lower right')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve for logistic regression')
plt.show()

