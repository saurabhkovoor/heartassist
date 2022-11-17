import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import os

# ns
from scipy import stats 
from matplotlib import rcParams
from matplotlib.cm import rainbow
import warnings
warnings.filterwarnings('ignore')
from collections import Counter

# Importing Libraries - Model Preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import KNNImputer

# Importing Libraries - Model Creation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

from kneed import KneeLocator

# Importing Libraries - Model Creation (Neural Networks - keras)
from keras.models import Sequential
from keras.layers import Dense
# from numpy import loadtxt

import pickle
import joblib
import time

# Importing Libraries - Model Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score,confusion_matrix, recall_score, precision_score, classification_report, average_precision_score, mean_squared_error, mean_absolute_error,matthews_corrcoef

def modelValidation(y_test, y_pred, startTrain =0, endTrain=0, startTest=0, endTest=0, classifierName=""):
    print("Results")

    classifReport = classification_report(y_test, y_pred)
    print(classifReport)
    
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy Score: {}".format(acc))

    prec = precision_score(y_test, y_pred, average="micro")
    print("Precision Score: {}".format(prec))

    rec = recall_score(y_test, y_pred, average="micro")
    print("Recall Score: {}".format(rec))

    f1score = f1_score(y_test, y_pred, average="micro")
    print("F1 Score Score: {}".format(f1score))

    mse = mean_squared_error(y_test, y_pred)
    print("Mean Square Error: {}".format(mse))

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("Root Mean Square Error: {}".format(rmse))

    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error: {}".format(mae))

    misclassRate = 1-accuracy_score(y_test, y_pred)
    print("Misclassification Rate: {}".format(misclassRate))

    mcc = matthews_corrcoef(y_test, y_pred)
    print("Matthews Correlation Coefficient Rate: {}".format(mcc))

    trainingTime = endTrain - startTrain
    print("Training Time: {}".format(trainingTime))

    testingTime = endTest - startTest
    print("Testing Time: {}".format(testingTime))

    totalTime = trainingTime + testingTime
    print("Total Time: {}".format(totalTime))

    return [classifierName, acc, prec, rec, f1score, mse, rmse, mae, misclassRate, mcc, trainingTime, testingTime, totalTime]

def ApplyEncoder(OriginalColumn): 
    global df
    Encoder = LabelEncoder()
    Encoder.fit(df[OriginalColumn])
    return Encoder.transform(df[OriginalColumn])

# Data Acquisition
try:
  df = pd.read_csv('dataset/uci3.csv')
except:
  df = pd.read_csv('DjangoAPI/dataset/uci3.csv')

# size of the dataset, (rows, cols)
print("Size of UCI dataset: " + str(df.shape))

# Data Understanding
# information of the dataframe
print(df.info())

# description of the dataframe
print("describe 1:")
print(df.describe().transpose())

# number of null attributes in each column
nullAttributes = df.isnull().sum()
missingVals = df.isna().sum() #df.isnull().sum()
print(nullAttributes)
print(missingVals)

# Identifying presence of any duplicate rows in the dataset
duplicate = df[df.duplicated()]
print("\nDuplicate Rows :")
print(duplicate)
print()

# Data Visualisation and EDA
# Distribution of Heart Disease
# sns.countplot(x = "num", data = df)
print(df["num"].value_counts())  #see the distribution of heart disease level


# Data Preprocessing
df['HD'] = [1 if x == 1 or x == 2 or x == 3 or x == 4 else 0 for x in df['num']] #separate column for presence/absence of heart disease
# sns.countplot(x = "HD", data = df)

#see the distribution of heart disease presence/absence 
# shows that the dataset is not much imbalanced, so there is no need to balance
print(df.loc[:, 'HD'].value_counts())
print("Counter 1:", format(Counter(df['HD'])))

# Dropping attributes unnecessary for further processing
df.drop(['id', 'dataset'], inplace=True, axis=1)
print(df.shape)


# Data Cleaning

# convert object/categorical datatype values to numerical dummy values, using label encoder 
# (removed because it is found that one hot encoding attributes performed better)
# obj_cols = df.columns[df.dtypes == "object"]
# list(obj_cols)
# for col in obj_cols:
#     df[col] = ApplyEncoder(col)

xtr= df.iloc[:,0:13]
y = df.iloc[:,13] # including only num, if HD, include 14

## Methods to Convert Categorical Data to Numerical
transformed_x=pd.get_dummies(xtr)
print("transformed X") # visualising effects of pd.get dummies
print(transformed_x.transpose())

# Data Imputation
# Replacing missing values
print("before null")
print(transformed_x.isnull().sum())

imputer = KNNImputer(n_neighbors=5)
transformed_x = imputer.fit_transform(transformed_x)

print("after null")
transformed_x = pd.DataFrame(transformed_x)
print(transformed_x.isnull().sum())

# SMOTE
smt = SMOTE(sampling_strategy='not majority')
transformed_x, y = smt.fit_resample(transformed_x, y)

print("Counter 2: {}", format(Counter(df['HD'])))

# Data Scaling
sc = MinMaxScaler()
fittedX = sc.fit(transformed_x)
filename = "scalers.pkl"
joblib.dump(fittedX, filename)
transformed_x = sc.transform(transformed_x)

# to view the effects of scaler on the dataset distribution
print("describe 2:")
print(df.describe().transpose())

# Data Partitioning
x_train, x_test, y_train, y_test = train_test_split(transformed_x, y, test_size=0.2, random_state=42,shuffle=True)

#Model Creation
compare_models = []

#Ridge Classifier - Linear Model
print("Ridge Classifier with Hyperparameter Tuning")
parameters = {'solver':("auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"), 'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
parameters = {'solver':["sag"], 'alpha': [0.3]}

rdg_model_wh = RidgeClassifier()
clf = GridSearchCV(rdg_model_wh, parameters, verbose=2)
clf.fit(x_train, y_train)

rdg_best_param = clf.best_params_
print("Best params for ridge:", rdg_best_param)
bestRDGSolver = rdg_best_param.get("solver")
bestRDGAlpha = rdg_best_param.get("alpha")

y_pred = clf.predict(x_test)
RGresults1 = modelValidation(y_test, y_pred)

print("\n#######\nRidge Classifier with Best Paramaters Applied, {} solver, {} alpha\n".format(bestRDGSolver,bestRDGAlpha))
rdg_model = RidgeClassifier(alpha= bestRDGAlpha, solver= bestRDGSolver)

startTrain = time.time()
rdg_model.fit(x_train, y_train)
endTrain = time.time()

startTest = time.time()
y_pred = rdg_model.predict(x_test)
endTest = time.time()

RGresults2 = modelValidation(y_test, y_pred, startTrain, endTrain, startTest, endTest, "Ridge")
compare_models.append(RGresults2)

#Support Vector Machines Model
print("\n#######\nSupport Vector Machines with Hyperparameter Tuning\n")
parameters = {'kernel':('linear', 'rbf', 'poly', 'sigmoid'), 'C':[0.1, 0.5, 1, 2, 5, 10, 20],
              'gamma':[0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1]}
parameters = {'kernel':['rbf'], 'C':[20],
              'gamma':[1]}

svc_model_wh = SVC(decision_function_shape='ovo')
clf = GridSearchCV(svc_model_wh, parameters, verbose=2)
clf.fit(x_train, y_train)

svc_best_param = clf.best_params_
print("Best params for SVC:", svc_best_param)
bestSVCKernel = svc_best_param.get("kernel")
bestSVCC = svc_best_param.get("C")
bestSVCGamma = svc_best_param.get("gamma")

y_pred = clf.predict(x_test)
SVCresults1 = modelValidation(y_test, y_pred)

print("\n#######\nSVC with Best Paramaters Applied, {} kernel, {} C, {} gamma\n".format(bestSVCKernel,bestSVCC,bestSVCGamma))
svc_model = SVC(kernel= bestSVCKernel, C= bestSVCC, gamma=bestSVCGamma)

startTrain = time.time()
svc_model.fit(x_train, y_train)
endTrain = time.time()

startTest = time.time()
y_pred = svc_model.predict(x_test)
endTest = time.time()

SVCresults2 = modelValidation(y_test, y_pred, startTrain, endTrain, startTest, endTest, "SVC")
compare_models.append(SVCresults2)

#K-Nearest Neighbors Model
print("\n#######\nK-Nearest Neighbors\n")
parameters = {'n_neighbors' : [3,5,7,9,11,13,15,17,19,21,23,25,27,29], 'weights' :['uniform', 'distance']}
parameters = {'n_neighbors' : [3], 'weights' :['distance']}

knn_model_wh = KNeighborsClassifier()
clf = GridSearchCV(knn_model_wh, parameters, verbose=2)
clf.fit(x_train, y_train)

knn_best_param = clf.best_params_
print("Best params for KNN:", knn_best_param)
bestKNNNeighbours = knn_best_param.get("n_neighbors")
bestKNNWeights = knn_best_param.get("weights")

y_pred = clf.predict(x_test)
KNNResults1 = modelValidation(y_test, y_pred)

print("\n#######\nKNN with Best Paramaters Applied, {} neighbours, {} weights\n".format(bestKNNNeighbours,bestKNNWeights))
knn_model = KNeighborsClassifier(n_neighbors= bestKNNNeighbours, weights= bestKNNWeights)

startTrain = time.time()
knn_model.fit(x_train, y_train)
endTrain = time.time()

startTest = time.time()
y_pred = knn_model.predict(x_test)
endTest = time.time()

KNNresults2 = modelValidation(y_test, y_pred, startTrain, endTrain, startTest, endTest, "KNN")
compare_models.append(KNNresults2)

#Naive Bayes (Gaussian Naive Bayes) Model
print("\n#######\nNaive Bayes (Gaussian Naive Bayes)\n")
parameters = {'var_smoothing': np.logspace(0,-9, num=100)}
parameters = {'var_smoothing': [0.43287612810830584]}

nb_model_wh = GaussianNB()
clf = GridSearchCV(nb_model_wh, parameters, verbose=2)
clf.fit(x_train, y_train)

nb_best_param = clf.best_params_
print("Best params for NB:", nb_best_param)
bestNBVarSmoothing = nb_best_param.get("var_smoothing")

y_pred = clf.predict(x_test)
NBResults1 = modelValidation(y_test, y_pred)

print("\n#######\nNB with Best Paramaters Applied, {} var_smoothing \n".format(bestNBVarSmoothing))
nb_model = GaussianNB(var_smoothing= bestNBVarSmoothing)

startTrain = time.time()
nb_model.fit(x_train, y_train)
endTrain = time.time()

startTest = time.time()
y_pred = nb_model.predict(x_test)
endTest = time.time()

NBresults2 = modelValidation(y_test, y_pred, startTrain, endTrain, startTest, endTest, "NB")
compare_models.append(NBresults2)


#Decision Trees Model
print("\n#######\nDecision Trees\n")

parameters = {
    'max_depth' : [None,4,5,6,7,8,9],
    'criterion' :['gini', 'entropy','log_loss']
   }
parameters = {
    'max_depth' : [None],
    'criterion' :['gini']
   }

dt_model_wh = DecisionTreeClassifier()
clf = GridSearchCV(dt_model_wh, parameters, verbose=2)
clf.fit(x_train, y_train)

dt_best_param = clf.best_params_
print("Best params for DT:", dt_best_param)
bestDTMaxDepth = dt_best_param.get("max_depth")
bestDTCriterion = dt_best_param.get("criterion")

y_pred = clf.predict(x_test)
DTresults1 = modelValidation(y_test, y_pred)

print("\n#######\nDecision Tree with Best Paramaters Applied, {} max depth, {} criterion\n".format(bestDTMaxDepth,bestDTCriterion))
dt_model = DecisionTreeClassifier(max_depth= bestDTMaxDepth, criterion= bestDTCriterion)

startTrain = time.time()
dt_model.fit(x_train, y_train)
endTrain = time.time()

startTest = time.time()
y_pred = dt_model.predict(x_test)
endTest = time.time()

DTresults2 = modelValidation(y_test, y_pred, startTrain, endTrain, startTest, endTest, "DT")
compare_models.append(DTresults2)

# Gradient Boosting Model
print("\n#######\nGradient Boosting\n")
parameters = {
    'n_estimators': [100, 150, 200],
    'max_depth': [None,6,7],
    'subsample': np.arange(0.05, 1.01, 0.05),
    'n_jobs': [1],
    'verbosity': [0]
        }
parameters = {
    'n_estimators': [100],
    'max_depth': [None],
    'subsample': [0.75],
    'n_jobs': [1],
    'verbosity': [0]
        }
xgb_model_wh = XGBClassifier()
clf = GridSearchCV(xgb_model_wh, parameters, verbose=2)
clf.fit(x_train, y_train)

xgb_best_param = clf.best_params_
print("Best params for XGB:", xgb_best_param)
bestXGBEstimators = xgb_best_param.get("n_estimators")
bestXGBMaxDepth = xgb_best_param.get("max_depth")
bestXGBSubsample = xgb_best_param.get("subsample")

y_pred = clf.predict(x_test)
XGBResults1 = modelValidation(y_test, y_pred)

print("\n#######\nXGB with Best Paramaters Applied, {} no. of estimators, {} max depth, {} sub sample\n".format(bestXGBEstimators,bestXGBMaxDepth,bestXGBSubsample))
xgb_model = XGBClassifier(n_estimators= bestXGBEstimators, max_depth= bestXGBMaxDepth, subsample=bestXGBSubsample)

startTrain = time.time()
xgb_model.fit(x_train, y_train)
endTrain = time.time()

startTest = time.time()
y_pred = xgb_model.predict(x_test)
endTest = time.time()

XGBresults2 = modelValidation(y_test, y_pred, startTrain, endTrain, startTest, endTest, "XGB")
compare_models.append(XGBresults2)

#Random Forest Model
print("\n#######\nRandom Forest\n")
parameters = {
    'n_estimators':  [100, 150, 200],
    'max_features': ['auto', 'sqrt', 'log2', None],
    'max_depth' : [None, 4,5,6,7],
    'criterion' :['gini', 'entropy','log_loss']
   }
parameters = {
    'n_estimators':  [200],
    'max_features': ['auto'],
    'max_depth' : [None],
    'criterion' :['gini']
   }
rf_model_wh = RandomForestClassifier()
clf = GridSearchCV(rf_model_wh, parameters, verbose=2)
clf.fit(x_train, y_train)

rf_best_param = clf.best_params_
print("Best params for random forest:", rf_best_param)
bestRFNEstimators = rf_best_param.get("n_estimators")
bestRFMaxFeatures = rf_best_param.get("max_features")
bestRFMaxDepth = rf_best_param.get("max_depth")
bestRFCriterion = rf_best_param.get("criterion")

y_pred = clf.predict(x_test)
RFresults1 = modelValidation(y_test, y_pred)

print("\n#######\nRandom Forest with Best Paramaters Applied, {} no of estimators, {} max features, {} max depth, {} criterion\n".format(bestRFNEstimators, bestRFMaxFeatures,bestRFMaxDepth,bestRFCriterion))
rf_model = RandomForestClassifier(n_estimators=bestRFNEstimators, max_features= bestRFMaxFeatures, max_depth= bestRFMaxDepth, criterion=bestRFCriterion)

startTrain = time.time()
rf_model.fit(x_train, y_train)
endTrain = time.time()

startTest = time.time()
y_pred = rf_model.predict(x_test)
endTest = time.time()

RFresults2 = modelValidation(y_test, y_pred, startTrain, endTrain, startTest, endTest, "RF")
compare_models.append(RFresults2)



#Extra Trees Model
print("\n#######\nExtraTreesClassifier\n")

parameters = {
    'n_estimators':  [10, 20, 50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [None, 4,5,6,7],
    'criterion' :['gini', 'entropy','log_loss']
   }
parameters = {
    'n_estimators':  [200],
    'max_features': ['log2'],
    'max_depth' : [None],
    'criterion' :['gini']
   }
ext_model_wh = ExtraTreesClassifier()
clf = GridSearchCV(ext_model_wh, parameters, verbose=2)
clf.fit(x_train, y_train)

ext_best_param = clf.best_params_
print("Best params for extra trees:", ext_best_param)
bestEXTNEstimators = ext_best_param.get("n_estimators")
bestEXTMaxFeatures = ext_best_param.get("max_features")
bestEXTMaxDepth = ext_best_param.get("max_depth")
bestEXTCriterion = ext_best_param.get("criterion")

y_pred = clf.predict(x_test)
EXTresults1 = modelValidation(y_test, y_pred)

print("\n#######\nExtra Trees with Best Paramaters Applied, {} no of estimators, {} max features, {} max depth, {} criterion\n".format(bestEXTNEstimators, bestEXTMaxFeatures,bestEXTMaxDepth,bestEXTCriterion))
ext_model = ExtraTreesClassifier(n_estimators=bestEXTNEstimators, max_features= bestEXTMaxFeatures, max_depth= bestEXTMaxDepth, criterion=bestEXTCriterion)

startTrain = time.time()
ext_model.fit(x_train, y_train)
endTrain = time.time()

startTest = time.time()
y_pred = ext_model.predict(x_test)
endTest = time.time()

EXTresults2 = modelValidation(y_test, y_pred, startTrain, endTrain, startTest, endTest, "EXT")
compare_models.append(EXTresults2)


#K-Means Clustering
print("\n#######\nK Means Clustering\n")

parameters = {
    'n_clusters':  [2,3,5,10, 20, 50],
    'max_iter': [50, 100, 150],
    'algorithm' :['lloyd', 'elkan','auto','full']
   }

parameters = {
    'n_clusters':  [200],
    'max_iter': [150],
    'algorithm' :['lloyd']
   }

kmc_model_wh = KMeans()
clf = GridSearchCV(kmc_model_wh, parameters, verbose=2)
clf.fit(x_train, y_train)

kmc_best_param = clf.best_params_
print("Best params for K-Means Clustering:", kmc_best_param)
bestKMCClusters = kmc_best_param.get("n_clusters")
bestKMCIters = kmc_best_param.get("max_iter")
bestKMCAlgo = kmc_best_param.get("algorithm")

y_pred = clf.predict(x_test)
KMCResults1 = modelValidation(y_test, y_pred)

print("\n#######\nK-Means Clustering with Best Paramaters Applied, {} no. of clusters, {} max iteration, {} algorithm\n".format(bestKMCClusters,bestKMCIters,bestKMCAlgo))
kmc_model = KMeans(n_clusters= bestKMCClusters, max_iter= bestKMCIters, algorithm=bestKMCAlgo)

startTrain = time.time()
kmc_model.fit(x_train, y_train)
endTrain = time.time()

startTest = time.time()
y_pred = kmc_model.predict(x_test)
endTest = time.time()

KMCresults2 = modelValidation(y_test, y_pred, startTrain, endTrain, startTest, endTest, "K-Means")
compare_models.append(KMCresults2)

#Logistic Regression
print("\n#######\nLogistic Regression\n")
parameters = {'C':[1.0, 10.0, 100.0, 1000.0], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
parameters = {'C':[1000.0], 'solver': ['lbfgs']}

lr_model_wh = LogisticRegression()
clf = GridSearchCV(lr_model_wh, parameters, verbose=2)
clf.fit(x_train, y_train)

lr_best_param = clf.best_params_
print("Best params for Logistic Regression:", lr_best_param)
bestLRC = lr_best_param.get("C")
bestLRSolver = lr_best_param.get("solver")

y_pred = clf.predict(x_test)
LRresults1 = modelValidation(y_test, y_pred)

print("\n#######\nLogistic Regression with Best Paramaters Applied, {} C, {} Solver\n".format(bestLRC, bestLRSolver))
lr_model = LogisticRegression(C=bestLRC, solver= bestLRSolver)

startTrain = time.time()
lr_model.fit(x_train, y_train)
endTrain = time.time()

startTest = time.time()
y_pred = lr_model.predict(x_test)
endTest = time.time()

LRresults2 = modelValidation(y_test, y_pred, startTrain, endTrain, startTest, endTest, "LR")
compare_models.append(LRresults2)

# ANN Multilayer Perceptron
print("\n#######\nArtificial Neural Network - Multilayer Perceptron\n")
activation = "tanh" # previous test shows combination of tanh and adam produced the best scores
solver = "adam"
hidden_layer = (27, 27, 27) # previous test shows this combination of hidden layer neuron produced the best scores
max_iteration = 723 # previous test shows this maximum iteration value produced the best scores

mlp_model = MLPClassifier(hidden_layer, activation=activation, solver=solver, max_iter=max_iteration, shuffle=False, random_state=42)

startTrain = time.time()
mlp_model.fit(x_train, y_train)
endTrain = time.time()

startTest = time.time()
y_pred = mlp_model.predict(x_test)
endTest = time.time()

ANNresults = modelValidation(y_test, y_pred, startTrain, endTrain, startTest, endTest, "ANN")
compare_models.append(ANNresults)

# Stacking Classifier
print("\n#######\nStacking Classifier\n")
estimators = [('SVC', svc_model),
              ('KNN', knn_model),
              ('DT', dt_model),
              ('XGB', xgb_model),
              ('RF', rf_model),
              ('EXT', ext_model),
              ('ANN', mlp_model)
             ]
estimatorslist=[["SVC", svc_model], ["KNN", knn_model], ["DT", dt_model], ["XGB",xgb_model], ["RF",rf_model], ["EXT", ext_model], ["MLP", mlp_model]]

for model in estimatorslist:
  print("\n#######\nStacking Classifier - {}\n".format(model[0]))
  stack_model = StackingClassifier(estimators = estimators, final_estimator = model[1])

  startTrain = time.time()
  stack_model.fit(x_train, y_train)
  endTrain = time.time()

  startTest = time.time()
  y_pred = stack_model.predict(x_test)
  endTest = time.time()

  Stackresults = modelValidation(y_test, y_pred, startTrain, endTrain, startTest, endTest, "Stack-{}".format(model[0]))
  compare_models.append(Stackresults)

  # Checking the feature importance of the best performing model (the chosen model after model comparison was the Stacking classifier with RandomForest as final estimator)
  if model[0] == "RF":
    col = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca", "sex_Female", "sex_Male", "cp_asymptomatic", "cp_atypical angina", "cp_non-anginal", "cp_typical angina", "fbs_False", "fbs_True", "restecg_lv hypertrophy", "restecg_normal", "restecg_st-t abnormality", "exang_False", "exang_True", "slope_downsloping", "slope_flat", "slope_upsloping", "thal_fixed defect", "thal_normal", "thal_reversable defect"]
    feature = pd.Series(rf_model.feature_importances_, index = col).sort_values(ascending = False)
    print("\n#######\Feature Importance\n")
    print(feature)

    plt.figure(figsize = (10,6))
    sns.barplot(x = feature, y = feature.index)
    plt.title("Feature Importance")
    plt.xlabel('Score')
    plt.ylabel('Features')
    plt.show()

    ##Saving/Pickling the best-performing chosen ML Model##
    filename = "heart-strat.pkl"
    joblib.dump(stack_model, filename)

# Model Comparison/Selection
print("\n#######\Comparison of Model Results\n")
score_frame = pd.DataFrame(compare_models)
score_frame.columns=["Name", "Acc", "Prec", "Rec", "F1-Score", "MSE", "RMSE", "MAE", "Misclass Rate", "MCC", "Training Time (s)", "Testing Time (s)", "TotalTime (s)"]
print(score_frame)
score_frame.to_excel("ModelCompare.xlsx")

# ANN Multilayer Perceptron
#testing effect of different hidden layers
def nn_pipeline(x_train, x_test, y_train, y_test, hidden_layer_size, activation, solver, max_iter, shuffle = False, random_state = 42):
    mlp = MLPClassifier(hidden_layer_size, activation=activation, solver=solver, max_iter=max_iter, shuffle=shuffle, random_state=random_state)
    
    startTrain = time.time()
    mlp.fit(x_train, y_train)
    endTrain = time.time()

    startTest = time.time()
    y_pred = mlp.predict(x_test)
    endTest = time.time()

    ANNResults = modelValidation(y_test, y_pred, startTrain, endTrain, startTest, endTest)
    return ANNResults

#testing effect of different activation function and solvers
def MLPActivationAndSolverTest():
    activation_functions = ["identity", "logistic", "tanh", "relu"]
    solvers = ["lbfgs", "sgd", "adam"]
    
    max_iteration = 1000
    hidden_layer = (32, 16) # replace with the best hidden layer found in the next test
    testedParameterAndMetrics = []
    
    for activation in activation_functions:
        for solver in solvers:
            print("Test {} activation with {} solver".format(activation, solver))
            ANNResults = nn_pipeline(x_train, x_test, y_train, y_test,hidden_layer,activation,solver,max_iteration)
            acc = ANNResults[1]
            prec = ANNResults[2]
            f1 = ANNResults[4]
            testedParameterAndMetrics.append([activation,solver, acc, f1, prec])
            print()
            
    testedParameterAndMetrics.sort(key=lambda x:(x[2], x[3], x[4]), reverse=True)
    print(testedParameterAndMetrics[0:3]) # change to just 0 to get best parameter
    nn_score_frame = pd.DataFrame(testedParameterAndMetrics)  
    nn_score_frame.columns=["Activation", "Solver", "Acc", "F1-Score", "Prec"]
    print(nn_score_frame)
    
#testing effect of different number of layers and neurons (structure of NN)
def MLPNoOfLayerAndNeuron():
    activation = "tanh" # previous test shows combination of tanh and adam produced the best scores
    solver = "adam"
    max_layers = 3
    max_neuron = 30
    max_iteration = 1000
    testedParameterAndMetrics = []
    for noOfLayers in range(1, max_layers+1):
        testedParMet = []
        for noOfNeurons in range(1,max_neuron+1):
            print("Test {} hidden layers with {} neurons".format(noOfLayers, noOfNeurons))
            hidden_layer = tuple([noOfNeurons for noLayer in range(noOfLayers)])
            ANNResults = nn_pipeline(x_train, x_test, y_train, y_test,hidden_layer,activation,solver,max_iteration)
            acc = ANNResults[1]
            prec = ANNResults[2]
            f1 = ANNResults[4]
            trainingTime = ANNResults[10]
            testingTime = ANNResults[11]
            totalTime = ANNResults[12]
            testedParameterAndMetrics.append([noOfLayers,noOfNeurons,hidden_layer, acc, f1, prec, trainingTime, testingTime, totalTime])
            testedParMet.append([noOfLayers,noOfNeurons,hidden_layer, acc, f1, prec, trainingTime, testingTime, totalTime])
            print()
        x = [i for i in range(1,max_neuron+1)]
        y = [parmet[3] for parmet in testedParMet]
        plt.plot(x, y)
        plt.title("No. of Neurons Test. {} Hidden Layer".format(noOfLayers))
        plt.ylabel("Acc Score")
        plt.xlabel("No. of Neurons")
        plt.show()
    
    testedParameterAndMetrics.sort(key=lambda x:(x[3], x[4], x[5]), reverse=True)
    print(testedParameterAndMetrics[0:5]) # change to just 0 to get best parameter        
    nn_score_frame = pd.DataFrame(testedParameterAndMetrics[0:10][:])  
    nn_score_frame.columns=["No. of Hidden Layers", "No. of Neurons", "Hidden Layers", "Acc", "F1-Score", "Prec", "Training Time (s)", "Testing Time (s)", "Total Time (s)"]
    print(nn_score_frame)
            
#testing effect of different max iteration
def MLPMaxIterTest():
    activation = "tanh" # previous test shows combination of tanh and adam produced the best scores
    solver = "adam"
    hidden_layer = (27, 27, 27) # replace with best hidden layer neuron combination
    max_iteration = 1000
    testedParameterAndMetrics = []
    
    for max_i in range(1, max_iteration):
        print("Test {} maximum iterations".format(max_i))
        ANNResults = nn_pipeline(x_train, x_test, y_train, y_test,hidden_layer,activation,solver,max_i)
        acc = ANNResults[1]
        prec = ANNResults[2]
        f1 = ANNResults[4]
        trainingTime = ANNResults[10]
        testingTime = ANNResults[11]
        totalTime = ANNResults[12]
        testedParameterAndMetrics.append([max_i, acc, f1, prec, trainingTime, testingTime, totalTime])
        print()
        # plot a graph and view the elbow point, where subsequent iterations doesn't result in increase in accuracy/score
    #enter statements
    x = [i for i in range(1,max_iteration)]
    y = [parmet[1] for parmet in testedParameterAndMetrics]
    plt.plot(x, y)
    plt.title("Max Iteration Test")
    plt.ylabel("Acc Score")
    plt.xlabel("Max No. of Iterations")
    plt.show()
    testedParameterAndMetrics.sort(key=lambda x:(x[1]), reverse=True)
    print(testedParameterAndMetrics[0:20])

    nn_score_frame = pd.DataFrame(testedParameterAndMetrics[0:10][:])
    nn_score_frame.columns=["Maximum No. of Iterations", "Acc", "F1-Score", "Prec", "Training Time (s)", "Testing Time (s)", "Total Time (s)"]
    print(nn_score_frame)
# Uncomment any of the following to initiate the corresponding test
# MLPActivationAndSolverTest()
# MLPNoOfLayerAndNeuron()
# MLPMaxIterTest()

# Other Models
# SGD Model
# print("\n#######\SGDClassifier 1\n")
# clf = SGDClassifier(max_iter=1000, tol=1e-4)
# clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)
# modelValidation(y_test, y_pred)

# AdaBoost Classifier
# adb = AdaBoostClassifier(n_estimators=500, learning_rate=0.01, random_state=0)
# model = adb.fit(x_train, y_train)
# y_pred_adaboost = model.predict(x_test)
# modelValidation(y_test, y_pred_adaboost)

# print(y_pred_adaboost)
# print("AdaBoost Classifier Model Accuracy:", accuracy_score(y_test, y_pred_adaboost))
# print(classification_report(y_test, y_pred_adaboost))

# KERAS NN
# classifier = Sequential()
# classifier.add(Dense(200, activation='relu', kernel_initializer='random_normal', input_dim=x_test.shape[1]))
# classifier.add(Dense(400, activation='relu', kernel_initializer='random_normal'))
# classifier.add(Dense(4, activation='relu' , kernel_initializer='random_normal'))
# classifier.add(Dense(1, activation='sigmoid' , kernel_initializer='random_normal'))
# classifier.compile(optimizer ='adam' ,loss='binary_crossentropy', metrics = ['accuracy'])
# classifier.fit(x_train, y_train, batch_size=20, epochs=50, verbose=0)
# eval_model = classifier.evaluate(x_train, y_train)
# print(eval_model)

# y_pred = classifier.predict(x_test)
# y_pred = (y_pred > 0.5)
# cm = confusion_matrix(y_test,y_pred)
# accuracy = (cm[0][0]+cm[1][1])/(cm[0][1] + cm[1][0] +cm[0][0] +cm[1][1])
# print(accuracy*100)

# y_pred = classifier.predict(x_test)
# y_pred = (y_pred > 0.5) #limiter
# confusion_matrix(y_test, y_pred)

# print(classification_report(y_test, y_pred))
# print("AUC-ROC Score: {}".format(roc_auc_score(y_test, y_pred, multi_class="ovo", average="macro")))

# cm=confusion_matrix(y_test, y_pred)
# ax = plt.subplot()
# sns.heatmap(cm, annot=True, ax = ax) #annot-True to annotate cells.

#labels, title and ticks
# ax.set_xlabel('Predicted');ax.set_ylabel('Actual')
# ax.set_title('Confusion Matrix')
# ax.xaxis.set_ticklabels(['No', 'Yes']);ax.yaxis.set_ticklabels(['No', 'Yes' ])