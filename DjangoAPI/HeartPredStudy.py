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
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# Importing Libraries - Model Creation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

# Importing Libraries - Model Creation (Neural Networks - keras)
from keras.models import Sequential
from keras.layers import Dense
# from numpy import loadtxt

# Importing Libraries - Model Creation (K-Means Clustering)
# from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

import pickle
import joblib
import time

# Importing Libraries - Model Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score,confusion_matrix, recall_score, precision_score, classification_report, average_precision_score, mean_squared_error, mean_absolute_error, multilabel_confusion_matrix,matthews_corrcoef


def visualise(mlp):
  # get number of neurons in each layer
  n_neurons = [len(layer) for layer in mlp.coefs_]
  n_neurons.append(mlp.n_outputs_)

  # calculate the coordinates of each neuron on the graph
  y_range = [0, max(n_neurons)]
  x_range = [0, len(n_neurons)]
  loc_neurons = [[[l, (n+1)*(y_range[1]/(layer+1))] for n in range(layer)] for l,layer in enumerate(n_neurons)]
  x_neurons = [x for layer in loc_neurons for x,y in layer]
  y_neurons = [y for layer in loc_neurons for x,y in layer]

  # identify the range of weights
  weight_range = [min([layer.min() for layer in mlp.coefs_]), max([layer.max() for layer in mlp.coefs_])]

  # prepare the figure
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  # draw the neurons
  ax.scatter(x_neurons, y_neurons, s=100, zorder=5)
  # draw the connections with line width corresponds to the weight of the connection
  for l,layer in enumerate(mlp.coefs_):
    for i,neuron in enumerate(layer):
      for j,w in enumerate(neuron):
        ax.plot([loc_neurons[l][i][0], loc_neurons[l+1][j][0]], [loc_neurons[l][i][1], loc_neurons[l+1][j][1]], 'white', linewidth=((w-weight_range[0])/(weight_range[1]-weight_range[0])*5+0.2)*1.2)
        ax.plot([loc_neurons[l][i][0], loc_neurons[l+1][j][0]], [loc_neurons[l][i][1], loc_neurons[l+1][j][1]], 'grey', linewidth=(w-weight_range[0])/(weight_range[1]-weight_range[0])*5+0.2)

def vis2d(ax, model, X_train, Y_train, X_test=[], Y_test=[]):
  # identify graph range
  x_range = [X_train[:,0].min()-0.5, X_train[:,0].max()+0.5]
  y_range = [X_train[:,1].min()-0.5, X_train[:,1].max()+0.5]
  if len(X_test) > 0:
    x_range = [min(x_range[0], X_test[:,0].min()-0.5), max(x_range[1], X_test[:,0].max()+0.5)]
    y_range = [min(y_range[0], X_test[:,1].min()-0.5), max(y_range[1], X_test[:,1].max()+0.5)]
  # create a meshgrid
  xx, yy = np.meshgrid(np.arange(x_range[0], x_range[1], .01), np.arange(y_range[0], y_range[1], .01))
  # identify the area of decision
  Z = model.predict([[x,y] for x,y in zip(xx.ravel(), yy.ravel())])
  Z = Z.reshape(xx.shape)
  # plot the decision areas
  ax.contourf(xx,yy,Z,alpha=.8)
  # plot the training and testing data
  ax.scatter([x[0] for x in X_train], [x[1] for x in X_train], c=Y_train, edgecolors='black')
  if len(X_test) > 0:
    ax.scatter([x[0] for x in X_test], [x[1] for x in X_test], c=Y_test, edgecolors='brown', alpha=.8)

def vis3d(fig, model, X_train, Y_train, X_test=[], Y_test=[]):
  possible_class = np.unique(Y_train)
  y_range = [0, 1]
  y_data_min = X_train.min(axis=0)
  y_data_max = X_train.max(axis=0)
  if len(X_test) > 0:
    y_data_min = np.amin([y_data_min, X_test.min(axis=0)], axis=0)
    y_data_max = np.amax([y_data_max, X_test.max(axis=0)], axis=0)
  single_y = np.arange(y_range[0], y_range[1], .1)
  single_y = single_y.reshape(len(single_y), 1)
  yy = []
  for i in range(X_train.shape[1]):
    if len(yy) == 0:
      yy = np.tile(single_y,1)
    else:
      old = np.tile(yy, (single_y.shape[0],1))
      new = np.repeat(single_y, yy.shape[0])
      new = new.reshape(len(new),1)
      yy = np.hstack([new, old])
  yy_data = [[yi*(y_data_max[i] - y_data_min[i])+y_data_min[i] for i,yi in enumerate(y)] for y in yy]
  zz = model.predict(yy_data)
  train_x = (X_train - y_data_min)/(y_data_max - y_data_min)
  axes = []
  for i in possible_class:
    ax = fig.add_subplot(len(possible_class), 1, i+1)
    ax.plot(yy[zz == i].transpose(), c=cm.Set2.colors[i%cm.Set2.N], alpha=0.5)
    ax.plot(train_x[Y_train == i].transpose(), c='black', lw=5, alpha=.8)
    ax.plot(train_x[Y_train == i].transpose(), c=cm.Dark2.colors[i%cm.Set2.N], lw=3, alpha=.8)
    ax.set_title("output = {}".format(i))
    ax.set_xticks([i for i in range(X_train.shape[1])])
    ax.set_ylim(y_range)
    axes.append(ax)
  return axes

def modelValidation(y_test, y_pred):
    print("Results")
    print(classification_report(y_test, y_pred))
    print("Accuracy Score: {}".format(accuracy_score(y_test, y_pred)))
    print("Precision Score: {}".format(precision_score(y_test, y_pred, average="micro")))
    print("Recall Score: {}".format(recall_score(y_test, y_pred, average="micro")))
    print("F1 Score Score: {}".format(f1_score(y_test, y_pred, average="micro")))
    print("Mean Square Error: {}".format(mean_squared_error(y_test, y_pred)))
    print("Root Mean Square Error: {}".format(mean_squared_error(y_test, y_pred, squared=False)))
    print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))
    print("Misclassification Rate: {}".format(1-accuracy_score(y_test, y_pred)))
    print("matthews_corrcoef Rate: {}".format(matthews_corrcoef(y_test, y_pred)))


# Data Acquisition
df = pd.read_csv('DjangoAPI/dataset/uci3.csv')

print("Size of UCI dataset: " + str(df.shape))

# Data Understanding
nullAttributes = df.isnull().sum()
print(df.info())
print("describe:")
print(df.describe().transpose())

missingVals = df.isna().sum() #df.isnull().sum()

#EDA

# Distribution of Heart Disease
# sns.countplot(x = "num", data = df)
df.loc[:, 'num'].value_counts()
print(df["num"].value_counts())  #see the distribution of heart disease level

# sns.countplot(x=df['HeartDisease'],hue='Sex',data=df)
# sns.countplot(x=df['HeartDisease'],hue='ChestPainType',data=df)

# sns.barplot(x=df['Sex'],y=df['RestingBP'],data=df)
# sns.barplot(x=df['Sex'],y=df['Cholesterol'],data=df)
# sns.barplot(x=df['HeartDisease'],y=df['Cholesterol'],data=df) #not indicative
# sns.barplot(x=df['HeartDisease'],y=df['RestingBP'],data=df)
# sns.barplot(x=df['Sex'],y=df['ST depression'],data=df)
# sns.barplot(x=df['HeartDisease'],y=df['ExerciseAngina'],data=df)
# sns.barplot(x=df['Sex'],y=df['ExerciseAngina'],data=df)

# sns.lineplot(x=df['Age'],y=df['BP'],data=df)
# sns.lineplot(x=df['Age'],y=df['Cholesterol'],data=df)
# sns.lineplot(x=df['Age'],y=df['ST depression'],data=df)
# Treating Outliers
# creating box plot to view distribution of dataset
# for i in df.iloc[:,:-1].columns:
#     sns.boxplot(df[i],data=df)
#     print(i)
#     plt.show()


# Data Preprocessing
df['HD'] = [1 if x == 1 or x == 2 or x == 3 or x == 4 else 0 for x in df['num']] #separate column for presence/absence of heart disease
# sns.countplot(x = "HD", data = df)
print(df.loc[:, 'HD'].value_counts()) #see the distribution of heart disease presence/absence

print("Counter 1:", format(Counter(df['HD'])))
Counter(df['HD'])

df.drop(['id', 'dataset'], inplace=True, axis=1)
print(df.shape)

print("df.dtypes")
print(df.dtypes) # to identify the datatypes, identify the categorical datatype values to convert to dummy variables




xtr= df.iloc[:,0:13]
y = df.iloc[:,13] # including only num, if HD, include 14
# print(xtr.head())

# y = pd.get_dummies(y) # to convert categorical variable into dummy/indicator variables
# print(y.shape)
# print(xtr.shape)

le=LabelEncoder()
print("before label encoder")
print(df)
df['sex']=le.fit_transform(df['sex'])
df['restecg']=le.fit_transform(df['restecg'])
df['cp']=le.fit_transform(df['cp'])
df['exang']=le.fit_transform(df['exang'])
df['slope']=le.fit_transform(df['slope'])
print("after label encoder")
print(df)


# Replacing missing values
xtr["trestbps"].fillna(xtr["trestbps"].mean(), inplace=True)
xtr["chol"].fillna(xtr["chol"].mean(), inplace=True)
xtr["fbs"].fillna(xtr["fbs"].mode()[0], inplace=True)
xtr["restecg"].fillna(xtr["restecg"].mode()[0], inplace=True)
xtr["thalch"].fillna(xtr["thalch"].mean(), inplace=True)
xtr["exang"].fillna(xtr["exang"].mode()[0], inplace=True)
xtr["oldpeak"].fillna(xtr["oldpeak"].mean(), inplace=True)
xtr["slope"].fillna(xtr["slope"].mode()[0], inplace=True)
xtr["ca"].fillna(xtr["ca"].mean(), inplace=True)
xtr["thal"].fillna(xtr["thal"].mode()[0], inplace=True)
missingVals2 = df.isna().sum()


## Methods to Convert Categorical Data to Numerical
transformed_x=pd.get_dummies(xtr)

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


#heatmap
# corr = df.corr()
# plt.figure(figsize=(14,14))
# sns.heatmap(corr, annot=True, fmt= '.2f',annot_kws={'size': 15}, cmap= 'coolwarm')
# plt.show()
# print(corr)
# plt.figure(figsize=(14,14))
# sns.heatmap(df.corr())
# plt.show()

# Data Partitioning
# x_train, x_test, y_train, y_test = train_test_split(transformed_x, y, test_size=0.3)
x_train, x_test, y_train, y_test = train_test_split(transformed_x, y, test_size=0.2, random_state=42,shuffle=True)

# Hyperparameter Tuning - Grid Search
# grid_search_rf = GridSearchCV(
#     estimator="est",
#     param_grid="param_grid"
# )
# grid_search_rf.fit(X_train, y_train)
# print('Best Parameters were: {}'.format(grid_search_rf.best_params_))
# print('Best CrossVal Score was:{}'.format(grid_search_rf.best_score_))

#Model Creation

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

# ANN Model with MLPClassifier
print("\n#######\nANN Model with MLPClassifier\n")
mlp = MLPClassifier(hidden_layer_sizes=(2), max_iter=10000)

mlp.partial_fit(x_train, y_train, np.unique(y_train))
visualise(mlp)


mlp.fit(x_train, y_train)
predictions = mlp.predict(x_test)
modelValidation(y_test, predictions)
visualise(mlp)
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# vis2d(ax, mlp, x_train, y_train, x_test, y_test)

# ANN Model with different Activation Function
print("\n#######\nANN Model with MLPClassifier Different Activation\n")
activation_functions = ['identity', 'logistic', 'tanh', 'relu']
# fig = plt.figure()
for i, actfcn in enumerate(activation_functions):
  print("\n#######\nANN Model with MLPClassifier and {} activation function\n".format(actfcn))
  mlp = MLPClassifier(hidden_layer_sizes=(3), activation=actfcn, max_iter=10000)
  mlp.fit(x_train, y_train)
  # ax = fig.add_subplot(1, len(activation_functions), i+1)
  # ax.set_title(actfcn)
  # vis2d(ax, mlp, x_train, y_train, x_test, y_test)
  predictions = mlp.predict(x_test)
  
  modelValidation(y_test, predictions)

# ANN Model with different number of hidden layers
print("\n#######\nANN Model with MLPClassifier Different hidden layer\n")
activation_functions = ['identity', 'logistic', 'tanh', 'relu']
hidden_layers = [(3), (3,3), (3,3,3)]
# fig = plt.figure()
for i,actfcn in enumerate(activation_functions):
  for j,hlyr in enumerate(hidden_layers):
    print("\n#######\nANN Model with MLPClassifier and {0} activation function and {1} hidden layer\n".format(actfcn,hlyr))
    mlp = MLPClassifier(hidden_layer_sizes=hlyr, activation=actfcn, max_iter=1000)
    mlp.fit(x_train, y_train)
    predictions = mlp.predict(x_test)
    modelValidation(y_test, predictions)

# Stratified K Fold
# kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
# cvscores = []
# for train, test in kfold.split(transformed_x, y) :
#     #create model
#     model = Sequential()
#     model.add(Dense(200, input_dim= 25, activation= 'relu'))
#     model.add(Dense(400, activation='relu'))
#     model.add(Dense(4, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
    
#     #Compile model
#     model.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['accuracy'])
    
#     #Fit the model
#     model.fit(transformed_x[train], y[train], epochs=100,verbose=0)
#     #evaluate the model
#     scores = model.evaluate(transformed_x[test], y[test], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)
# print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

##IMPORTANT - for pickling##
# filename = "heart-strat.pkl"
# joblib.dump(classifier, filename)


#Ridge Classifier - Linear Model
print("\n#######\nRidge Classifier\n")
lin_model = RidgeClassifier()
lin_model.fit(x_train, y_train)
y_preds = lin_model.predict(x_test)
modelValidation(y_test, y_preds)
# print(classification_report(y_test, y_preds))
# confusion_matrix(y_test, y_preds)
# accuracy_score(y_test, y_preds)

#Support Vector Machines Model
print("\n#######\nSupport Vector Machines\n")
svc_model = SVC(decision_function_shape='ovo')
svc_model.fit(x_train, y_train)
y_preds = svc_model.predict(x_test)
modelValidation(y_test, y_preds)

#Support Vector Machines Model 2 with grid search hyperparameter
# parameters = {'kernel':('linear', 'rbf'), 'C':[1.0, 10.0, 100.0, 1000.0],
#               'gamma':[1,0.1,0.01]}

# print("\n#######\nSupport Vector Machines with grid search hyperparameter\n")
# svc_model = SVC(decision_function_shape='ovo')
# clf = GridSearchCV(svc_model, parameters, verbose=2)
# clf.fit(x_train, y_train)
# svc_best_param = clf.best_params_
# print("Best params for SVM:", svc_best_param)
# y_pred = clf.predict(x_test)
# modelValidation(y_test, y_pred)
# print(classification_report(y_test,predict))
# print(confusion_matrix(y_test, predict))


#K-Nearest Neighbors Model
print("\n#######\nK-Nearest Neighbors\n")
knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(x_train, y_train)
y_preds = knn_model.predict(x_test)
modelValidation(y_test, y_preds)


#Naive Bayes (Gaussian Naive Bayes) Model
print("\n#######\nNaive Bayes (Gaussian Naive Bayes)\n")
nb_model = GaussianNB()
nb_model.fit(x_train,y_train)
y_preds = nb_model.predict(x_test)
modelValidation(y_test, y_preds)

#Decision Trees Model
print("\n#######\Decision Trees\n")
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train,y_train)
y_preds = dt_model.predict(x_test)
modelValidation(y_test, y_preds)

#Decision Trees Model 2
print("\n#######\Decision Trees 2\n")
weights = {0:1, 1:0.5, 2:0.5, 3:0.5, 4:0.5}
clf = DecisionTreeClassifier(criterion='entropy', max_depth=5)
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
modelValidation(y_test, y_pred)

#Gradient Boosting Model
print("\n#######\Gradient Boosting\n")
gradient_booster = GradientBoostingClassifier(learning_rate=0.02, max_depth=3, n_estimators=150)
gradient_booster.fit(x_train, y_train)
y_pred = gradient_booster.predict(x_test)
modelValidation(y_test, y_pred)

# Gradient Boosting Model 2
print("\n#######\Gradient Boosting 2\n")
xgb=XGBClassifier()
xgb.fit(x_train,y_train)
pred_3=xgb.predict(x_test)
modelValidation(y_test, pred_3)

#Gradient Boosting Model 3
print("\n#######\Gradient Boosting 3\n")
gradient_booster = GradientBoostingClassifier(learning_rate=0.01, max_depth=3, n_estimators=500)
gradient_booster.fit(x_train, y_train)
y_pred = gradient_booster.predict(x_test)
modelValidation(y_test, y_pred)

# AdaBoost Classifier
adb = AdaBoostClassifier(n_estimators=500, learning_rate=0.01, random_state=0)
model = adb.fit(x_train, y_train)
y_pred_adaboost = model.predict(x_test)
modelValidation(y_test, y_pred_adaboost)

# print(y_pred_adaboost)
# print("AdaBoost Classifier Model Accuracy:", accuracy_score(y_test, y_pred_adaboost))
# print(classification_report(y_test, y_pred_adaboost))


#Random Forest Model
print("\n#######\Random Forest\n")
rfc_model = RandomForestClassifier()
rfc_model.fit(x_train,y_train)
y_preds = rfc_model.predict(x_test)

modelValidation(y_test, y_preds)


#Random Forest Model
# parameters = {'kernel':('linear', 'rbf'), 'C':[1.0, 10.0, 100.0, 1000.0],'gamma':[1,0.1,0.01]}
# print("\n#######\Random Forest with Grid Search Hyperparameter Optimisation\n")
# rfc_model = RandomForestClassifier()
# clf = GridSearchCV(rfc_model, parameters, verbose=2)
# clf.fit(x_train,y_train)
# rf_best_param = clf.best_params_
# print("Best params for rf:", rf_best_param)
# y_preds = clf.predict(x_test)

# # y_preds = rfc_model.predict(x_test)
# print(classification_report(y_test, y_preds))
# confusion_matrix(y_test, y_preds)
# accuracy_score(y_test, y_preds)
# print(rfc_model.score(x_test,y_test))

# print("\n#######\nSupport Vector Machines 2\n")
# svc_model = SVC(decision_function_shape='ovo')
# clf = GridSearchCV(svc_model, parameters, verbose=2)
# clf.fit(x_train, y_train)
# svc_best_param = clf.best_params_
# print("Best params for SVM:", svc_best_param)
# predict = clf.predict(x_test)
# print(classification_report(y_test,predict))
# print(confusion_matrix(y_test, predict))


#Random Forest Model 2
print("\n#######\Random Forest 2\n")
clf = RandomForestClassifier(n_estimators=150)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
modelValidation(y_test, y_pred)

#Random Forest Model 3
print("\n#######\Random Forest 3 Entropy\n")
clf = RandomForestClassifier(criterion="entropy", n_estimators=150)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
modelValidation(y_test, y_pred)

#Random Forest Model 4
print("\n#######\Random Forest 4\n")
clf = RandomForestClassifier(criterion="gini", n_estimators=150)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
modelValidation(y_test, y_pred)

#Extra Trees Model 1
print("\n#######\ExtraTreesClassifier 1\n")
clf = ExtraTreesClassifier(n_estimators=150)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
modelValidation(y_test, y_pred)

#Extra Trees Model 2
print("\n#######\ExtraTreesClassifier 2\n")
clf = ExtraTreesClassifier(n_estimators=500)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
modelValidation(y_test, y_pred)

#SGD Model 2
print("\n#######\SGDClassifier 1\n")
clf = SGDClassifier(max_iter=1000, tol=1e-4)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
modelValidation(y_test, y_pred)


#Clustering
print("\n#######\K Means Clustering\n")
model = KMeans(n_clusters=3, max_iter=100, random_state = 42).fit(x_train)
y_pred = model.predict(x_test)
modelValidation(y_test, y_pred)


#Logistic Regression
print("\n#######\Logistic Regression\n")
clf= LogisticRegression(random_state=0).fit(x_train, y_train)
y_pred = clf.predict(x_test)
modelValidation(y_test, y_pred)

#Logistic Regression with hyperparameter optimisation
print("\n#######\Logistic Regression\n")
clf= LogisticRegression(random_state=0).fit(x_train, y_train)
y_pred = clf.predict(x_test)
modelValidation(y_test, y_pred)
