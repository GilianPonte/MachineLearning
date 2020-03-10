
import sklearn
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn import tree
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.externals.six import StringIO  
import pydot 


# In[13]:


df = load_breast_cancer()
df = pd.DataFrame(np.c_[df['data'], df['target']],
                  columns= np.append(df['feature_names'], ['target']))
for col in df.columns: 
    print(col) 

print(df.head())
total_rows=len(df.axes[0])
print(total_rows)


# Outlier detection and visualization

# In[3]:


histograms = df.hist()
df.hist("target")


# In[2]:


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size = .2)


# In[3]:


#PCA with scikit learn
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train_pca = pca = PCA().fit(X_train)
X_test_pca = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


# In[4]:


plot = 1

# plot explained variance
if plot == 1:
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)') #for each component
    plt.title('Breast Cancer data set Explained Variance')
    plt.savefig('foo.png')
    plt.show()


# In[5]:


print(np.cumsum(pca.explained_variance_ratio_))


# Selecting the amount of principle components

# In[6]:


# 10 features 
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)


# In[7]:


# baseline linear model
reg = LogisticRegression(random_state=0).fit(X_train, y_train)
prediction = reg.predict(X_test)
score = reg.score(X_test,y_test)
print(score)

reg_pca = LogisticRegression(random_state=0).fit(X_train_pca, y_train)
score_pca = reg_pca.score(X_test_pca,y_test)
print(score_pca)


# In[8]:


LPM = linear_model.LinearRegression()
LPM = LPM.fit(X_train, y_train)
LPM.coef_
predictionLPM = LPM.predict(X_test)
scoreLPM = LPM.score(X_test, y_test)
print(scoreLPM)

LPMpca = linear_model.LinearRegression()
LPMpca = LPMpca.fit(X_train_pca, y_train)
LPMpca.coef_
predictionLPM = LPMpca.predict(X_test_pca)
scoreLPMpca = LPMpca.score(X_test_pca, y_test)
print(scoreLPMpca)


# In[9]:


#baseline decicision tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
tree.export_graphviz(clf, out_file='tree.dot')  
dot_data = StringIO() 
tree.export_graphviz(clf, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph[0].write_pdf("decisiontree.pdf")

predictionBaseline = clf.predict(X_test)
scoreclf = clf.score(X_test, y_test)
#print(classification_report(y_test,predictionBaseline,target_names=['malignant', 'benign']))
print(scoreclf)


#baseline decicision tree
clfPca = tree.DecisionTreeClassifier()
clfPca = clfPca.fit(X_train_pca, y_train)
tree.export_graphviz(clfPca, out_file='treepca.dot')   
dot_data = StringIO() 
tree.export_graphviz(clfPca, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph[0].write_pdf("decisiontreepca.pdf")

predictionBaselinePca = clfPca.predict(X_test_pca)
scoreclf = clfPca.score(X_test_pca, y_test)
#print(classification_report(y_test,predictionBaselinePca,target_names=['malignant', 'benign']))
print(scoreclf)


# In[18]:


# KNN classifier on original data 
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)
score = knn.score(X_test,y_test)
print(score)

knn.fit(X_train_pca, y_train)
score_pca = knn.score(X_test_pca,y_test)
print(score_pca)


# In[14]:


# Decision tree with Gridsearch
clf = tree.DecisionTreeClassifier()

#create a dictionary of all values we want to test for n_neighbors
param_grid = {'max_depth': np.arange(1, 50)}

#use gridsearch to test all values for n_neighbors
clf_gscv = GridSearchCV(clf, param_grid, cv=10)

#fit model to data
clf_gscv.fit(X_train_pca, y_train)

#check top performing n_neighbors value
print(clf_gscv.best_params_)

#check mean score for the top performing value of n_neighbors
print(clf_gscv.best_score_)


# In[15]:


#KNN with PCA or without PCA and Gridsearch
knn2 = KNeighborsClassifier()

#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 50)}

#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)

#fit model to data
knn_gscv.fit(X_train_pca, y_train)

#check top performing n_neighbors value
print(knn_gscv.best_params_)

#check mean score for the top performing value of n_neighbors
print(knn_gscv.best_score_)


# In[32]:


## Plot results from gridsearches
def plot_cv_results(cv_results, param_x, metric='mean_test_score'):
    """
    cv_results - cv_results_ attribute of a GridSearchCV instance (or similar)
    param_x - name of grid search parameter to plot on x axis
    param_z - name of grid search parameter to plot by line color
    """
    cv_results = pd.DataFrame(cv_results)
    col_x = 'param_' + param_x
    fig, ax = plt.subplots(1, 1, figsize=(11, 8))
    sns.pointplot(x= col_x, y=metric, data=cv_results, ci=95, ax = ax)
    ax.set_title("CV Grid Search Results")
    ax.set_xlabel(param_x)
    ax.set_ylabel(metric)
    return fig


# In[34]:


# Single function to make plot for each Gridsearch

fig = plot_cv_results(knn_gscv.cv_results_, 'n_neighbors')


# In[59]:


#10 fold cross validation with PCA applied
k_fold = KFold(10)
X_pca = pca.fit_transform(X)
classifiers = []

for k, (train, test) in enumerate(k_fold.split(X_pca, y)):
    clfk = tree.DecisionTreeClassifier()
    clfk = clfk.fit(X_pca[train], y[train])
    predictionBaseline = clfk.predict(X_pca[test])
    print ("Classification report for %d fold", k)
    print(classification_report(y[test],predictionBaseline,target_names=['malignant', 'benign']))
    classifiers.append(clfk)
votes = []


# In[60]:


# Construct ensemble based on majority vote
for classifier in classifiers:
    classifier.fit(X_train_pca,y_train)
    votes.append(classifier.predict(X_test_pca))

ensembleVotes = np.zeros((len(y_test),1), dtype=int)
predictionEnsemble = np.zeros((len(y_test),1), dtype=int)

for prediction in votes:
    for idx in range(0,len(prediction)):
        ensembleVotes[idx]+= prediction[idx]

for idx in range(0,len(prediction)):
    if ensembleVotes[idx] > 5:
        predictionEnsemble[idx] = 1
print("ensemble")
print(classification_report(y_test,predictionEnsemble,target_names=['malignant', 'benign']))


# In[ ]:


## Regularization


# In[15]:


# Ridge regression
param_grid = {'alpha': np.arange(start=0, stop=100, step=10)}
regridge = linear_model.Ridge()

#use gridsearch to test all values for n_neighbors
reg_gscv = GridSearchCV(regridge, param_grid, cv=10, return_train_score = True)

reg_gscv.fit(X_train_pca, y_train)

def plot_cv_results(cv_results, param_x, metric='mean_test_score'):
    """
    cv_results - cv_results_ attribute of a GridSearchCV instance (or similar)
    param_x - name of grid search parameter to plot on x axis
    param_z - name of grid search parameter to plot by line color
    """
    cv_results = pd.DataFrame(cv_results)
    col_x = 'param_' + param_x
    fig, ax = plt.subplots(1, 1, figsize=(11, 8))
    sns.pointplot(x= col_x, y=metric, data=cv_results, ci=95, ax = ax)
    ax.set_title("CV Grid Search Results")
    ax.set_xlabel(param_x)
    ax.set_ylabel(metric)
    return fig

fig = plot_cv_results(reg_gscv.cv_results_, 'alpha')


# In[19]:


# Logistic regression

logitl2 = linear_model.LogisticRegression(penalty='l2', C = 1.0)

param_grid = {'C': np.arange(.1, .9, step = .1)}

reg_gscv = GridSearchCV(logitl2 , param_grid, cv=10, return_train_score = True)

reg_gscv.fit(X_train, y_train)


def plot_cv_results(cv_results, param_x, metric='mean_test_score'):
    """
    cv_results - cv_results_ attribute of a GridSearchCV instance (or similar)
    param_x - name of grid search parameter to plot on x axis
    param_z - name of grid search parameter to plot by line color
    """
    cv_results = pd.DataFrame(cv_results)
    col_x = 'param_' + param_x
    fig, ax = plt.subplots(1, 1, figsize=(11, 8))
    sns.pointplot(x=col_x , y=metric, data=cv_results, ci=95, ax = ax)
    ax.set_title("CV Grid Search Results")
    ax.set_xlabel(param_x)
    ax.set_ylabel(metric)
    return fig

fig = plot_cv_results(reg_gscv.cv_results_, 'C')
print (reg_gscv.best_score_, reg_gscv.best_params_) 


# In[17]:


## decision tree regularization

parameters = {'max_depth':range(1,40)}
clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters, n_jobs=4)
clf.fit(X_train_pca, y_train)
tree_model = clf.best_estimator_
print (clf.best_score_, clf.best_params_) 
