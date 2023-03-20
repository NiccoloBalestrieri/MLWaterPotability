# MLWaterPotability

> This project will present an analysis of an end-to-end project related to a dataset concerning information on drinking water potability.
These considerations, supported by specific and precise parameters, could be useful.
The entire source code is written in **Python**.
You can find this water potability dataset at **https://www.kaggle.com/datasets/adityakadiwal/water-potability**

<p align="center">
<img src="https://aiworldschool.com/wp-content/uploads/2021/01/AI-for-Water-1.png" align="center" width="500">
</p>


## Table of contents
- [Water Potability with Machine Learning](#water-potability-with-machine-learning)
  - [Table of contents](#table-of-contents)
  - :zap:[Quick Start](#quick-start-)
    - [NumPy](https://numpy.org/)
    - [Pandas](https://pandas.pydata.org/)
    - [Matplotlib](https://matplotlib.org)
    - [Shipy](https://docs.scipy.org)
    - [Sklearn](https://scikit-learn.org/stable/)
  - :eyes:[Data exploration](#data-exploration-)
     - [Data visualization](#data-visualization)
     - [Dataset splitting](#dataset-splitting)
  - :hammer_and_wrench:[Preprocessing](#preprocessing-)
     - [Feature selection](#feature-selection)
       - [Mutual information](#mutual-information)
       - [Chi2](#chi2)
     - [Feature scaling](#feature-scaling)
  - :question:[Models comparison](#models-comparison-)
  - :books:[Fine tuning](#fine-tuning-)
  - :white_check_mark:[Evaluation](#evaluation-)
  - :heavy_plus_sign:[Extras](#extras-)

---


## Quick start ⚡

> Libraries that you need to install and import to build this software.
- NumPy
```python
import numpy as np
```
- Pandas 
```python
import pandas as pd
```
- Matplotlib
```python
import matplotlib.pyplot as plt
```
- Shipy
```python
from scipy import stats
```
- Sklearn 
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.manifold import TSNE
```

You can install directly all the libraries with requirements.txt file.
```
pip install -r /path/to/requirements.txt
```

## Data exploration 👀
> Now let's analyze the characteristics of the features.

### Data visualization
The graphs for each feature present in the dataset will be shown below: categorical variables are represented by bar graphs (in yellow), while numeric variables are represented by histograms (in green).
```python
print("\nPlot graph for all the feature\n")
    #range create a sequence of number with the number of columns
    for i in range(len(water_potability.columns)):
        if water_potability.columns[i] in ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']:
            #plot the graph numerical variables
            plt.hist(water_potability[water_potability.columns[i]], facecolor='g')
            plt.xlabel(water_potability.columns[i])
            plt.ylabel("Number of occurences")
            plt.show()
        else:
           #plot the graph for categorical variables
           water_potability[water_potability.columns[i]].value_counts().sort_index().plot(kind = "bar", facecolor = 'yellow')
           plt.xlabel(water_potability.columns[i])
           plt.ylabel("Number of occurences")
           plt.show()

    #plot the percentages of the target variable
    print("\nPlot the percentages of the target variable\n")
    labels = ['Non-Potable', 'Potable']
    data = [water_potability['Potability'].value_counts()[0], water_potability['Potability'].value_counts()[1]]
    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels = labels, explode = [0, 0.1], autopct='%1.1f%%', shadow=True, colors = "green")
    plt.title("Water Potability", fontsize=20)
    plt.show()
```

The percentages of the target variable were represented with a pie chart:
<p align="center">
<img src="https://user-images.githubusercontent.com/92525345/226470255-41f120ce-9b74-452b-8bdf-2adf712391a3.png" alt="alt text" width="500"/>

### Dataset splitting
In this phase, the starting dataset will be partitioned into an 80% training-set and a 20% test-set. The training-set has dimension (2620, 6), the test-set has dimension (656, 6) and they were created using the keyword stratify so that the latter has the same number of examples for both a class and for the other.
For the validation-set, one could opt for a solution without K-Fold Cross-validation. However, by carrying out several tests, it emerged that the results are slightly fluctuating depending on the portion of the validation-set selected during the training phase. I decided to extract a 20% from training set which is essential for calculating the metric of accuracy.
```python
 print("\nSplit the dataset \n")
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
 print("Training-test shape: " + str(X_train.shape) + " and " + str(y_train.shape))
 print("Test-test shape: " + str(X_test.shape) + " and " + str(y_test.shape))
```


## Preprocessing 🛠️
### Manage NaN value
```python
print("\nHeatmap of the NaN value\n")
sns.heatmap(water_potability.isna(), cmap="Reds_r")
plt.show()
```
A heatmap is plotted showing the position of the NaNs value. This view is very convenient to see if there are any NaN distributed in particular areas of the dataset or not.

<p align="center">
<img src="https://user-images.githubusercontent.com/92525345/226472091-65017545-f656-431a-8b09-10c45268db3b.png" alt="alt text" width="500"/>

As it possible to see some features contain missing values, for this reason it's necessary to fill or remove these values. Features with missing values are 3: ph, sulfate and trihalomethanes.
So I decided to fill these values with the average, since I have dealing with numeric variables. I also made this decision because I thought it was inappropriate to go and eliminate the missing values, because I would have deleted about 39% of the dataset and would then have gotten gods later not very satisfactory results.
```python
#fill the NaN value of the feature of the dataset
water_potability['ph'].fillna(value=water_potability['ph'].mean(),inplace=True)
water_potability['Sulfate'].fillna(value=water_potability['Sulfate'].mean(),inplace=True)
water_potability['Trihalomethanes'].fillna(value=water_potability['Trihalomethanes'].mean(),inplace=True)
```
### Feature selection
The dataset, therefore, contains some values that turn out to be useless, for this reason the feature selection is performed. With it I'm going to eliminate those values ​​that are of little significance for the task, in particular going to use the mutual information.
```python
y = water_potability.pop('Potability') #pop is used to cancel the target columns
X = water_potability
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=mutual_info_classif, k="all")
fit = bestfeatures.fit(X,y)
wpscores = pd.DataFrame(fit.scores_)
wpcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([wpcolumns,wpscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores)
```
Below I show what are the scores, obtained by the mutual information, thanks to the use of the "SelectKBest":

<p align="center">
<img src="https://user-images.githubusercontent.com/92525345/226472850-6e67fc83-34d1-4dc4-b769-834f671767a1.png" alt="alt text" width="500"/>

It can be seen from the "Score" values that there are not very good examples for this reason I select as important features all those that have a "Score" other than 0.

<p align="center">
<img src="https://user-images.githubusercontent.com/92525345/226473175-c7034d5b-ea1d-4f01-8627-2c1bc23c4246.png" alt="alt text" width="500"/>


### Feature scaling
After the feature selection there are on average 6 features, such features differ in order of magnitude from each other and it was chosen to scale using z-score: it allows you to scale the features so that they have zero mean and unit variance. The z-score is defined as:
𝑥 ′ = (𝑥 − 𝑥 )/𝜎 𝑐𝑜𝑛 𝑥 = ara𝑣𝑔(𝑥).

## Models comparison ❓
> In this section we will analyze and choose the best classification model (with default parameters) on which, subsequently, we will perform fine-tuning of the parameters.
Specifically, to maintain the proportion of 80% training-set and 20% validation-set, the n_splits parameter of the KFold Cross-validation is set to 5 (the same number that would be obtained by performing 20 % on the training-set).

```python
kf = KFold(n_splits = 5, random_state = None, shuffle = False)
logistic_regression = LogisticRegression()
frst = RandomForestClassifier()
ada = AdaBoostClassifier()
grad = GradientBoostingClassifier()
tree = DecisionTreeClassifier()
xgboost = XGBClassifier()

precision_log_reg, precision_frst, precision_ada, precision_grad, precision_tree, precision_xg = 0, 0, 0, 0, 0, 0
recall_log_reg, recall_frst, recall_ada, recall_grad, recall_tree, recall_xg = 0, 0, 0, 0, 0, 0
f1_log_reg, f1_frst, f1_ada, f1_grad, f1_tree, f1_xg = 0, 0, 0, 0, 0, 0

for train_index, validation_index in kf.split(X_train, y_train):
    xt = X_train.iloc[train_index]
    xv = X_train.iloc[validation_index]
    yt = y_train.iloc[train_index]
    yv = y_train.iloc[validation_index]

    #Build a forest of trees from the training set (X, y)
    logistic_regression.fit(xt, yt)
    frst.fit(xt, yt)
    ada.fit(xt, yt)
    grad.fit(xt, yt)
    tree.fit(xt, yt)
    xgboost.fit(xt, yt)

    #Predict using the linear model
    y_pred_reg = logistic_regression.predict(xv)
    y_pred_frst = frst.predict(xv)
    y_pred_ada = ada.predict(xv)
    y_pred_grad = grad.predict(xv)
    y_pred_tree = tree.predict(xv)
    y_pred_xg = xgboost.predict(xv)

    #calculating precision, recall and f1-score

    precision_log_reg += precision_score(yv, y_pred_reg, average = 'micro')
    precision_frst += precision_score(yv, y_pred_frst, average = 'micro')
    precision_ada += precision_score(yv, y_pred_ada, average = 'micro')
    precision_grad += precision_score(yv, y_pred_grad, average = 'micro')
    precision_tree += precision_score(yv, y_pred_tree, average = 'micro')
    precision_xg += precision_score(yv, y_pred_xg, average = 'micro')

    recall_log_reg += recall_score(yv, y_pred_reg, average = 'micro')
    recall_frst += recall_score(yv, y_pred_frst, average = 'micro')
    recall_ada += recall_score(yv, y_pred_ada, average = 'micro')
    recall_grad += recall_score(yv, y_pred_grad, average = 'micro')
    recall_tree += recall_score(yv, y_pred_tree, average = 'micro')
    recall_xg += recall_score(yv, y_pred_xg, average = 'micro')

    f1_log_reg += f1_score(yv, y_pred_reg, average = 'micro')
    f1_frst += f1_score(yv, y_pred_frst, average = 'micro')
    f1_ada += f1_score(yv, y_pred_ada, average = 'micro')
    f1_grad += f1_score(yv, y_pred_grad, average = 'micro')
    f1_tree += f1_score(yv, y_pred_tree, average = 'micro')
    f1_xg += f1_score(yv, y_pred_xg, average = 'micro')

print('Precision logistic regression: ' + str(precision_log_reg / 5))
print('Recall logistic regression: ' + str(recall_log_reg / 5))
print('F1 score logistic regression: ' + str(f1_log_reg / 5) + "\n")

print('Precision random forest: ' + str(precision_frst / 5))
print('Recall random forest: ' + str(recall_frst / 5))
print('F1 score random forest: ' + str(f1_frst / 5) + "\n")

print('Precision adaboost: ' + str(precision_ada / 5))
print('Recall adaboost: ' + str(recall_ada / 5))
print('F1 score adaboost: ' + str(f1_ada / 5) + "\n")

print('Precision gradient boosting: ' + str(precision_grad / 5))
print('Recall gradient boosting: ' + str(recall_grad / 5))
print('F1 score gradient boosting: ' + str(f1_grad / 5) + "\n")

print('Precision decision tree: ' + str(precision_tree / 5))
print('Recall decision tree: ' + str(recall_tree / 5))
print('F1 score decision tree: ' + str(f1_tree / 5) + "\n")

print('Precision xgboosting: ' + str(precision_xg / 5))
print('Recall xgboosting: ' + str(recall_xg / 5))
print('F1 score xgboosting: ' + str(f1_xg / 5) + "\n")
```
The algorithms used are:
- Logistic Regression
- Decision Tree
- Random Forest
- AdaBoost
- GradientBoosting

By making several attempts, going to calculate precision, recall and f1 score, I immediately notice that the data is not of quality because I have, however pertains to logistic regression, all values ​​equal to 0 and for the other models I have extremely low results.
For this reason, to improve these results, I use average which is a parameter of the methods of precision, recall and f1 score and I set it equal to micro (average = "micro").
I considered using that parameter type because it is ideal if there is a suspicion that there may be a class imbalance (i.e. yes may have many more examples of one class than of other classes).
The results are the following:
<p align="center">
<img src="https://user-images.githubusercontent.com/92525345/226474868-5ac91bef-a4e3-4ee9-9d6f-7ecb28f069d4.png" alt="alt text" width="300"/>


## Fine tuning 📚
> At this point, the model on which to carry out fine-tuning operations is Random Forest.
Given that the dataset has modest dimension, it is preferable to combine Grid Search and Random Search together: initially a Random Search was carried out on the number of trees (n_estimators), displaying the training-set and validation-set graph to establish a possible overfitting or underfitting. The research was performed always performing a KFold Cross Validation.
```python
kf = KFold(n_splits = 5, random_state = None, shuffle = False)
print("\nRandom Search\n")
train_results, validation_results = [], []     
prec_val, prec_train = 0, 0
n_estimators_set = [5, 20, 50, 100]
for i in n_estimators_set:
    prec_val, prec_train = 0, 0
    optimal_random_forest = RandomForestClassifier(n_estimators = i)#numero di alberi nella random forest
    for train_index, validation_index in kf.split(X_train, y_train):
        xt = X_train.iloc[train_index]
        xv = X_train.iloc[validation_index]
        yt = y_train.iloc[train_index]
        yv = y_train.iloc[validation_index] #ho utilizzato iloc anche qui in quanto passando come parametro y_train e non y, non ho una sola colonna ma gli indici e il valore corrispondente
        optimal_random_forest.fit(xt, yt)
        y_pred_vt_rf_rnd = optimal_random_forest.predict(xv)
        y_pred_xt_rf_rnd = optimal_random_forest.predict(xt)
        prec_val += precision_score(yv, y_pred_vt_rf_rnd, average = 'micro')
        prec_train += precision_score(yt, y_pred_xt_rf_rnd, average = 'micro')

    train_results.append(prec_train / 5)
    validation_results.append(prec_val / 5)
    print("Computing " +  str(i) + " with validation precision: " + str((prec_val / 5) * 100))
```
Trying with [10, 100, 500, 1000] I obtain the following results:
<p align="center">
<img src="https://user-images.githubusercontent.com/92525345/226475463-3f12249b-a6a3-47d9-9144-3dc69930aaa5.png" width="500"/>

The following graph shows that the best situation is when then_estimators are roughly equal to 100, as the precision value of the validation set is the highest.

<p align="center">
<img src="https://user-images.githubusercontent.com/92525345/226475821-87d2412a-0aef-45c8-8ab5-f0a22b6242ad.png" alt="alt text" width="500"/>

At this point a Grid-Search is performed (again with KFold Cross Validation) both to identify the best number of estimators and to find the most correct regularization parameters. 

```python
n_trees = 100
val_range = range(n_trees - 50 , n_trees + 50, 15)
best_score = {}
lista = []
count = 0
kf = KFold(n_splits = 5, random_state = None, shuffle = False)
for i in val_range:
    for j in range(1, 6, 1):
        best_random_forest = RandomForestClassifier(n_estimators = i, max_depth = j)
        for train_index, validation_index in kf.split(X_train, y_train):
            xt = X_train.iloc[train_index]
            xv = X_train.iloc[validation_index]
            yt = y_train.iloc[train_index]
            yv = y_train.iloc[validation_index]
            best_random_forest.fit(xt, yt)
            final_random_forest = best_random_forest.predict(xv)
            lista.append(i)
            lista.append(j)
            lista.append(precision_score(yv, final_random_forest, average = 'micro') * 100)
            best_score[count] = lista
            count += 1
            lista = []
m = 0
md = 0
ne = 0
for i in range(len(best_score.values())):
    print(list(best_score.values())[i][2])
    if list(best_score.values())[i][2] >= m:
        m = list(best_score.values())[i][2]
        md = list(best_score.values())[i][1]
        ne = list(best_score.values())[i][0]

print("max precision score: " + str(m) + "max_depth: " + str(md) + "n_estimators: " + str(ne))
```
  
The number of estimators and the most correct regularization parameters are difficult to find because these values change frequently, so with this:

```python
m = 0
md = 0
ne = 0
for i in range(len(best_score.values())):
    print(list(best_score.values())[i][2])
    if list(best_score.values())[i][2] >= m:
        m = list(best_score.values())[i][2]
        md = list(best_score.values())[i][1]
        ne = list(best_score.values())[i][0]

print("max precision score: " + str(m) + "max_depth: " + str(md) + "n_estimators: " + str(ne))
```

The best precision score and its related data are always selected max_depth and the n_estimators.

## Evaluation ✅
> Now it is possible to train with the best parameters previously obtained and evaluate the final result no longer on the validation-set, but on the test-set. In addition to accuracy, a confusion matrix is presented.
```python
best_n_estimators, best_max_depth = ne, md 
print(best_n_estimators, best_max_depth)
random_forest = RandomForestClassifier(n_estimators = best_n_estimators, max_depth = best_max_depth)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
print("Final precision test set: " + str(precision_score(y_test, y_pred, average = 'micro') * 100))
conf_matrix_view = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels = ["positive", "negative"])
conf_matrix_view.plot()
conf_matrix_view.ax_.set(title = "Confusion matrix for water potability", xlabel = "Predicted", ylabel = "Real class")
plt.show()
```

<p align="center">
<img src="https://user-images.githubusercontent.com/92525345/226476808-3f31021d-fa22-4c74-a4f5-7829e8b3e268.png" alt="alt text" width="500"/>

 
## Extras ➕
> After the feature selection operation, the selected features became 6. Therefore a dimensionality reduction technique was applied to roughly understand how many examples say that water results be drinkable and how many not: t-SNE.
To do this, I use the dataset in its entirety as it has dimensions modest and does not create computational problems. The proposed visualization is in both 2D and 3D.
```python
#tsne 3D
ax = plt.axes(projection = "3d")
labels = ["Negative", "Positive"]
tsne = TSNE(n_components=3, perplexity = 5, init = "pca") #Dimension we want to reduce our multi-variate data
dim_result = tsne.fit_transform(X) #data should be a list of vectors
X_0 = dim_result[:,0]
X_1 = dim_result[:,1]
X_2 = dim_result[:,2]
scatter = ax.scatter(X_0, X_1, X_2, c = y)
handles, _ = scatter.legend_elements(prop = "colors")
ax.legend(handles, labels)
plt.show()

#tsne 2D
labels = ["Negative", "Positive"]
tsne = TSNE(n_components = 2, perplexity = 5, init = "pca") 
dim_result = tsne.fit_transform(X)
X_0 = dim_result[:,0]
X_1 = dim_result[:,1]
scatter = plt.scatter(X_0, X_1, c = y)
handles, _ = scatter.legend_elements(prop = "colors")
plt.legend(handles, labels)
plt.show()
```
<p align="center">
<img src="https://user-images.githubusercontent.com/92525345/226477625-67202f48-5f1d-4189-9317-ad267939905a.png" alt="alt text" width="500"/>
<img src="https://user-images.githubusercontent.com/92525345/226477634-32ad220d-4f82-4b08-9f56-0be16f456632.png" alt="alt text" width="300"/>

