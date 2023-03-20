import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.manifold import TSNE
# pylint: disable=E1101

def main():

    #read dataset
    water_potability = pd.read_csv("water_potability.csv")
    #iniziamo la fase di data exploration
    print("\nDataset Information\n")
    print(water_potability.head(8))
    water_potability.info()
    
    #plot the graph
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

    #Count NaN value
    print("\nDataset NaN value\n")
    print(water_potability.isnull().sum())

    #Percentuale NaN value
    print("\nDataset NaN value %\n")
    print((water_potability.isnull().sum() / 3276) * 100)
    
    #Let's plot a heatmap showing the position of the nan. This visualization is very convenient to see if there are nan distributed in particular areas of the dataset or not
    print("\nHeatmap of the NaN value\n")
    sns.heatmap(water_potability.isna(), cmap="Reds_r")
    plt.show()

    #fill the NaN value of the feature of the dataset
    water_potability['ph'].fillna(value=water_potability['ph'].mean(),inplace=True)
    water_potability['Sulfate'].fillna(value=water_potability['Sulfate'].mean(),inplace=True)
    water_potability['Trihalomethanes'].fillna(value=water_potability['Trihalomethanes'].mean(),inplace=True)
    
    #data information
    print("\nNew Dataset Information\n")
    print(water_potability.head(8))
    water_potability.info()

    #feature selection
    print("\nFeauture Selection \n")
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

    #new dataset with important feature
    print("\nNew Dataset\n")
    theshold = 0.0
    xvalue = featureScores[featureScores.Score != theshold]
    print(xvalue)
    for i in X.columns:
        if i not in list(xvalue.Specs):
            X.pop(i)
    
    print("\nSplit the dataset \n")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
    print("Training-test shape: " + str(X_train.shape) + " and " + str(y_train.shape))
    print("Test-test shape: " + str(X_test.shape) + " and " + str(y_test.shape))
    
    print("\nFeauture Scaling\n")
    scaler = StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))
    print(X_train)

    #ML model comparison
    print("\nModel comparisation\n")
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

    #random serach
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

    plt.plot(n_estimators_set, train_results, color = "blue", label = "Training")
    plt.plot(n_estimators_set, validation_results, color = "red", label = "Validation")
    plt.scatter(n_estimators_set, validation_results, s = 40, facecolors = "none", edgecolors = "r")
    plt.scatter(n_estimators_set, train_results, s = 40, facecolors = "none", edgecolors = "b")
    plt.legend()
    plt.show()

    #grid search
    print("\nGRID SEARCH: ")
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

    print("\nEvaluation on test-set\n")
    best_n_estimators, best_max_depth = ne, md #scelgo 95 perchè rispetto a 1340 è piu piccolo cosi impiego meno tempo
    print(best_n_estimators, best_max_depth)
    random_forest = RandomForestClassifier(n_estimators = best_n_estimators, max_depth = best_max_depth)
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    print("Final precision test set: " + str(precision_score(y_test, y_pred, average = 'micro') * 100))
    conf_matrix_view = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels = ["positive", "negative"])
    conf_matrix_view.plot()
    conf_matrix_view.ax_.set(title = "Confusion matrix for water potability", xlabel = "Predicted", ylabel = "Real class")
    plt.show()
    
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

main()