import modin.pandas as pd
from modin.config import Engine
Engine.put("dask") 
import sys
import os
import pandas as pd
import daal4py as d4p
from xgboost import XGBClassifier
import time
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')
pio.renderers.default='notebook' 
intel_pal, color=['#0071C5','#FCBB13'], ['#7AB5E1','#FCE7B2']
temp=dict(layout=go.Layout(font=dict(family="Franklin Gothic", size=12), 
                           height=500, width=1000))

stdout_backup = sys.stdout
log_file = open("debug.log", "w")
sys.stdout = log_file

# Read data
#data = pandas.read_pickle('media/data_100000.pkl')
data = pd.read_csv("dataset.csv",nrows = 1000000)
data = data.drop('Water Temperature',axis=1)
data = data.drop('Source',axis=1)
data = data.drop('Month',axis=1)
data = data.drop('Air Temperature',axis=1)
data = data.drop('Day',axis=1)
data = data.drop('Time of Day',axis=1)
data = data.drop('Index',axis=1)


pH = data['pH'].mean()
print(pH)
data['pH'].fillna(pH, inplace=True)

Iron = data['Iron'].mean()
print(Iron)
data['Iron'].fillna(Iron, inplace=True)

Nitrate = data['Nitrate'].mean()
print(Nitrate)
data['Nitrate'].fillna(Nitrate, inplace=True)

Zinc = data['Zinc'].mean()
print(Zinc)
data['Zinc'].fillna(Zinc, inplace=True)

Turbidity = data['Turbidity'].mean()
print(Turbidity)
data['Turbidity'].fillna(Turbidity, inplace=True)

Fluoride = data['Fluoride'].mean()
print(Fluoride)
data['Fluoride'].fillna(Fluoride, inplace=True)

Copper = data['Copper'].mean()
print(Copper)
data['Copper'].fillna(Copper, inplace=True)

Odor = data['Odor'].mean()
print(Odor)
data['Odor'].fillna(Odor, inplace=True)

Sulfate = data['Sulfate'].mean()
print(Sulfate)
data['Sulfate'].fillna(Sulfate, inplace=True)

Conductivity = data['Conductivity'].mean()
print(Conductivity)
data['Conductivity'].fillna(Conductivity, inplace=True)

Chlorine = data['Chlorine'].mean()
print(Chlorine)
data['Chlorine'].fillna(Chlorine, inplace=True)

Manganese = data['Manganese'].mean()
print(Manganese)
data['Manganese'].fillna(Manganese, inplace=True)

TotalDissolvedSolids = data['Total Dissolved Solids'].mean()
print(TotalDissolvedSolids)
data['Total Dissolved Solids'].fillna(TotalDissolvedSolids, inplace=True)


dic={'Colorless':1,'Near Colorless':2,'Light Yellow':3,'Yellow':4,'Faint Yellow':5}
data.replace(dic,inplace=True)
data['Color'].fillna(6, inplace=True)

data.fillna(value=0)
data.fillna(0, inplace=True)

print(data)
print("Data shape: {}\n".format(data.shape))
print(data.head())


print(data.isna().sum())
missing=data.isna().sum().sum()
duplicates=data.duplicated().sum()
print("\nThere are {:,.0f} missing values in the data.".format(missing))
print("There are {:,.0f} duplicate records in the data.".format(duplicates))


from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.svm import SVC

def prepare_train_test_data(data, target_col, test_size):
    
    """
    Function to scale and split the data into training and test sets
    """

    scaler = RobustScaler()   
    sc = StandardScaler()

    X = data.drop(target_col, axis=1)
    Y = data[target_col]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=10)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Train Shape: {}".format(X_train_scaled.shape))
    print("Test Shape: {}".format(X_test_scaled.shape))
    
    return X_train_scaled, X_test_scaled, Y_train, Y_test

## Prepare Train and Test datasets ##
print("Preparing Train and Test datasets")
X_train, X_test, Y_train, Y_test = prepare_train_test_data(data=data, 
                                                           target_col='Target', 
                                                           test_size=.2)
print("Preparing Train and Test datasets end")

def plot_model_res(model_name, y_test, y_prob):
    
    """
    Function to plot ROC/PR Curves and predicted target distribution
    """
    
    intel_pal=['#0071C5','#FCBB13']
    color=['#7AB5E1','#FCE7B2']
    
    ## ROC & PR Curve ##
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr,tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)
    
    fig = make_subplots(rows=1, cols=2, 
                        shared_yaxes=True, 
                        subplot_titles=['Receiver Operating Characteristic<br>(ROC) Curve',
                                        'Precision-Recall Curve<br>AUPRC = {:.3f}'.format(auprc)])
    
    fig.add_trace(go.Scatter(x=np.linspace(0,1,11), y=np.linspace(0,1,11), 
                             name='Baseline',mode='lines',legendgroup=1,
                             line=dict(color="Black", width=1, dash="dot")), row=1,col=1)    
    fig.add_trace(go.Scatter(x=fpr, y=tpr, line=dict(color=intel_pal[0], width=3), 
                             hovertemplate = 'True positive rate = %{y:.3f}, False positive rate = %{x:.3f}',
                             name='AUC = {:.4f}'.format(roc_auc),legendgroup=1), row=1,col=1)
    fig.add_trace(go.Scatter(x=recall, y=precision, line=dict(color=intel_pal[0], width=3), 
                             hovertemplate = 'Precision = %{y:.3f}, Recall = %{x:.3f}',
                             name='AUPRC = {:.4f}'.format(auprc),showlegend=False), row=1,col=2)
    fig.update_layout(template=temp, title="{} ROC and Precision-Recall Curves".format(model_name), 
                      hovermode="x unified", width=900,height=500,
                      xaxis1_title='False Positive Rate (1 - Specificity)',
                      yaxis1_title='True Positive Rate (Sensitivity)',
                      xaxis2_title='Recall (Sensitivity)',yaxis2_title='Precision (PPV)',
                      legend=dict(orientation='v', y=.07, x=.45, xanchor="right",
                                  bordercolor="black", borderwidth=.5))
    fig.write_image("1.png")
    
    ## Target Distribution ##     
    plot_df=pd.DataFrame.from_dict({'State 0':(len(y_prob[y_prob<=0.5])/len(y_prob))*100, 
                                    'State 1':(len(y_prob[y_prob>0.5])/len(y_prob))*100}, 
                                   orient='index', columns=['pct'])
    fig=go.Figure()
    fig.add_trace(go.Pie(labels=plot_df.index, values=plot_df.pct, hole=.45, 
                         text=plot_df.index, sort=False, showlegend=False,
                         marker=dict(colors=color,line=dict(color=intel_pal,width=2.5)),
                         hovertemplate = "%{label}: <b>%{value:.2f}%</b><extra></extra>"))
    fig.update_layout(template=temp, title='Predicted Target Distribution',width=700,height=450,
                      uniformtext_minsize=15, uniformtext_mode='hide')
    fig.write_image("2.png")

def plot_model_res1(model_name, y_test, y_prob):
    
    """
    Function to plot ROC/PR Curves and predicted target distribution
    """
    
    intel_pal=['#0071C5','#FCBB13']
    color=['#7AB5E1','#FCE7B2']
    
    ## ROC & PR Curve ##
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr,tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)
    
    fig = make_subplots(rows=1, cols=2, 
                        shared_yaxes=True, 
                        subplot_titles=['Receiver Operating Characteristic<br>(ROC) Curve',
                                        'Precision-Recall Curve<br>AUPRC = {:.3f}'.format(auprc)])
    
    fig.add_trace(go.Scatter(x=np.linspace(0,1,11), y=np.linspace(0,1,11), 
                             name='Baseline',mode='lines',legendgroup=1,
                             line=dict(color="Black", width=1, dash="dot")), row=1,col=1)    
    fig.add_trace(go.Scatter(x=fpr, y=tpr, line=dict(color=intel_pal[0], width=3), 
                             hovertemplate = 'True positive rate = %{y:.3f}, False positive rate = %{x:.3f}',
                             name='AUC = {:.4f}'.format(roc_auc),legendgroup=1), row=1,col=1)
    fig.add_trace(go.Scatter(x=recall, y=precision, line=dict(color=intel_pal[0], width=3), 
                             hovertemplate = 'Precision = %{y:.3f}, Recall = %{x:.3f}',
                             name='AUPRC = {:.4f}'.format(auprc),showlegend=False), row=1,col=2)
    fig.update_layout(template=temp, title="{} ROC and Precision-Recall Curves".format(model_name), 
                      hovermode="x unified", width=900,height=500,
                      xaxis1_title='False Positive Rate (1 - Specificity)',
                      yaxis1_title='True Positive Rate (Sensitivity)',
                      xaxis2_title='Recall (Sensitivity)',yaxis2_title='Precision (PPV)',
                      legend=dict(orientation='v', y=.07, x=.45, xanchor="right",
                                  bordercolor="black", borderwidth=.5))
    fig.write_image("3.png")
    
    ## Target Distribution ##     
    plot_df=pd.DataFrame.from_dict({'State 0':(len(y_prob[y_prob<=0.5])/len(y_prob))*100, 
                                    'State 1':(len(y_prob[y_prob>0.5])/len(y_prob))*100}, 
                                   orient='index', columns=['pct'])
    fig=go.Figure()
    fig.add_trace(go.Pie(labels=plot_df.index, values=plot_df.pct, hole=.45, 
                         text=plot_df.index, sort=False, showlegend=False,
                         marker=dict(colors=color,line=dict(color=intel_pal,width=2.5)),
                         hovertemplate = "%{label}: <b>%{value:.2f}%</b><extra></extra>"))
    fig.update_layout(template=temp, title='Predicted Target Distribution',width=700,height=450,
                      uniformtext_minsize=15, uniformtext_mode='hide')
    fig.write_image("4.png")


## Initialize SVC model ##
parameters = {
    'class_weight': 'balanced',
    'probability': True,
    'random_state': 21}
svc = SVC(**parameters)

## Tune Hyperparameters ##
strat_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=21)
print("\nTuning hyperparameters..")
grid = {
    'C': np.logspace(-1, 2, 30),
    'kernel': ['poly', 'rbf', 'sigmoid']
    }
grid_search = RandomizedSearchCV(svc, param_distributions=grid, 
                                    cv=strat_kfold, n_iter=5, scoring='roc_auc', 
                                    verbose=1, n_jobs=-1, random_state=21)
print(Y_train)

grid_search.fit(X_train, Y_train)
    
print("Done!\nBest hyperparameters:", grid_search.best_params_)
print("Best cross-validation AUC: {:.4f}".format(grid_search.best_score_))

svc = grid_search.best_estimator_
svc_prob = svc.predict_proba(X_test)[:,1]
svc_pred = pd.Series(svc.predict(X_test), name='Target')
svc_auc = roc_auc_score(Y_test, svc_prob)
svc_f1 = f1_score(Y_test, svc_pred)  
   
## Print model results ##
print("\nTest F1 accuracy: {:.2f}%, AUC: {:.5f}".format(svc_f1*100,svc_auc))
plot_model_res(model_name='SVC', y_test=Y_test, y_prob=svc_prob)


## Initialize XGBoost model ##
ratio = float(np.sum(Y_train == 0)) / np.sum(Y_train == 1)
parameters = {'scale_pos_weight': ratio.round(2), 
                'tree_method': 'hist',
                'random_state': 21}
xgb_model = XGBClassifier(**parameters)

## Tune hyperparameters ##
strat_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=21)
print("\nTuning hyperparameters..")
grid = {'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'max_depth': [3, 4, 5],
        }

grid_search = GridSearchCV(xgb_model, param_grid=grid, 
                            cv=strat_kfold, scoring='roc_auc', 
                            verbose=1, n_jobs=-1)
grid_search.fit(X_train, Y_train)

print("Done!\nBest hyperparameters:", grid_search.best_params_)
print("Best cross-validation AUC: {:.4f}".format(grid_search.best_score_))
    
## Convert XGB model to daal4py ##
xgb = grid_search.best_estimator_
daal_model = d4p.get_gbt_model_from_xgboost(xgb.get_booster())

## Calculate predictions ##
daal_prob = d4p.gbt_classification_prediction(nClasses=2,
    resultsToEvaluate="computeClassLabels|computeClassProbabilities",
    fptype='float').compute(X_test, daal_model).probabilities # or .predictions
xgb_pred = pd.Series(np.where(daal_prob[:,1]>.5, 1, 0), name='Target')
xgb_auc = roc_auc_score(Y_test, daal_prob[:,1])
xgb_f1 = f1_score(Y_test, xgb_pred)  
    
## Plot model results ##
print("\nTest F1 Accuracy: {:.2f}%, AUC: {:.5f}".format(xgb_f1*100, xgb_auc)) 
plot_model_res1(model_name='XGBoost', y_test=Y_test, y_prob=daal_prob[:,1])

log_file.close()
# restore the output to initial pattern
sys.stdout = stdout_backup

