import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from keras.layers import Input,Dense, Dropout
from keras.models import Model,Sequential
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score,adjusted_rand_score,v_measure_score, completeness_score, homogeneity_score, silhouette_score,roc_curve, auc, f1_score, precision_recall_curve, precision_score,recall_score,accuracy_score,confusion_matrix


def results_df(X_true,X_pred,y_test,y_rank):
    X_true = np.array(X_true)
    X_pred = np.array(X_pred)
    y_test = np.array(y_test)
    y_rank = np.array(y_rank)

    residual = X_pred - X_true
    residual_avg = np.abs(np.mean(residual,axis=1))
    df = pd.DataFrame(data = {'residual_avg':residual_avg,
                              'y':y_test,
                              'y_rank':y_rank,
                             })
    return df

def simple_ann_autoencoder_regression(param_cols,layer1,layer2,layer3):
    model = Sequential()
    model.add(Dense(units = layer1,activation='relu',input_shape = (len(param_cols),)))
    model.add(Dense(units = layer2,activation='relu'))
    model.add(Dense(units = layer3,activation='relu'))
    model.add(Dense(units = len(param_cols),activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

def plot_roc(y_true,y_pred):
    fpr, tpr, thresholds = roc_curve(y_true,y_pred)
    roc_auc = auc(fpr,tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for reconstruction error classification')
    plt.legend(loc="lower right")
    plt.show()
    
def preprocess_data(df,param_cols,y_col, non_scale_cols,early_step,test_size=.1,timeseries = True):
    #add early intervals 
    
    if timeseries:
        df['y_early'] = 0
        df['y_rank'] = 0
        failure_count = 1
        for i in range(df.shape[0]):
            if df.iloc[i]['y'] == 1:
                df.loc[i-early_step:i,'y_early'] = 1
                df.loc[i-early_step:i,'y_rank'] = np.arange(early_step+1,0,-1)
                failure_count += 1
    else:
        df['y_rank'] = df[y_col]

    sc = StandardScaler()
    df_scaled = pd.DataFrame(data = sc.fit_transform(df[param_cols]),columns=param_cols)
    df_scaled = pd.concat([df[non_scale_cols],df_scaled], ignore_index=True,axis=1)
    
    X = df_scaled[param_cols]
    y = df_scaled[y_col]
    
    X_nominal = df_scaled[df_scaled[y_col] == 0][param_cols]
    X_event = df_scaled[df_scaled[y_col] > 0][param_cols]

    y_nominal = df[df[y_col] == 0][y_col]
    y_event = df[df[y_col] > 0][y_col]

    y_rank_nominal = df[df[y_col] == 0]['y_rank']
    y_rank_event = df[df[y_col] > 0]['y_rank']

    X_train, X_test, y_train, y_test,y_rank_train, y_rank_test = train_test_split(X_nominal, y_nominal, y_rank_nominal, test_size=test_size, random_state=0)

    X_test = X_test.append(X_event)
    y_test = y_test.append(y_event)
    y_rank_test = y_rank_test.append(y_rank_event)
    
    return X_train, X_test, y_train, y_test,y_rank_train, y_rank_test


def preprocess_data_cv(df,param_cols,y_col, non_scale_cols,early_step, k = 10):
    #add early intervals

    early_step = early_step
    df['y_early'] = 0
    df['y_rank'] = 0

    failure_count = 1
    for i in range(df.shape[0]):
        if df.iloc[i]['y'] == 1:
            df.loc[i-early_step:i,'y_early'] = 1
            df.loc[i-early_step:i,'y_rank'] = np.arange(early_step+1,0,-1)
            failure_count += 1

    sc = StandardScaler()
    df_scaled = pd.DataFrame(data = sc.fit_transform(df[param_cols]),columns=param_cols)
    df_scaled = pd.concat([df[non_scale_cols],df_scaled],axis=1)

    X = df_scaled[param_cols]
    y = df_scaled[y_col]
    
    X_nominal = df_scaled[df_scaled[y_col] == 0][param_cols]
    X_event = df_scaled[df_scaled[y_col] > 0][param_cols]

    y_nominal = df_scaled[df_scaled[y_col] == 0][y_col]
    y_event = df_scaled[df_scaled[y_col] > 0][y_col]

    y_rank_nominal = df_scaled[df_scaled[y_col] == 0]['y_rank']
    y_rank_event = df_scaled[df_scaled[y_col] > 0]['y_rank']
    
    X_train_list = []
    y_train_list = []

    kf=KFold(n_splits=k,shuffle=True)

    X_train_list =[]
    y_train_list = []
    y_rank_train_list = []
    X_test_list = []
    y_test_list = []
    y_rank_test_list = []

    for train_index,test_index in kf.split(X_nominal):
        X_train, y_train, y_rank_train = X_nominal.iloc[train_index], y_nominal.iloc[train_index], y_rank_nominal.iloc[train_index]
        X_test, y_test, y_rank_test = X_nominal.iloc[test_index], y_nominal.iloc[test_index], y_rank_nominal.iloc[test_index]

        X_test = X_test.append(X_event)
        y_test = y_test.append(y_event)
        y_rank_test = y_rank_test.append(y_rank_event)
        
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        y_rank_train_list.append(y_rank_train)
        X_test_list.append(X_test)
        y_test_list.append(y_test)
        y_rank_test_list.append(y_rank_test)

    return X_train_list, X_test_list, y_train_list, y_test_list,y_rank_train_list, y_rank_test_list


def fit_clusters(X,y,clusters, method, seed=0):
    model_dict={'clusters':clusters,'sse':[],'f1':[],'adj_rand':[],'adj_MI':[],'complete':[],'homo':[]}
    
    for k in clusters:
        model=KMeans(n_clusters=k,random_state=seed)
        model.fit(X)
        
        labels = model.predict(X)
        model_dict['sse'].append(-model.score(X,labels))
        model_dict['adj_MI'].append(adjusted_mutual_info_score(y,labels))   
        model_dict['adj_rand'].append(adjusted_rand_score(y,labels))
        model_dict['complete'].append(completeness_score(y,labels))
        model_dict['homo'].append(homogeneity_score(y,labels))
        
    return model_dict,labels 



def plot_clusters(model_dict):
    f=plt.figure(figsize=(13,4))
    ax1,ax2,ax3 = f.subplots(1, 3)

    ax1.plot(model_dict['clusters'],model_dict['sse'],'r-')
    ax1.set_title('K means metrics')
    ax1.legend(['SSE'],loc=7)
    ax1.set_xlabel('n_clusters')
    ax1.set_ylabel('SSE')
    
    ax2.plot(model_dict['clusters'],model_dict['adj_MI'],'b-')
    ax2.plot(model_dict['clusters'],model_dict['adj_rand'],'b--')
    ax2.legend(['adj_MI','adj_rand'],loc=7)
    ax2.set_xlabel('n_clusters')
    ax2.set_ylabel('value')

    ax3.plot(model_dict['clusters'],model_dict['homo'],'g-')
    ax3.plot(model_dict['clusters'],model_dict['complete'],'g--')
    ax3.legend(['homogeneity','completeness'],loc=7)
    ax3.set_xlabel('n_clusters')
    ax3.set_ylabel('value')
