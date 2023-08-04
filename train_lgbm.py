import pandas as pd
import numpy as np
import csv
import lightgbm as lgb
import sklearn
import math
import scipy
import sys

# run: python train_lgbm.py 0.2 5 100 clock1 Nhanes_train.csv Nhanes_val.csv

filename_train = sys.argv[5] # Nhanes_train.csv
filename_val = sys.argv[6] # Nhanes_val.csv
X_train_pre = pd.read_csv(filename_train, index_col='SEQN')
X_val_pre = pd.read_csv(filename_val, index_col='SEQN')

learning_rate = float(sys.argv[1]) # 0.2
max_depth = int(sys.argv[2]) # 5
nbr = int(sys.argv[3]) # 100
clock = sys.argv[4] # clock1 clock2
label = 'RIDAGEMN'
cols = []
with open('train_columns_list.txt', 'r') as filehandle:
    for line in filehandle:
        curr_place = line[:-1]
        cols.append(curr_place)
        
if clock == 'clock1':
    cols_to_remove = ['RIAGENDR','BMXBMI', 'BMXWT', 'BMXHT']
    for c in cols_to_remove:
        cols.remove(c)
    model_filename_to_save = 'lgb_model_clock1.txt'
elif clock == 'clock2':
    model_filename_to_save = 'lgb_model_clock2.txt'

X_train = X_train_pre[list(cols)]
y_train = X_train_pre[label]
X_val = X_val_pre[list(cols)]
y_val = X_val_pre[label]

hyper_params = {
                'objective': 'regression', # default = regression
                #'metric': 'l2',
                'num_leaves': 2*max_depth, # default = 31
                'boosting': 'gbdt', # default = gbdt
                'max_depth': max_depth, # default = -1
                #'n_estimators': 100, # default = 100, aliases: num_boost_round
                'learning_rate': learning_rate, # default = 0.1
                'verbose': 0,
                'zero_as_missing': False,
                'force_col_wise' : True
                 }
pool_train = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
pool_val = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

model = lgb.train(hyper_params, pool_train, valid_sets=pool_val,
                  num_boost_round=nbr, keep_training_booster=True)

model.save_model(model_filename_to_save, num_iteration=model.best_iteration)

pred = model.predict(X_val)
print('MSE:', sklearn.metrics.mean_squared_error(y_val, pred))
print('RMSE:', math.sqrt(sklearn.metrics.mean_squared_error(y_val, pred)))
print('MAE:', sklearn.metrics.mean_absolute_error(y_val, pred))
print('Pearson r:', scipy.stats.pearsonr(y_val, pred)[0])