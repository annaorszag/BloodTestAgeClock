import pandas as pd
import numpy as np
import csv
import lightgbm as lgb
import sklearn
import math
import scipy
import sys

# run: python test_lgbm.py lgb_model_clock1.txt Nhanes_test_age_matched.csv athletes_data_1st_ex.csv preds_age-matched_NHANES_test_clock1.csv preds_athletes_clock1.csv

model_filename = sys.argv[1] # 'lgb_model_clock1.txt'
filename_test_nhanes = sys.argv[2] # 'Nhanes_test_age_matched.csv'
filename_test_athletes = sys.argv[3] # 'athletes_data_1st_ex.csv'

model = lgb.Booster(model_file=model_filename)
X_test_pre = pd.read_csv(filename_test_nhanes, index_col='SEQN')
X_test_athletes = pd.read_csv(filename_test_athletes, index_col = 'ID')

filename_Nhanes_test_pred = sys.argv[4]
filename_athletes_pred = sys.argv[5] # preds_athletes.csv

label='RIDAGEMN'
cols =  []
with open('train_columns_list.txt', 'r') as filehandle:
    for line in filehandle:
        curr_place = line[:-1]
        cols.append(curr_place)

if model_filename[-10:-4] == 'clock1':
    cols_to_remove = ['RIAGENDR','BMXBMI', 'BMXWT', 'BMXHT']
    for c in cols_to_remove:
        cols.remove(c)

X_test_nhanes = X_test_pre[list(cols)]
y_test_nhanes = X_test_pre[label]

pred_nhanes = model.predict(X_test_nhanes)
print('NHANES MSE:', sklearn.metrics.mean_squared_error(y_test_nhanes, pred_nhanes))
print('NHANES RMSE:', math.sqrt(sklearn.metrics.mean_squared_error(y_test_nhanes, pred_nhanes)))
print('NHANES MAE:', sklearn.metrics.mean_absolute_error(y_test_nhanes, pred_nhanes))
print('NHANES Pearson r:', scipy.stats.pearsonr(y_test_nhanes, pred_nhanes)[0])

preds_Nhanes_test = y_test_nhanes.reset_index()
preds_Nhanes_test['pred'] = pred_nhanes
preds_Nhanes_test.to_csv(filename_Nhanes_test_pred)

def prepare_data(df, label = 'RIDAGEYR', columns = list(cols)):
    if 'RIDAGEYR' in columns:
        columns.remove('RIDAGEYR')
    if 'RIDAGEMN' in columns:
        columns.remove('RIDAGEMN')
    df.dropna(subset=label, inplace=True)
    df = df[columns+[label]].drop_duplicates()
    X = df[columns+[label]]
    XX =df[columns]
    y=df[label]
    return XX, X, y, columns, label

XX_athletes, X_athletes, y_athletes, c, label = prepare_data(X_test_athletes, 
                                 label='RIDAGEMN',
                                    columns = cols)

pred = model.predict(XX_athletes)
print('Athletes MSE:', sklearn.metrics.mean_squared_error(y_athletes, pred))
print('Athletes RMSE:', math.sqrt(sklearn.metrics.mean_squared_error(y_athletes, pred)))
print('Athletes MAE:', sklearn.metrics.mean_absolute_error(y_athletes, pred))
print('Athletes Pearson r:', scipy.stats.pearsonr(y_athletes, pred)[0])

preds_athletes_pc = y_athletes.reset_index()
preds_athletes_pc['pred'] = pred
preds_athletes_pc.to_csv(filename_athletes_pred)