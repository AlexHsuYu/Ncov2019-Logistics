import sys, os, gc
import zipfile

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

from joblib import Parallel, delayed
from sklearn import preprocessing, metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from .config import Config
from .cli import *

def model():
    df_feature = pd.read_csv('./ncov2019-Logistics/temp/part1_feature.csv')
    for f in df_feature.select_dtypes('object'):
        if f not in ['date', 'type']:
            print(f)
            lbl = LabelEncoder()
            df_feature[f] = lbl.fit_transform(df_feature[f].astype(str))

    df_test = df_feature[df_feature['type'] == 'test'].copy()


    # print(df_test)
    df_train = df_feature[df_feature['type'] == 'train'].copy()
    df_train = shuffle(df_train, random_state=Config.seed)


    ycol = 'target'
    feature_names = list(
    filter(lambda x: x not in [ycol, 'id', 'wave_index', 'tracking_id', 'expect_time', 'date', 'type', 'group',
                               'courier_wave_start_lng', 'courier_wave_start_lat', 'shop_id', 'current_time_date'], df_train.columns))

    model = lgb.LGBMClassifier(num_leaves=64,
                           max_depth=10,
                           learning_rate=0.1,
                           n_estimators=10000000,
                           subsample=0.8,
                           feature_fraction=0.8,
                           reg_alpha=0.5,
                           reg_lambda=0.5,
                           random_state=Config.seed,
                           metric=None
                           )


    oof = []
    prediction = df_test[['id', 'group']]
    prediction['target'] = 0
    df_importance_list = []

    kfold = GroupKFold(n_splits=5)
    for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(df_train[feature_names], df_train[ycol], df_train['group'])):
        X_train = df_train.iloc[trn_idx][feature_names]
        Y_train = df_train.iloc[trn_idx][ycol]

        X_val = df_train.iloc[val_idx][feature_names]
        Y_val = df_train.iloc[val_idx][ycol]

        print('\nFold_{} Training ================================\n'.format(fold_id+1))

        lgb_model = model.fit(X_train,
                          Y_train,
                          eval_names=['train', 'valid'],
                          eval_set=[(X_train, Y_train), (X_val, Y_val)],
                          verbose=500,
                          eval_metric='auc',
                          early_stopping_rounds=50)

        pred_val = lgb_model.predict_proba(
            X_val, num_iteration=lgb_model.best_iteration_)[:, 1]
        df_oof = df_train.iloc[val_idx][['id', 'group', ycol]].copy()
        df_oof['pred'] = pred_val
        oof.append(df_oof)


        print(df_test[feature_names])
        pred_test = lgb_model.predict_proba(
            df_test[feature_names], num_iteration=lgb_model.best_iteration_)[:, 1]
        prediction['target'] += pred_test / 5

        df_importance = pd.DataFrame({
            'column': feature_names,
            'importance': lgb_model.feature_importances_,
        })
        df_importance_list.append(df_importance)

        del lgb_model, pred_val, pred_test, X_train, Y_train, X_val, Y_val
        gc.collect()


    df_importance = pd.concat(df_importance_list)
    df_importance = df_importance.groupby(['column'])['importance'].agg(
    'mean').sort_values(ascending=False).reset_index()

    df_oof = pd.concat(oof)
    df_temp = df_oof.groupby(['group']).apply(wave_label_func).reset_index()
    df_temp.columns = ['group', 'label']
    acc = df_temp[df_temp['label'] == 1].shape[0] / df_temp.shape[0]
    print('acc:', acc)
    prediction['label'] = prediction.groupby(
    ['group'])['target'].transform(label_func)
    sub_part1 = prediction[prediction['label'] == 1]
    df_oof = df_oof[df_oof['target'] == 1]
    next_action = pd.concat([df_oof[['id']], sub_part1[['id']]])
    next_action.to_csv('./ncov2019-Logistics/temp/next_action.csv', index=False)

def model_second():
    df_feature = pd.read_csv('./ncov2019-Logistics/temp/part2_feature.csv')


    df_test = df_feature[df_feature['type'] == 'test'].copy()
    df_train = df_feature[df_feature['type'] == 'train'].copy()
    prediction = df_test[['courier_id', 'wave_index', 'tracking_id',
                      'courier_wave_start_lng', 'courier_wave_start_lat', 'action_type', 'expect_time', 'date']]
    prediction['expect_time'] = 0



    for f in df_feature.select_dtypes('object'):
        if f not in ['date', 'type']:
            print(f)
            lbl = LabelEncoder()
            lbl = lbl.fit(df_train[f].astype(
                str).values.tolist()+df_test[f].astype(str).values.tolist())
            df_train[f] = lbl.transform(df_train[f].astype(str))
            df_test[f] = lbl.transform(df_test[f].astype(str))
    ycol = 'expect_time'
    feature_names = list(
        filter(lambda x: x not in [ycol, 'id', 'wave_index', 'tracking_id', 'target', 'date', 'type', 'group',
                               'courier_wave_start_lng', 'courier_wave_start_lat'], df_train.columns))

    model = lgb.LGBMRegressor(num_leaves=64,
                          max_depth=10,
                          learning_rate=0.1,
                          n_estimators=10000000,
                          subsample=0.8,
                          feature_fraction=0.8,
                          reg_alpha=0.5,
                          reg_lambda=0.5,
                          random_state=Config.seed,
                          metric=None
                          )


    oof = []
    df_importance_list = []

    kfold = GroupKFold(n_splits=5)
    for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(df_train[feature_names], df_train[ycol], df_train['group'])):
        X_train = df_train.iloc[trn_idx][feature_names]
        Y_train = df_train.iloc[trn_idx][ycol]

        X_val = df_train.iloc[val_idx][feature_names]
        Y_val = df_train.iloc[val_idx][ycol]

        print('\nFold_{} Training ================================\n'.format(fold_id+1))

        lgb_model = model.fit(X_train,
                          Y_train,
                          eval_names=['train', 'valid'],
                          eval_set=[(X_train, Y_train), (X_val, Y_val)],
                          verbose=500,
                          eval_metric='mae',
                          early_stopping_rounds=50)

        pred_val = lgb_model.predict(
        X_val, num_iteration=lgb_model.best_iteration_)
        df_oof = df_train.iloc[val_idx][['id', ycol]].copy()
        df_oof['pred'] = pred_val
        oof.append(df_oof)

        pred_test = lgb_model.predict(
            df_test[feature_names], num_iteration=lgb_model.best_iteration_)
        prediction['expect_time'] += pred_test / 5

        df_importance = pd.DataFrame({
            'column': feature_names,
            'importance': lgb_model.feature_importances_,
        })
        df_importance_list.append(df_importance)

        del lgb_model, pred_val, pred_test, X_train, Y_train, X_val, Y_val
        gc.collect()

    df_importance = pd.concat(df_importance_list)
    df_importance = df_importance.groupby(['column'])['importance'].agg(
        'mean').sort_values(ascending=False).reset_index()
    df_oof = pd.concat(oof)
    mae = metrics.mean_absolute_error(df_oof[ycol], df_oof['pred'])
    print('mae:', mae)
 
    os.makedirs('./ncov2019-Logistics/sub/{}'.format(int(mae)), exist_ok=True)
    f = zipfile.ZipFile('./ncov2019-Logistics/sub/{}.zip'.format(int(mae)), 'w', zipfile.ZIP_DEFLATED)
    for date in prediction['date'].unique():
        df_temp = prediction[prediction['date'] == date]
        del df_temp['date']
        df_temp.to_csv('./ncov2019-Logistics/sub/{}/action_{}.txt'.format(int(mae), date), index=False)
        f.write('./ncov2019-Logistics/sub/{}/action_{}.txt'.format(int(mae), date), 'action_{}.txt'.format(date))
    f.close()
