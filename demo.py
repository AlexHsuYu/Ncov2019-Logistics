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




pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


ratio = 0.55


logs = './logs/'
train_path = './datasets/eleme_round1_train_20200313/'
test_path = './datasets/eleme_round1_testA_20200313/'
temp_path = './temp/'


seed = 2000

def read_data():
    # courier 數據
    courier_list = []
    for f in os.listdir(os.path.join(train_path, 'courier')):
        date = f.split('.')[0].split('_')[1]
        df = pd.read_csv(os.path.join(train_path, 'courier', f))
        df['date'] = date
        courier_list.append(df)

    for f in os.listdir(os.path.join(test_path, 'courier')):
        date = f.split('.')[0].split('_')[1]
        df = pd.read_csv(os.path.join(test_path, 'courier', f))
        df['date'] = date
        courier_list.append(df)    
    # 
    df_courier = pd.concat(courier_list, sort=False)
    df_courier.to_csv(os.path.join(temp_path, 'courier.csv'),index=0)

    #order數據
    order_list = []
    for f in os.listdir(os.path.join(train_path, 'order')):
        date = f.split('.')[0].split('_')[1]
        df = pd.read_csv(os.path.join(train_path, 'order', f))
        df['date'] = date
        order_list.append(df)

    for f in os.listdir(os.path.join(test_path, 'order')):
        date = f.split('.')[0].split('_')[1]
        df = pd.read_csv(os.path.join(test_path, 'order', f))
        df['date'] = date
        order_list.append(df)

    df_order = pd.concat(order_list, sort=False)
    df_order.to_csv(os.path.join(temp_path, 'order.csv'),index=0)
    
    #distance數據
    distance_list = []
    for f in os.listdir(os.path.join(train_path, 'distance')):
        date = f.split('.')[0].split('_')[1]
        df = pd.read_csv(os.path.join(train_path, 'distance', f))
        df['date'] = date
        distance_list.append(df)

    for f in os.listdir(os.path.join(test_path, 'distance')):
        date = f.split('.')[0].split('_')[1]
        df = pd.read_csv(os.path.join(test_path, 'distance', f))
        df['date'] = date
        distance_list.append(df)


    df_distance = pd.concat(distance_list, sort=False)
    df_distance['group'] = df_distance['date'].astype(
    'str') + df_distance['courier_id'].astype('str') + df_distance['wave_index'].astype('str')
    df_distance.to_csv(os.path.join(temp_path, 'distance.csv'),index=0)

    df_actions = []
    for f in os.listdir(os.path.join(train_path, 'action')):
        date = f.split('.')[0].split('_')[1]
        df = pd.read_csv(os.path.join(train_path, 'action', f))
        df['date'] = date
        df['type'] = 'train'
        df_actions.append(df)

    for f in os.listdir(os.path.join(test_path, 'action')):
        date = f.split('.')[0].split('_')[1]
        df = pd.read_csv(os.path.join(test_path, 'action', f))
        df['date'] = date
        df['type'] = 'test'
        df_actions.append(df)



    res = Parallel(n_jobs=12)(delayed(read_feat)(df) for df in tqdm(df_actions))
    df_feature = [item[0] for item in res]
    df_history = [item[1] for item in res]

    df_feature = pd.concat(df_feature, sort=False)
    df_history = pd.concat(df_history, sort=False)
    df_feature['group'] = df_feature['date'].astype(
    'str') + df_feature['courier_id'].astype('str') + df_feature['wave_index'].astype('str')
    df_history['group'] = df_history['date'].astype(
    'str') + df_history['courier_id'].astype('str') + df_history['wave_index'].astype('str')
    df_feature['target'] = df_feature['target'].astype('float')
    df_feature['id'] = range(df_feature.shape[0])

    df_history.to_csv(os.path.join(temp_path, 'action_history.csv'),index=0)
    df_feature.to_csv(os.path.join(temp_path, 'base_feature.csv'),index=0)




def read_feat(df):
    label_list = []
    history_list = []
    type = df['type'].values[0]

   
    groups = df.groupby(['courier_id', 'wave_index'])
    for name, group in tqdm(groups):
        if type == 'train':
            label_data = group.tail(int(group.shape[0] * ratio))
            history_data = group.drop(label_data.index)

            if label_data.shape[0] < 3:
                continue
            else:
                
                label_data['target'] = 0
                label_data.reset_index(drop=True, inplace=True)
                label_data.loc[0, 'target'] = 1
                label_list.append(label_data)
                history_list.append(history_data)
        else:
            label_data = group[group['expect_time'] == 0]
            history_data = group.drop(label_data.index)

            label_data['target'] = None
            label_list.append(label_data)
            history_list.append(history_data)

    return pd.concat(label_list, sort=False), pd.concat(history_list, sort=False)    

def select_feature():
    df_history_action = pd.read_csv('./temp/action_history.csv')
    df_feature = pd.read_csv('./temp/base_feature.csv')
    df_courier = pd.read_csv('./temp/courier.csv')
    df_order = pd.read_csv('./temp/order.csv')
    df_distance = pd.read_csv('./temp/distance.csv')


    print(df_history_action.head())
    df_temp = df_history_action.groupby(['group'])['expect_time'].apply(
    lambda x: x.values.tolist()[-1]).reset_index()
    df_temp.columns = ['group', 'current_time']
    df_feature = df_feature.merge(df_temp, how='left')
    df_temp = df_history_action.groupby(['group'])['tracking_id'].apply(
    lambda x: x.values.tolist()[-1]).reset_index()
    df_temp.columns = ['group', 'last_tracking_id']
    df_feature = df_feature.merge(df_temp, how='left')

    df_temp = df_history_action.groupby(['group'])['action_type'].apply(
    lambda x: x.values.tolist()[-1]).reset_index()
    df_temp.columns = ['group', 'last_action_type']
    df_feature = df_feature.merge(df_temp, how='left')

    df_distance = df_distance.rename(columns={'tracking_id': 'last_tracking_id',
                                          'source_type': 'last_action_type', 'target_tracking_id': 'tracking_id', 'target_type': 'action_type'})
    df_feature = df_feature.merge(df_distance.drop(
    ['courier_id', 'wave_index', 'date'], axis=1), how='left')



    df_feature = df_feature.merge(
    df_order[['tracking_id', 'weather_grade', 'aoi_id', 'shop_id', 'promise_deliver_time',
              'estimate_pick_time']], how='left')



    df_feature = df_feature.merge(df_courier, how='left')


    df_feature.to_csv('./temp/part1_feature.csv',index=0)


def model():
    df_feature = pd.read_csv('./temp/part1_feature.csv')
    for f in df_feature.select_dtypes('object'):
        if f not in ['date', 'type']:
            print(f)
            lbl = LabelEncoder()
            df_feature[f] = lbl.fit_transform(df_feature[f].astype(str))

    df_test = df_feature[df_feature['type'] == 'test'].copy()


    # print(df_test)
    df_train = df_feature[df_feature['type'] == 'train'].copy()
    df_train = shuffle(df_train, random_state=seed)


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
                           random_state=seed,
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
    next_action.to_csv('./temp/next_action.csv', index=False)



def wave_label_func(group):
    target_list = group['target'].values.tolist()
    pred_list = group['pred'].values.tolist()
    max_index = pred_list.index(max(pred_list))
    if target_list[max_index] == 1:
        return 1
    else:
        return 0

def label_func(group):
    group = group.values.tolist()
    max_index = group.index(max(group))
    label = np.zeros(len(group))
    label[max_index] = 1
    return label


def select_feature_second():
    next_action = pd.read_csv('./temp/next_action.csv')
    df_feature = pd.read_csv('./temp/base_feature.csv')
    print(df_feature.shape)
    df_feature = next_action.merge(df_feature, how='left')
    print(df_feature.shape)
    df_feature['type'].value_counts()
    df_feature.to_csv('./temp/part2_feature.csv',index=False)

def model_second():
    df_feature = pd.read_csv('./temp/part2_feature.csv')


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
                          random_state=seed,
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

  

    os.makedirs('./sub/{}'.format(int(mae)), exist_ok=True)
    f = zipfile.ZipFile('./sub/{}.zip'.format(int(mae)), 'w', zipfile.ZIP_DEFLATED)
    for date in prediction['date'].unique():
        df_temp = prediction[prediction['date'] == date]
        del df_temp['date']
        df_temp.to_csv('./sub/{}/action_{}.txt'.format(int(mae), date), index=False)
        f.write('./sub/{}/action_{}.txt'.format(int(mae), date), 'action_{}.txt'.format(date))
    f.close()

# def generate_file_index():
#     category = pd.DataFrame(columns=['path','category'])
    
#     for dirPath, dirNames, fileNames in os.walk(train_data):
#         for f in fileNames:
#             if os.path.join(dirPath, f).split('.')[-1] == 'txt':
#                 category = category.append({'path':os.path.join(dirPath, f),
#                     'category':os.path.join(dirPath, f).split('/')[-1].split('.')[0]} , ignore_index=True)

#     category = category.astype('category')
#     category.sort_values(by=['category'], inplace=True)
#     category.to_csv('./logs/category.csv',index=0)

# def read_data(path):
#     df = pd.read_csv(path, index_col=False)
#     full_df = pd.DataFrame()
    
    



#     for i in range(1): #29 of data, 4 groups
#         action = pd.read_csv(df.iloc[i][0], index_col=False)
#         courier = pd.read_csv(df.iloc[i+29][0],index_col=False)
#         distance = pd.read_csv(df.iloc[i+58][0], index_col=False)
#         order = pd.read_csv(df.iloc[i+87][0],index_col=False)
        
#         full_df = pd.concat([action, courier, distance, order], axis=1)
#         # full_df = pd.concat([action, courier, distance, order], axis=1, ignore_index = True)
#         # full_df.sort_values(by=['courier_id'], inplace=True) 

#     print(full_df)
#         # print(row)
    
#     # print(full_df.head(10))
#     full_df.to_csv('All_in_one.csv',index=0)
#     # print(action.shape)
#     # print(courier.shape)
#     # print(distance.shape)
#     # print(order.shape)

if __name__ == '__main__':
# #   generate_file_index()
#   read_data('./logs/category.csv')
    read_data()
    select_feature()
    model()
    select_feature_second()
    model_second()
    
