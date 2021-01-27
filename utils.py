import os, sys, gc
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 999)
pd.set_option('display.max_columns', 999)


from .config import Config


def read_data():
    #courier 數據 
    courier_list = []
    
    #courier train_data
    for f in os.listdir(os.path.join(Config.train_path, 'courier')):
        date = f.split('.')[0].split('_')[-1]
        df = pd.read_csv(os.path.join(Config.train_path, 'courier', f))
        df['date'] = date
        courier_list.append(df)
    #courier test_data
    for f in os.listdir(os.path.join(Config.test_path, 'courier')):
        data = f.split('.')[0].split('_')[-1]
        df = pd.read_csv(os.path.join(Config.test_path, 'courier', f))
        df['date'] = date
        courier_list.append(df)

    df_courier = pd.concat(courier_list, sort=False)
    df_courier.to_csv(os.path.join(Config.temp_path, 'courier.csv'), index=0)

    #order數據
    order_list = []
    #order train data
    for f in os.listdir(os.path.join(Config.train_path, 'order')):
        date = f.split('.')[0].split('_')[-1]
        df = pd.read_csv(os.path.join(Config.train_path, 'order', f))
        df['date'] = date
        order_list.append(df)
    #order test_data
    for f in os.listdir(os.path.join(Config.test_path, 'order')):
        data = f.split('.')[0].split('_')[-1]
        df = pd.read_csv(os.path.join(Config.test_path, 'order', f))
        df['date'] = date
        order_list.append(df)

    df_order = pd.concat(order_list, sort=False)
    df_order.to_csv(os.path.join(Config.temp_path, 'order.csv'), index=0)

    #distance數據
    distance_list = []
    #distance train data
    for f in os.listdir(os.path.join(Config.train_path, 'distance')):
        date = f.split('.')[0].split('_')[-1]
        df = pd.read_csv(os.path.join(Config.train_path, 'distance', f))
        df['date'] = date
        distance_list.append(df)
    #distance test_data
    for f in os.listdir(os.path.join(Config.test_path, 'distance')):
        data = f.split('.')[0].split('_')[-1]
        df = pd.read_csv(os.path.join(Config.test_path, 'distance', f))
        df['date'] = date
        distance_list.append(df)

    df_distance = pd.concat(distance_list, sort=False)
    df_distance['group'] = df_distance['date'].astype(
    'str') + df_distance['courier_id'].astype('str') + df_distance['wave_index'].astype('str')
    df_distance.to_csv(os.path.join(Config.temp_path, 'distance.csv'),index=0)

    #action數據
    df_actions = []
    #action train data
    for f in os.listdir(os.path.join(Config.train_path, 'action')):
        data = f.split('.')[0].split('_')[1]
        df = pd.read_csv(os.path.join(Config.train_path, 'action', f))
        df['date'] = date
        df['type'] = 'train'
        df_actions.append(df)
    #action test data
    for f in os.listdir(os.path.join(Config.test_path, 'action')):
        date = f.split('.')[0].split('_')[1]
        df = pd.read_csv(os.path.join(Config.test_path, 'action', f))
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

        #劃分數據
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

        print(pd.concat(label_list, sort=False).head())
        print(pd.concat(history_list, sort=False).head())

    return pd.concat(label_list, sort=False), pd.concat(history_list, sort=False)
        
    


