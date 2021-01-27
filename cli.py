import os, sys
import pandas as pd, numpy as np


def select_feature():
    df_history_action = pd.read_csv('./ncov2019-Logistics/temp/action_history.csv')
    df_feature = pd.read_csv('./ncov2019-Logistics/temp/base_feature.csv')
    df_courier = pd.read_csv('./ncov2019-Logistics/temp/courier.csv')
    df_order = pd.read_csv('./ncov2019-Logistics/temp/order.csv')
    df_distance = pd.read_csv('./ncov2019-Logistics/temp/distance.csv')


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


    df_feature.to_csv('./ncov2019-Logistics/temp/part1_feature.csv',index=0)

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
    next_action = pd.read_csv('./ncov2019-Logistics/temp/next_action.csv')
    df_feature = pd.read_csv('./ncov2019-Logistics/temp/base_feature.csv')
    print(df_feature.shape)
    df_feature = next_action.merge(df_feature, how='left')
    print(df_feature.shape)
    df_feature['type'].value_counts()
    df_feature.to_csv('./ncov2019-Logistics/temp/part2_feature.csv',index=False)
