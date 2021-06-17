from features.base import StaticFeature, DynamicFeature
from functools import reduce
from datetime import datetime
import pandas as pd


'''def build_features(features_lst: list, prediction_date: datetime, id_col='city_code',
                   filter_func=lambda row: row['total_pop'] > 10000):
    if len(features_lst) == 0:
        raise Exception("The given features list must have at least one feature.")
    df_lst = []
    for feature in features_lst:
        if isinstance(feature, StaticFeature):
            df_lst.append(feature.run())
        elif isinstance(feature, DynamicFeature):
            df = feature.run(prediction_date)
            df_lst.append(df)
        else:
            raise Exception(f"Features must be of types StaticFeature or DynamicFeature, given {type(feature)}.")
    final_df = reduce(lambda x, y: pd.merge(x, y, on=id_col), df_lst)
    # validate all features has the same ids
    #assert(final_df.shape[0] == df_lst[0].shape[0])

    if filter_func is not None:
        final_df = final_df[filter_func]
    final_df.set_index(id_col, inplace=True)
    return final_df'''


def build_features(features_lst: list, prediction_date, id_col='city_code', date_col='day_date',
                        filter_func=lambda row: row['total_pop'] > 10000):
    if len(features_lst) == 0:
        raise Exception("The given features list must have at least one feature.")
    if type(prediction_date) is not list:
        prediction_date_lst = [prediction_date]
    else:
        prediction_date_lst = prediction_date
    dates_df_lst = []
    for p in prediction_date_lst:
        current_df_lst = []
        for feature in features_lst:
            if isinstance(feature, StaticFeature):
                current_df_lst.append(feature.run())
            elif isinstance(feature, DynamicFeature):
                df = feature.run(p)
                current_df_lst.append(df)
            else:
                raise Exception(f"Features must be of types StaticFeature or DynamicFeature, given {type(feature)}.")
        current_final_df = reduce(lambda x, y: pd.merge(x, y, on=id_col), current_df_lst)
        if filter_func is not None:
            current_final_df = current_final_df[filter_func]
        current_final_df[date_col] = p.replace(hour=0, minute=0, second=0, microsecond=0)
        dates_df_lst.append(current_final_df)

    final_df = pd.concat(dates_df_lst)
    if type(prediction_date) is not list:
        del final_df[date_col]
        final_df.set_index(id_col, inplace=True)
    else:
        final_df.set_index([id_col, date_col], inplace=True)

    return final_df
