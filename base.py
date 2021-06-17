import inspect
import datetime
import os
from abc import ABC, abstractmethod
import pandas as pd
from utils.functions import hash_params


class Feature(ABC):
    def __init__(self, feature_function, id_col='city_code', filter_on=None, **kwargs):
        self._id_col = id_col
        self._params_dict = kwargs
        self._feature_function = feature_function
        self._filter_on = filter_on

    @property
    def feature_name(self):
        return self._feature_function.__name__

    @property
    def id_col(self):
        return self._id_col

    @abstractmethod
    def run(self, *args):
        pass


class StaticFeature(Feature):
    def __init__(self, feature_function, **kwargs):
        super().__init__(feature_function, **kwargs)

    def run(self):
        location = os.path.dirname(__file__)
        location = location[:location.rfind(os.path.sep)]
        feature_path = os.path.join(location,
                                    f'cache/static_features/{self.feature_name}')
        pickle_file_name = f'{hash_params(self._params_dict)}.pkl'
        full_feature_path = os.path.join(feature_path, pickle_file_name)
        if os.path.exists(full_feature_path):
            df = pd.read_pickle(full_feature_path)
        else:
            df = self._feature_function(**self._params_dict)
            if not os.path.exists(feature_path):
                os.makedirs(feature_path)
            df.to_pickle(full_feature_path)

        # take filtered columns if required
        if self._filter_on is not None:
            if isinstance(self._filter_on, list):
                filter_list = self._filter_on
            else:
                filter_list = [self._filter_on]
            df = df[[self.id_col] + filter_list]
        return df


class DynamicFeature(Feature):
    def __init__(self, feature_function, offset=0, date_col='day_date', **kwargs):
        super().__init__(feature_function, **kwargs)
        self._date_col = date_col
        self._offset = offset

    @property
    def date_col(self):
        return self._date_col

    def _get_feature_location(self, offset):
        location = os.path.dirname(__file__)
        location = location[:location.rfind(os.path.sep)]
        cached_params = self._params_dict.copy()
        cached_params.update({'offset': offset})
        return os.path.join(location, f'cache/dynamic_features/{self.feature_name}/{hash_params(cached_params)}')

    def _force_run(self, prediction_date, offset):
        if offset == 0:
            df = self._feature_function(prediction_date, **self._params_dict)
        else:
            df = self._run_with_offset(prediction_date=prediction_date - datetime.timedelta(days=offset), offset=0)
            #df[self.date_col] = df.apply(lambda row: row[self.date_col]+datetime.timedelta(days=offset), axis=1)
            renamed_cols = {c: f'{c}_before_{offset}_days' for c in df.columns if c not in [self.id_col, self.date_col]}
            df.rename(columns=renamed_cols, inplace=True)

        feature_path = self._get_feature_location(offset)
        pickle_file_name = f"{prediction_date.strftime('%Y%m%d')}.pkl"
        full_pickle_path = os.path.join(feature_path, pickle_file_name)
        df.to_pickle(full_pickle_path)

        return df

    def run(self, prediction_date):
        df = self._run_with_offset(prediction_date=prediction_date, offset=self._offset)
        # take filtered columns if required
        if self._filter_on is not None:
            if isinstance(self._filter_on, list):
                filter_list = self._filter_on
            else:
                filter_list = [self._filter_on]
            df = df[[self.id_col] + filter_list]
        return df

    def _run_with_offset(self, prediction_date, offset):
        prediction_date = prediction_date.replace(hour=0, minute=0, second=0, microsecond=0)

        feature_path = self._get_feature_location(offset)
        if not os.path.exists(feature_path):
            os.makedirs(feature_path)

        pickle_file_name = f"{prediction_date.strftime('%Y%m%d')}.pkl"
        full_pickle_path = os.path.join(feature_path, pickle_file_name)
        if os.path.exists(full_pickle_path):
            df = pd.read_pickle(full_pickle_path)
        else:
            df = self._force_run(prediction_date=prediction_date, offset=offset)
        return df


def static_feature(feature_func):
    def wrapper(*args, **kwargs):
        func_args_names = [k for k in inspect.signature(feature_func).parameters]
        if feature_func.__defaults__ is not None:
            args_to_drop_count = len(feature_func.__defaults__)
            func_args_names = func_args_names[:-args_to_drop_count]
        additional_kwargs = dict(zip(func_args_names, args))
        kwargs.update(additional_kwargs)
        return StaticFeature(feature_func, **kwargs)
    return wrapper


def dynamic_feature(feature_func):
    def wrapper(*args, **kwargs):
        func_args_names = [k for k in inspect.signature(feature_func).parameters][2:]
        if feature_func.__defaults__ is not None:
            args_to_drop_count = len(feature_func.__defaults__)
            func_args_names = func_args_names[:-args_to_drop_count]
        additional_kwargs = dict(zip(func_args_names, args))
        kwargs.update(additional_kwargs)
        return DynamicFeature(feature_func, **kwargs)
    return wrapper
