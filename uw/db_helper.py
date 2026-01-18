"""Helper functions specifically for Postgre module to prevent circular imports"""
import decimal, math, datetime
import pandas as pd
from typing import Any
from settings import sett

def is_datetime(my_string, debug=False):
    # List of common date time formats to check against
    date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%m-%d-%Y', '%d-%m-%Y', '%d/%m/%Y', '%Y%m%d', '%m%d%Y', '%d%m%Y']
    time_formats = ['%H:%M:%S', '%I:%M:%S %p', '%H%M%S']
    # Loop through formats and try to parse string as datetime
    if isinstance(my_string, datetime.datetime) or isinstance(my_string, datetime.date):
        return True
    if isinstance(my_string, str) and my_string.isdigit():
        return False
    if isinstance(my_string, int) or isinstance(my_string, float):
        return False
    if my_string is not None and isinstance(my_string, str):
        if len(my_string) >= 8:
            for date_fmt in date_formats:
                for time_format in time_formats:
                    for ty in ['date_only', 'time_only', 'both']:
                        if ty == 'date_only':
                            try:
                                dt = datetime.datetime.strptime(my_string, f'{date_fmt}')
                                if debug:
                                    print('ty==date_only')
                                return True
                            except ValueError:
                                pass
                        elif ty == 'time_only':
                            try:
                                dt = datetime.datetime.strptime(my_string, f'{time_format}')
                                if debug:
                                    print('ty==time_only')
                                return True
                            except ValueError:
                                pass
                        elif ty == 'both':
                            try:
                                dt = datetime.datetime.strptime(my_string, f'{date_fmt} {time_format}')
                                if debug:
                                    print('ty==both')
                                return True
                            except ValueError:
                                pass
                            try:
                                dt = datetime.datetime.strptime(my_string, f'{date_fmt}{time_format}')
                                if debug:
                                    print('ty==both')
                                return True
                            except ValueError:
                                pass
    return False

def get_type(my_string, pandas=False):
    """Will return an interpreted data type of any Py variable"""
    if my_string is None:
        return None
    elif is_datetime(my_string) and pandas:
        return 'datetime64[ns]'
    elif is_datetime(my_string):
        return 'datetime'
    elif isinstance(my_string, str):
        return 'str'
    elif isinstance(my_string, int):
        return 'int'
    elif isinstance(my_string, float):
        return 'float'
    elif isinstance(my_string, decimal.Decimal):
        return 'decimal'
    elif isinstance(my_string, list):
        return 'list'
    elif isinstance(my_string, dict):
        return 'dict'
    elif isinstance(my_string, tuple):
        return 'tuple'
    else:
        return type(my_string)

def list_to_df_orig(my_list, first_row_is_header=True):
    """Give me a list and I will look at the first row of data and PROPERLY coerce
    and convert it to a DataFrame with proper column data types."""
    if my_list and my_list != [[]]:
        if first_row_is_header:
            column_names = my_list[0]
            if len(my_list[1:]) > 0:
                data = my_list[1:]
            else:
                return pd.DataFrame([pd.Series([None] * len(column_names), index=column_names)], columns=column_names)
        else:
            column_names = [f'col{x}' for x in range(len(my_list[0]))]
            data = my_list
        column_types = [get_type(x, True) for x in data[0]]
        column_dict = dict(zip(column_names, column_types))
        df = pd.DataFrame(data, columns=column_names)
        for col_name, data_type in column_dict.items():
            try:
                if data_type is None:
                    if 'date' in col_name.lower() or col_name.lower().endswith('dt') or col_name.lower().endswith('pac') or col_name.lower().endswith('east'):
                        df[col_name] = pd.to_datetime(df[col_name])
                    else:
                        df[col_name] = df[col_name].astype(object)
                elif data_type == 'decimal':
                    df[col_name] = df[col_name].astype('float')
                elif data_type == 'datetime64[ns]':
                    df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
                else:
                    df[col_name] = df[col_name].astype(data_type)
            except (ValueError, TypeError):
                df[col_name] = df[col_name].astype(object)
        return df
    else:
        return pd.DataFrame([[None]], columns=[''])


def list_to_df(my_list, first_row_is_header=True):
    """Convert a list to a DataFrame with proper type handling."""
    pd.set_option('future.no_silent_downcasting', True)
    if not my_list or my_list == [[]]:
        return pd.DataFrame([[None]], columns=[''])
    try:
        if first_row_is_header:
            column_names = my_list[0]
            data = my_list[1:]
        else:
            column_names = [f'col_{i}' for i in range(len(my_list[0]))]
            data = my_list
        # First pass: create DataFrame with object dtype to preserve values
        df = pd.DataFrame(data, columns=column_names, dtype=object)
        column_types = [get_type(x, True) for x in data[0]]
        column_dict = dict(zip(column_names, column_types))
        # Second pass: convert types
        for col_name, data_type in column_dict.items():
            # print(f'{col_name}: {data_type}')
            try:
                if col_name.lower() in ['delisted', 'track', 'is_active', 'active']:  # Add other boolean column names as needed
                    df[col_name] = df[col_name].astype(str).str.lower().map({
                        'true': True, 't': True, '1': True, '1.0': True, 1: True, 1.0: True,
                        'false': False, 'f': False, '0': False, '0.0': False, 0: False, 0.0: False,
                        'none': None, 'null': None, 'nan': None, '': None
                    }, na_action='ignore').fillna(False).astype(bool)
                elif 'datetime' in data_type  or 'date' in col_name.lower() or col_name.lower().endswith('dt') or col_name.lower().endswith('pac') or col_name.lower().endswith('east'):
                    df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
                    mask = df[col_name].notna()
                    df.loc[mask, col_name] = df.loc[mask, col_name].dt.strftime('%Y-%m-%d %H:%M:%S')
                    df[col_name] = df[col_name].replace('NaT', None)
                    df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
                elif data_type == 'str':
                    df[col_name] = df[col_name].astype(str).str.strip().fillna('')
                elif data_type == 'int':
                    df[col_name] = df[col_name].astype(int).fillna(0)
                elif data_type in ['float', 'decimal', 'real']:
                    df[col_name] = df[col_name].astype(float).fillna(0.0)
                elif df[col_name].notna().any():  # Only try numeric conversion if there are non-null values
                    try:
                        numeric_series = pd.to_numeric(df[col_name], errors='coerce')
                        if not numeric_series.dtype == object:  # Only if conversion was successful
                            df[col_name] = numeric_series
                    except (ValueError, TypeError):
                        pass
                # print(f'Resulting conversion for {col_name}: {df[col_name].dtype}')
            except Exception as e:
                pass
        return df

    except Exception as e:
        sett.log.error(f"Error in list_to_df: {str(e)} returning empty DataFrame.")
        return pd.DataFrame(columns=column_names if first_row_is_header else [])

def assert_date_string(date_string):
    try:
        datetime.datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def fix_none(value, default=0):
    return default if value is None else value

def replace_none_dict(dict_, string_keys=None, num_keys=None):
    """Replace None values in a dictionary with empty strings or a default value."""
    if dict_ == [] or dict_ == {} or not dict_:
        return {}
    if not isinstance(dict_, dict):
        return dict_
    for key, value in dict_.items():
        if string_keys and key in string_keys and value is None:
            dict_[key] = ''
        elif num_keys and key in num_keys and value is None:
            dict_[key] = 0.0
        elif value is None:
            dict_[key] = ''
    return dict_

def normalize_value(value: Any) -> Any:
    """Convert NaN/NaT-like values into None (NULL in Postgres).
    Handles: - float('nan')
    - pandas / numpy NaN, NaT, etc., if pandas is installed
    - Leaves regular values and None unchanged.
    """
    if value is None:
        return None
    if pd is not None:
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass # Some types cannot be checked with pd.isna; ignore and continue
    if isinstance(value, float) and math.isnan(value):
        return None
    return value