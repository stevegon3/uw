"""Basic python/date helpers
Try to include db and http functions in invest.py"""
import datetime, calendar, time
from datetime import timedelta, timezone
from collections import OrderedDict
import re
from slack_sdk import WebClient
import pandas as pd
from dateutil.tz import gettz
import platform, requests
from typing import List
from config.settings import sett
from util.postgres import pg


def get_prev_market_open_date(the_dt=datetime.date.today()) -> str:
    if isinstance(the_dt, str):
        the_dt = datetime.datetime.strptime(the_dt[0:10], '%Y-%m-%d').date()
    prev_day = the_dt - timedelta(days=1)
    while is_market_closed(prev_day):
        prev_day = prev_day - timedelta(days=1)
    return prev_day.strftime('%Y-%m-%d')


def get_next_market_open_date(the_dt=datetime.date.today()):
    if isinstance(the_dt, str):
        the_dt = datetime.datetime.strptime(the_dt[0:10], '%Y-%m-%d').date()
    next_day = the_dt + timedelta(days=1)
    while is_market_closed(next_day):
        next_day = next_day + timedelta(days=1)
    return next_day.strftime('%Y-%m-%d')


def get_yrmo_between(start_dt, end_dt):
    dates = [start_dt, end_dt]
    start, end = [datetime.datetime.strptime(_, "%Y-%m-%d") for _ in dates]
    return OrderedDict(
        ((start + datetime.timedelta(_)).strftime(r"%Y-%m"), None) for _ in range((end - start).days)).keys()


def get_yrmody_between(start_dt_str: str, end_dt_str: str, include_end_dt=False) -> List[str]:
    start_dt = datetime.datetime.strptime(start_dt_str, "%Y-%m-%d") if isinstance(start_dt_str, str) else start_dt_str
    end_dt = datetime.datetime.strptime(end_dt_str, "%Y-%m-%d") if isinstance(end_dt_str, str) else end_dt_str
    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt
    date_range = (end_dt - start_dt).days + 1 if include_end_dt else (end_dt - start_dt).days
    return list(OrderedDict(((start_dt + datetime.timedelta(_)).strftime(r"%Y-%m-%d"), None) for _ in range(date_range)).keys())


def get_market_open_days_between(start_dt_str, end_dt_str, include_end_dt=False) -> List[str]:
    new_dates = []
    start_dt = datetime.datetime.strptime(start_dt_str, "%Y-%m-%d") if isinstance(start_dt_str, str) else start_dt_str
    start_dt_str = start_dt.strftime("%Y-%m-%d")
    end_dt = datetime.datetime.strptime(end_dt_str, "%Y-%m-%d") if isinstance(end_dt_str, str) else end_dt_str
    end_dt_str = end_dt.strftime("%Y-%m-%d")
    for date in get_yrmody_between(start_dt_str, end_dt_str, include_end_dt):
        if not is_market_closed(date):
            new_dates.append(date)
    if include_end_dt and end_dt_str not in new_dates and not is_market_closed(end_dt_str):
        new_dates.append(end_dt_str)
    return new_dates


def unix_to_date(u, pac_time=False):
    if isinstance(u, str):
        ux = int(u)
    else:
        ux = u
    if len(str(ux)) == 13:
        ux = ux / 1000
    if pac_time:
        return datetime.datetime.fromtimestamp(ux, gettz(sett.tz_pac))
    else:
        return datetime.datetime.fromtimestamp(ux, timezone.utc)


def date_to_unix(d, time_only=False):
    """Return a time in seconds from 1/1/1970"""
    if time_only:
        return calendar.timegm(datetime.datetime.strptime(d, '%H:%M:%S').timetuple())
    elif isinstance(d, str) and len(d) == 10:
        return calendar.timegm(datetime.datetime.strptime(d, '%Y-%m-%d').timetuple())
    elif isinstance(d, str) and len(d) == 16:
        return calendar.timegm(datetime.datetime.strptime(f'{d}:00', '%Y-%m-%d %H:%M:%S').timetuple())
    elif isinstance(d, str) and len(d) == 19:
        return calendar.timegm(datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S').timetuple())
    elif isinstance(d, datetime.datetime):
        return calendar.timegm(d.timetuple())
    else:
        raise 'Not a valid date or time'


def dtNow(return_type='str'):
    if return_type == 'str':
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    else:
        return datetime.datetime.now()


def post_slack(message, channel='#notify-inv'):
    try:
        client = WebClient(token=sett.slack_token)
    except:
        sett.log.error(f"Problem logging into Slack chat")
        return -1
    if isinstance(channel, str):
        channels = [channel]
    else:
        channels = channel
    for ch in channels:
        try:
            res_slack = client.chat_postMessage(channel=ch, text=message)
        except Exception as e:
            sett.log.error(f"Error posting to slack: {e}\nmessage: {message}")
    return res_slack


def get_os():
    os_name = platform.system()
    if os_name == "Windows":
        sett.log.info("You are running on Windows.")
        return "windows"
    elif os_name == "Linux":
        sett.log.info("You are running on Ubuntu or another Linux distribution.")
        return "linux"
    else:
        sett.log.info(f"You are running on {os_name}.")
        return os_name


def get_dt(return_type='str'):
    """Get the current date/time in Pacific Time"""
    time_pac = datetime.datetime.now(gettz(sett.tz_pac))
    if return_type.lower() == 'str':
        return time_pac.strftime('%Y-%m-%d %H:%M:%S')
    elif return_type.lower() in ['dt', 'ts', 'datetime', 'timestamp']:
        return time_pac
    elif return_type.lower() in ['date_str']:
        return time_pac.strftime('%Y-%m-%d')


def dt_to_str(dt, truncate_to_date=True) -> str:
    """Convert a datetime to a string"""
    if isinstance(dt, str) and truncate_to_date:
        return dt[:10]
    if isinstance(dt, str) and not truncate_to_date:
        return dt
    if isinstance(dt, float) or isinstance(dt, int):
        new_dt = unix_to_date(dt)
    else:
        new_dt = dt
    if isinstance(new_dt, datetime.datetime) and truncate_to_date:
        return new_dt.strftime('%Y-%m-%d')
    if isinstance(new_dt, datetime.datetime) and not truncate_to_date:
        return new_dt.strftime('%Y-%m-%d %H:%M:%S')

    sett.log.warning(f'bad date in dt_to_str {dt}')
    return ''


def parse_contract(contract_name: str, format='long') -> dict:
    """Parse NVDA230414P00285000 or SPY260331P590 (need to add 000 at the end)
    return {'symbol': symbol, 'exp_dt': exp_dt, 'strike': strike, 'opt_type': opt_side"""
    contract_name_tradier = fidelity_to_tradier(contract_name)
    opt_side = 'P' if 'P' in (contract_name[9], contract_name[10], contract_name[11]) else 'C'
    exp_dt_str = re.search('[0-9]+', contract_name).group()
    end_symbol_idx = contract_name.index(exp_dt_str)
    symbol = contract_name[0:end_symbol_idx]
    price_sec = re.search('[0-9]+', contract_name[end_symbol_idx + 6:]).group()
    after_opt_type = contract_name.replace(symbol, '').split(opt_side)[1]
    exp_dt = datetime.datetime.strptime(re.compile(r'\d{2}\d{2}\d{2}').search(contract_name).group(), '%y%m%d').strftime('%Y-%m-%d')
    if len(after_opt_type) > 4:  # eg NVDA230414P00285000
        strike = float(price_sec) / 1000
    else:  # eg SPY260331P590
        strike = float(price_sec)
    if format == 'short':
        return f"{symbol}{exp_dt_str}{opt_side}{int(strike)}"
    else:
        return {'symbol': symbol, 'exp_dt': exp_dt, 'strike': strike, 'opt_type': opt_side}


def get_contract_broker(contract_name: str) -> str:
    if contract_name == '' or not contract_name:
        return None
    match = re.match(r"(?P<symbol>[A-Z]+)(?P<exp>\d{6})(?P<type>[CP])(?P<strike>\d+)", contract_name)
    if not match:
        return None
    symbol = match.group("symbol")
    if len(contract_name.replace(symbol, '')) > 11:
        return 'Tradier'
    else:
        return 'Fidelity'


def tradier_to_fidelity(contract_name):
    """Convert a Tradier contract to Fidelity format.
    """
    if get_contract_broker(contract_name) == 'Fidelity':
        return contract_name
    match = re.match(r"(?P<symbol>[A-Z]+)(?P<exp>\d{6})(?P<type>[CP])(?P<strike>\d+)", contract_name)
    if not match:
        return None
    symbol = match.group("symbol")
    exp_date = match.group("exp")
    opt_type = match.group("type")
    strike = int(int(match.group("strike")) / 1000)  # Convert to decimal format
    return f"{symbol}{exp_date}{opt_type}{strike}"


def fidelity_to_tradier(contract_name):
    """Convert a Fidelity contract to Tradier format.
    """
    if get_contract_broker(contract_name) == 'Tradier':
        return contract_name
    match = re.match(r"(?P<symbol>[A-Z]+)(?P<exp>\d{6})(?P<type>[CP])(?P<strike>\d+)", contract_name)
    if not match:
        return None
    symbol = match.group("symbol")
    exp_date = match.group("exp")
    opt_type = match.group("type")
    strike = int(match.group("strike")) * 1000  # Tradier uses 1/1000th precision for strike prices
    return f"{symbol}{exp_date}{opt_type}{strike:08d}"


def is_1_d(lst):
    return not any(isinstance(i, list) for i in lst)


def get_df_high_low(df: pd.DataFrame, col_value: str, col_name: str, col_sort: str) -> list:
    """Give me a DF and col names and I will return the highest and lowest value
    Sorted by the col_sort column.
    Returns [high_name, high_value, low_name, low_value]
    """
    latest = df[col_sort].dt.normalize().max()
    today_df = df[df[col_sort].dt.normalize() == latest]
    if today_df.empty:
        highest_row = None
        lowest_row = None
    else:
        highest_idx = today_df[col_value].idxmax()
        lowest_idx = today_df[col_value].idxmin()
        highest_row = today_df.loc[highest_idx]
        lowest_row = today_df.loc[lowest_idx]
    # highest_row and lowest_row contain the full rows; access values like:
    high_name, high_value = highest_row[col_name], highest_row[col_value]
    low_name, low_value = lowest_row[col_name], lowest_row[col_value]
    return [high_name, high_value, low_name, low_value]


def get_with_retries(url: str, max_retries=3, delay=1, timeout_secs=12, debug=False, **kwargs) -> requests.Response:
    """Make a GET request with retries and dedupe kwargs"""
    request_kwargs = kwargs.copy()
    request_kwargs.pop('timeout_secs', None)
    for attempt in range(1, max_retries + 1):
        try:
            res = requests.get(url, timeout=timeout_secs, **request_kwargs)
            if debug:
                sett.log.debug(f"Attempt {attempt}/{max_retries} - Status: {res.status_code}")
            return res
        except Exception as e:
            if attempt < max_retries:
                sett.log.warning(f"Failed to connect to {url}: {e} on {attempt=}")
                time.sleep(delay)
            else:
                sett.log.error(f"Max retries exceeded for {url}")
                raise e


def get_year_quarter(date_str):
    date = datetime.datetime.strptime(date_str[:10], '%Y-%m-%d')
    year = date.year
    quarter = (date.month - 1) // 3 + 1
    return f'{year}Q{quarter}'


def is_market_closed(close_dt='now') -> bool:
    """Is the market closed on this day? Holiday or Weekend"""
    if close_dt == 'now':
        close_dt = datetime.datetime.now()
    if isinstance(close_dt, str):
        close_dt = datetime.datetime.strptime(close_dt, '%Y-%m-%d')
    if close_dt.isoweekday() in [6, 7]:
        return True
    return pg.get_num(f"SELECT count(*) FROM market_holiday WHERE holiday_dt = '{close_dt.strftime('%Y-%m-%d')}'::date") > 0


def was_market_open(close_dt='now') -> bool:
    """Was the market open now or on a particular day"""
    if close_dt == 'now':
        close_dt = datetime.datetime.now()
    else:
        if isinstance(close_dt, datetime.datetime):
            close_dt = close_dt.strftime('%Y-%m-%d')
        elif isinstance(close_dt, str):
            close_dt = datetime.datetime.strptime(close_dt, '%Y-%m-%d')
        else:
            raise Exception(f"close_dt must be datetime or str, not {type(close_dt)}")
    return not is_market_closed(close_dt)


def is_market_open() -> bool:
    """Is the market open right now?"""
    if not was_market_open('now'):
        return False
    # If time between 9:30am and 4pm, market is open
    now = datetime.datetime.now()
    if now.hour < 6 or (now.hour == 6 and now.minute < 30) or now.hour >= 13:
        return False
    return True


def check_excel_open(file_path):
    try:
        with open(file_path, 'a'):
            pass
    except PermissionError:
        user_input = input(f"WARNING: Please Close {file_path} the file and press any key").strip().lower()


def to_float(value: None) -> float:
    """Typically used for Excel values"""
    if isinstance(value, str):
        return float(value.replace('$', ''))
    elif isinstance(value, float):
        return value
    elif isinstance(value, int):
        return float(value)
    elif value is None:
        return 0
    return value


def upload_file_to_slack(file_stream, title, channels='day-close'):
    client = WebClient(token=sett.slack_token)
    if isinstance(channels, str):
        channels = [channels]
    elif channels is None:
        channels = ['day-close']
    for channel in channels:
        if channel not in sett.slack_channel_map:
            sett.log.warning(f"Channel {channel} not found in slack_channel_map")
            return False
        channel_id = sett.slack_channel_map[channel]
        try:
            response = client.files_upload_v2(
                file=file_stream,
                filename=f"{title.replace(' ', '_').lower()}.png",
                channel=channel_id,
                initial_comment=title
            )
            return response["file"]["url_private"]
        except Exception as e:
            sett.log.error(f"Error uploading file: {e} to channel {channel} with {channel_id=}")
            raise


def build_slack_msg(symbol, change_pct, change, calc_change_pct, last, bid_price, ask_price, calc_change_dlr):
    """Build slack message."""
    if calc_change_pct > 0.00:
        slack_message = f":smile: {symbol}: +{calc_change_pct:.1%}"
    elif calc_change_pct == 0:
        slack_message = f":neutral_face: {symbol}: {calc_change_pct:.1%}"
    else:
        slack_message = f":cry: loss {symbol}: {calc_change_pct:.1%}"
    slack_message += f" ${calc_change_dlr:,.0f} l${last:,.2f} day ch:{change_pct:.1%}"
    if change > 0.00:
        slack_message += ":chart_with_upwards_trend:"
    elif change < 0.00:
        slack_message += ":chart_with_downwards_trend:"
    slack_message += f" b${bid_price:,.2f} a${ask_price:,.2f}"
    return slack_message