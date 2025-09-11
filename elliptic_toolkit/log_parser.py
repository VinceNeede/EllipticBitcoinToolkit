import re
import pandas as pd


regex_map = {
    'cv': r'^\[CV (\d+)/[0-9]+',
    'param_value': r'(\w+)=\s*([^,\s;]+)',
    'int': r'^-?\d+$',
    'float': r'^[-+]?(?:\d*\.\d+|\d+\.\d*|\d+[eE][-+]?\d+|\d*\.\d+[eE][-+]?\d+)$',
    'time': r'^([-+]?\d+\.?\d*)\s*(s|min|m|h)$'
}

def trim_hyperparameter_results(df):
    """
    Return a DataFrame with only columns that have more than one unique value.
    """
    return df.loc[:, df.nunique() > 1]

def _parse_value(value):
    value = value.strip()
    if value == 'True':
        return True
    elif value == 'False':
        return False
    elif value == 'None':
        return None

    int_match = re.fullmatch(regex_map['int'], value)
    if int_match:
        return int(value)
    
    # If not an integer, try to match a float or scientific notation number.
    float_match = re.fullmatch(regex_map['float'], value)
    if float_match:
        return float(value)

    # Finally, try to match a time string.
    time_match = re.fullmatch(regex_map['time'], value)
    if time_match:
        num, unit = time_match.groups()
        num = float(num)
        if unit == 's':
            return num
        elif (unit == 'min') or (unit == 'm'):
            return num * 60
        elif unit == 'h':
            return num * 3600

    return value


def _parse_search_cv_log_lines(lines, trim=True):
    """
    Parse hyperparameter search results from an iterable of log lines.
    If trim is True, only return columns with more than one unique value.
    """
    data = []
    for line in lines:
        if 'END' in line:
            line = line.strip()
            params = {}

            # Extract CV fold number
            cv_match = re.search(regex_map['cv'], line)
            if cv_match:
                params['cv_fold'] = int(cv_match.group(1))

            # Extract hyperparameters using regex
            pattern = regex_map['param_value']
            matches = re.findall(pattern, line)
            for param, value in matches:
                params[param] = _parse_value(value)

            data.append(params)
    res = pd.DataFrame(data)
    if trim:
        return trim_hyperparameter_results(res)
    return res

def parse_search_cv_logs(file_path, trim=True):
    """
    Parse the hyperparameter search results.
    If trim is True, only return columns with more than one unique value.

    Parameters
    ----------
    file_path : str
        Path to the log file.
    trim : bool, default=True
        Whether to trim the DataFrame to only columns with more than one unique value.

    Returns
    -------
    res : pandas.DataFrame
        DataFrame with hyperparameter results.

    Notes
    -----
    Assumes each relevant line in the log file contains 'END', cv number as [CV x/y] 
    and hyperparameters in the format ``param=value``.
    
    Specific example line::
    
        [CV 1/5] END accuracy=0.95, learning_rate=0.01, num_layers=3,; acc=0.95, total time=3min
    
    The specific regex patterns can be adjusted in the ``regex_map`` dictionary.
    """
    with open(file_path, 'r') as f:
        return _parse_search_cv_log_lines(f, trim=trim)
