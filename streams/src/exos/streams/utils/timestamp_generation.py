import numpy as np
import datetime
import random


def generate_timestamp(nsamples, arrival_rate='Fixed', time_unit='seconds', time_interval=1):
    """
    Generate a numpy array of timestamps of length nsamples
    Parameters:
    -----------
    nsamples: int
        number of data points

    arrival_rate: str, default 'Fixed'
        a token to set whether the time difference is fixed or not

    time_unit: str, default 'seconds'
        the value can be: days, seconds, microseconds, milliseconds, minutes, hours, week

    time_interval: float, default 1
        by default the every timestamp tuple will have 1 second difference
    """
    start = datetime.datetime.today()

    if time_unit == 'seconds':
        timedelta = datetime.timedelta(seconds=time_interval)
    elif time_unit == 'microseconds':
        timedelta = datetime.timedelta(microseconds=time_interval)
    elif time_unit == 'milliseconds':
        timedelta = datetime.timedelta(milliseconds=time_interval)
    elif time_unit == 'minutes':
        timedelta = datetime.timedelta(minutes=time_interval)
    elif time_unit == 'hours':
        timedelta = datetime.timedelta(hours=time_interval)
    elif timedelta == 'weeks':
        timedelta = datetime.timedelta(weeks=time_interval)
    else:
        timedelta = datetime.timedelta(days=time_interval)

    if arrival_rate == 'Fixed':
        timestamps = [start]
        for i in range(1, nsamples):
            timestamps.append(start + i * timedelta)
    else:
        end = start + timedelta
        random.seed(42)
        timestamps = [random.random() * (end - start) + start for _ in range(nsamples)]

    return np.array(timestamps, dtype="datetime64")
