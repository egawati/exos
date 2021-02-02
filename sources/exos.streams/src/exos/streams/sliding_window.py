import numpy as np
import datetime
import random
import math

from skmultiflow.utils import get_dimensions

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


class BufferedWindow(object):
    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time
        self.size = 0
        self.latest_timestamp = None
        self.earliest_timestamp = None
        self.members = {'uids': [], 'timestamps': [], 'X': None, 'y': None}

    def update_buffer_content(self, X, y, timestamp, uid):
        self.members['uids'].append(uid)
        if self.members['X'] is None:
            self.members['X'] = np.zeros((0, get_dimensions(X)[1]))
            self.members['y'] = np.zeros((0, get_dimensions(y)[1]))
            self.earliest_timestamp = timestamp
        self.members['X'] = np.vstack((self.members['X'], X))
        self.members['y'] = np.vstack((self.members['y'], y))
        self.members['timestamps'].append(timestamp)
        self.size += 1
        self.latest_timestamp = timestamp


class TemporalSlidingWindow(object):
    """ Keep a temporal sliding window of the most recent data samples.
    Parameters
    ----------

    start_time: datetime64

    window_size: np.timedelta64, default 5 minutes
        Time interval for the window

    sliding_size : np.timedelta64, default 1 minute
        Time interval to slide the window

    Raises
    ------
    ValueError
        If at any moment, a sample with a different number of attributes than
         those already observed is passed.

    Notes
    -----
    It updates its stored samples by the FIFO method, which means
    that when the window (buffer) slides, old samples are dumped to give
    place to new samples.

    """

    def __init__(self, start_time, window_size=np.timedelta64(5, "m"),
                 sliding_size=np.timedelta64(1, "m"), sliding='Fixed', keep_history=True):
        super().__init__()

        self.window_size = window_size
        self.sliding_size = sliding_size
        self.sliding = sliding

        self._n_features = -1
        self._n_targets = -1
        self._X_queue = None
        self._y_queue = None
        self._point_timestamps = None
        self.uid = []
        self.current_window = None
        self._is_initialized = False

        self.start_time = start_time
        self.end_time = self.start_time + window_size

        self.current_time = start_time

        self.window_idx = 0
        self.keep_history = keep_history
        self.overlap_windows = None

        self.nsamples = 0

        self.new_slide = False

    def configure(self):
        self._is_initialized = True
        self.current_window = 'window_' + str(self.window_idx)
        if self.keep_history:
            self.w_history = {}

        if self.window_size > self.sliding_size:
            self.overlap_windows = [None] * math.ceil(self.window_size / self.sliding_size)
            buffer = BufferedWindow(self.start_time, self.end_time)
            self.overlap_windows[0] = buffer
            logging.info('The size of the overlap_windows is {}'.format(len(self.overlap_windows)))
            logging.info('buffer {} is set for overlap_window idx 0'.format(buffer))

        if not self.overlap_windows:
            self._X_queue = np.zeros((0, self._n_features))
            self._y_queue = np.zeros((0, self._n_targets))
            self._point_timestamps = np.zeros(0, dtype='datetime64[us]')

    def add_sample(self, X, y, arrival_time):
        if not self._is_initialized:
            self._n_features = get_dimensions(X)[1]
            self._n_targets = get_dimensions(y)[1]
            self.configure()

        if self._n_features != get_dimensions(X)[1]:
            raise ValueError("Inconsistent number of features in X: {}, previously observed {}.".
                             format(get_dimensions(X)[1], self._n_features))

        if not self.overlap_windows:
            self._add_sample_no_overlap(X, y, arrival_time)
        else:
            self._add_sample_overlap(X, y, arrival_time)

    def _add_sample_overlap(self, X, y, arrival_time):
        logging.info('-----------------------------------------------------------------------------------')
        self.nsamples += 1
        self.current_time = arrival_time[-1]
        if self.current_time > (self.start_time + self.sliding_size):
            self.new_slide = True
            self.window_idx += 1
            self.start_time += self.sliding_size
            end_time = self.start_time + self.window_size
            buffer = BufferedWindow(self.start_time, end_time)
            idx = self.window_idx % len(self.overlap_windows)
            self.overlap_windows[idx] = buffer
        else:
            self.new_slide = False
        self._update_uid()
        logging.info("data point {}".format(self.uid[-1]))

        # examine which buffered window should X go (X can belong to more than one buffer)
        for i in range(len(self.overlap_windows)):
            buf = self.overlap_windows[i]
            if buf:
                if self.current_time >= buf.start_time and self.current_time <= buf.end_time:
                    buf.update_buffer_content(X, y, self.current_time, self.uid[-1])

        if self.keep_history:
            self._update_window_history()

    def _update_uid(self):
        if not self.uid:
            self.uid = [1]
        else:
            self.uid.append(self.uid[-1] + 1)

    def _add_sample_no_overlap(self, X, y, arrival_time):
        self.current_time = arrival_time[-1]

        # slide the window when necessary
        if self.current_time > (self.start_time + self.sliding_size):
            self.new_slide = True
            self._slide_window_no_overlap()
        else:
            self.new_slide = False

        self._X_queue = np.vstack((self._X_queue, X))
        self._y_queue = np.vstack((self._y_queue, y))
        self._point_timestamps = np.append(self._point_timestamps, arrival_time)
        self._update_uid()

    def _slide_window_no_overlap(self):
        if self.keep_history:
            self._update_window_history()
        if self.sliding == 'Fixed':
            self.start_time = self.start_time + self.sliding_size
        else:
            self.start_time = self.current_time
        self._delete_data_points()
        self.end_time = self.start_time + self.window_size
        self.window_idx += 1
        self.current_window = 'window_' + str(self.window_idx)

    def _delete_data_points(self):
        """ Delete old data points from the window """
        # check data points having timestamp older than allowed in the window
        check = [i for i, result in enumerate(self._point_timestamps < self.start_time) if result]
        if check:
            index = check[-1] + 1
            self._X_queue = self._X_queue[index:, :]
            self._y_queue = self._y_queue[index:, :]
            self._point_timestamps = self._point_timestamps[index:]
            self.uid = self.uid[index:]

    def check_last_window(self):
        if self.keep_history:
            self._update_window_history()

    def _update_window_history(self):
        if not self.overlap_windows:
            self._update_window_hist_no_overlap()
        else:
            self._update_window_hist_overlap()

    def _update_window_hist_overlap(self):
        max_idx = len(self.overlap_windows)
        if self.window_idx < max_idx:
            for i in range(max_idx):
                buf = self.overlap_windows[i]
                if buf:
                    self._add_window_hist(i, buf)
                    logging.info("writing data point to w_history {}".format(i))
        else:
            curw = 'window_' + str(self.window_idx)
            if curw not in self.w_history.keys():
                temp = [item.start_time for item in self.overlap_windows]
                i = temp.index(self.start_time)
                self._add_window_hist(self.window_idx, self.overlap_windows[i])
            else:
                wrange = self.window_idx - max_idx
                for i in range(max_idx):
                    for idx in range(wrange + 1, self.window_idx + 1):
                        if idx % max_idx == i:
                            self._add_window_hist(idx, self.overlap_windows[i])

    def _add_window_hist(self, idx, buf):
        self.w_history['window_' + str(idx)] = {'X_queue': buf.members['X'],
                                                'y_queue': buf.members['y'],
                                                'timestamps': buf.members['timestamps'],
                                                'nsamples': buf.size,
                                                'uid': buf.members['uids'],
                                                'start_time': buf.start_time,
                                                'end_time': buf.end_time}

    def _update_window_hist_no_overlap(self):
        self.w_history['window_' + str(self.window_idx)] = {'X_queue': self._X_queue,
                                                            'y_queue': self._y_queue,
                                                            'timestamps': self._point_timestamps,
                                                            'nsamples': self._X_queue.shape[0],
                                                            'uid': self.uid,
                                                            'start_time': self.start_time,
                                                            'end_time': self.end_time}

    def features_buffer(self, window_idx):
        """ Get the features buffer
        """
        if self.keep_history:
            return self.w_history[window_idx]['X_queue']

    def targets_buffer(self, window_idx):
        """ Get the latest targets buffer
        """
        if self.keep_history:
            return self.w_history[window_idx]['y_queue']

    def timestamps_buffer(self, window_idx):
        """ Get the timestamps buffer.
        """
        if self.keep_history:
            return self.w_history[window_idx]['timestamps']

    @property
    def n_targets(self):
        """ Get the number of targets. """
        return self._n_targets

    @property
    def n_features(self):
        """ Get the number of features. """
        return self._n_features

    def size(self, window_idx):
        """ Get the window size. """
        if self.keep_history:
            return self.w_history[window_idx]['nsamples']
