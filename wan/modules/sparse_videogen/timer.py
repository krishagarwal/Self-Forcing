import os
from contextlib import ContextDecorator

import torch

ENABLE_LOGGING = int(os.getenv("TIME_BENCH", "0")) >= 1
CLEAR_LOG_DATA = int(os.getenv("TIME_BENCH", "0")) == 2


operator_log_data = {}


def clear_operator_log_data():
    operator_log_data.clear()


class TimeLoggingContext(ContextDecorator):
    def __init__(self, operation_type):
        self.operation_type = operation_type
        self.start_event = None
        self.end_event = None

    def __enter__(self):
        if ENABLE_LOGGING:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if ENABLE_LOGGING:
            self.end_event.record()
            torch.cuda.synchronize()
            duration = self.start_event.elapsed_time(self.end_event)
            if self.operation_type not in operator_log_data:
                operator_log_data[self.operation_type] = 0
            operator_log_data[self.operation_type] += duration


time_logging_decorator = TimeLoggingContext
