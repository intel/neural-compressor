import time
import torch
class CPUTimer:
    def __init__(self, timelogs):
        self.timelogs = timelogs

    def __enter__(self):
        self.start = time.time()

    def __exit__(self):
        end = time.time()
        self.timelogs.append((end - self.start) * 1000) # ms
    
    def get_avg_time(self):
        return sum(self.timelogs) / len(self.timelogs)

class GPUTimer:
    def __init__(self, timelogs):
        self.timelogs = timelogs

    def __enter__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()

    def __exit__(self):
        self.end_event.record()
        self.end_event.synchronize()
        elapsed_time = self.start_event.elapsed_time(self.end_event)
        self.timelogs.append(elapsed_time)

    def get_avg_time(self):
        return sum(self.timelogs) / len(self.timelogs)