import tensorflow as tf

class TFSmoothQuant:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader

    def _smooth_calibrate(self, calib_iter):
        """calibrate the graph, find the input (activation) max value per channel of each op (e.g. matmul)."""
        input_maxes_per_channel = {}
        # here add your code
        # ...
        #
        return input_maxes_per_channel
    
    def _adjust_parameters(self, input_maxes_per_channel):
        """calculate the smooth scale => transform the w/a with the smooth scale"""
        pass

    def recover(self):
        """Recover the model weights."""
        pass
    
    def smooth_transform(self, alpha=0.5, calib_iter=100):
        """calib to get max per channel => calculate the smooth scale => transform the w/a with the smooth scale"""
        input_maxes_per_channel = self._smooth_calibrate(calib_iter=calib_iter)
        self._adjust_parameters(input_maxes_per_channel)
        return self.model