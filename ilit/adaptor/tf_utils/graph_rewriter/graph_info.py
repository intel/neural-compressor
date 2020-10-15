#
#  -*- coding: utf-8 -*-
#


class TFLowbitPrecisionPatterns(object):
    patterns = {
        "2.1.0": [[["ConcatV2"]], [["MaxPool", "AvgPool"]], [["MatMul"], ("BiasAdd")],
                  [["Conv2D", "DepthwiseConv2dNative"], ("BiasAdd"), ("Add", "AddN", "AddV2"),
                   ("Relu", "Relu6")]],
        "default": [[["ConcatV2"]], [["MaxPool", "AvgPool"]], [["MatMul"], ("BiasAdd")],
                    [["Conv2D", "DepthwiseConv2dNative"], ("BiasAdd"), ("Add", "AddN", "AddV2"),
                     ("Relu", "Relu6")]],
    }

    def __init__(self, version):
        self.version = version

    def get_supported_patterns(self):
        """Get the supported pattern list

        Returns:
            [string list]: patterns list
        """
        if self.version not in self.patterns:
            return self.patterns['default']
        return self.patterns[self.version]
