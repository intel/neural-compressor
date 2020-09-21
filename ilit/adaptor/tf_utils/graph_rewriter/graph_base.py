#
#  -*- coding: utf-8 -*-
#

from abc import abstractmethod
import logging


class GraphRewriterBase(object):
    """Graph Rewrite Base class.
    We abstract this base class and define the interface only.

    Args:
        object (model): the input model to be converted.
    """
    def __init__(self, model):
        self.model = model
        self.logger = logging.getLogger()

    @abstractmethod
    def do_transformation(self):
        """Base Interface that need to be implemented by each sub class.
        """
        raise NotImplementedError
