"""
ATIO -- All Trains in One
"""
from .singleTask.augment_schedule import AugmentModel

__all__ = ['ATIO']


class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            'camd': AugmentModel,
        }

    def getTrain(self, args):
        return self.TRAIN_MAP[args['model_name']](args)
