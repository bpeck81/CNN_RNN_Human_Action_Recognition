import numpy as np
from .data_loader import *
from torch.utils.data.sampler import SubsetRandomSampler


# Follow this class convention to add a new dataset.
class HMDB(VideoLoader):
    def __init__(self, seq_len, transforms = None, root_dir='/home/user/'):
        self.title = 'HMDB'
        super(HMDB, self).__init__('hmdb', transforms, root_dir, seq_len)
