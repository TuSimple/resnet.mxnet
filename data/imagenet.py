import os
# import config
from config.edict_config import config
import mxnet as mx
from mxnet.io import DataBatch, DataIter
import numpy as np

class SyntheticDataIter(DataIter):
    def __init__(self, num_classes, data_shape, max_iter, dtype):
        self.batch_size = data_shape[0]
        self.cur_iter = 0
        self.max_iter = max_iter
        self.dtype = dtype
        label = np.random.randint(0, num_classes, [self.batch_size,])
        data = np.random.uniform(-1, 1, data_shape)
        self.data = mx.nd.array(data, dtype=self.dtype, ctx=mx.Context('cpu_pinned', 0))
        self.label = mx.nd.array(label, dtype=self.dtype, ctx=mx.Context('cpu_pinned', 0))
    def __iter__(self):
        return self
    @property
    def provide_data(self):
        return [mx.io.DataDesc('data', self.data.shape, self.dtype)]
    @property
    def provide_label(self):
        return [mx.io.DataDesc('softmax_label', (self.batch_size,), self.dtype)]
    def next(self):
        self.cur_iter += 1
        if self.cur_iter <= self.max_iter:
            return DataBatch(data=(self.data,),
                             label=(self.label,),
                             pad=0,
                             index=None,
                             provide_data=self.provide_data,
                             provide_label=self.provide_label)
        else:
            raise StopIteration
    def __next__(self):
        return self.next()
    def reset(self):
        self.cur_iter = 0


def imagenet_iterator(data_dir, batch_size, kv, image_shape):
    if config.benchmark is not None and config.benchmark is True:
        data_shape = (batch_size,) + image_shape
        train = SyntheticDataIter(config.num_classes, data_shape, 500, np.float32)
        return (train, None)

    train = mx.io.ImageRecordIter(
            path_imgrec         = os.path.join(data_dir, "train.rec"),
            label_width         = 1,
            data_name           = 'data',
            label_name          = 'softmax_label',
            data_shape          = image_shape,
            batch_size          = batch_size,
            pad                 = 0,
            fill_value          = 127,
            facebook_aug        = True,
            max_random_area     = 1.0,
            min_random_area     = 0.08,
            max_aspect_ratio    = 4.0 / 3.0,
            min_aspect_ratio    = 3.0 / 4.0,
            brightness          = 0.4,
            contrast            = 0.4,
            saturation          = 0.4,
            mean_r              = 123.68,
            mean_g              = 116.28,
            mean_b              = 103.53,
            std_r               = 58.395,
            std_g               = 57.12,
            std_b               = 57.375,
            pca_noise           = 0.1,
            scale               = 1,
            inter_method        = 2,
            rand_mirror         = True,
            shuffle             = True,
            shuffle_chunk_size  = 4096,
            preprocess_threads  = 8,
            prefetch_buffer     = 16,
            num_parts           = kv.num_workers,
            part_index          = kv.rank)
    
    val = mx.io.ImageRecordIter(
            path_imgrec         = os.path.join(data_dir, "val.rec"),
            label_width         = 1,
            data_name           = 'data',
            label_name          = 'softmax_label',
            resize              = 352,
            batch_size          = batch_size,
            data_shape          = image_shape,
            mean_r              = 123.68,
            mean_g              = 116.28,
            mean_b              = 103.53,
            std_r               = 58.395,
            std_g               = 57.12,
            std_b               = 57.375,
            scale               = 1,
            inter_method        = 2,
            rand_crop           = False,
            rand_mirror         = False,
            num_parts           = kv.num_workers,
            part_index          = kv.rank)
    
    num_examples = 1281167
    return train, val, num_examples
