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


class MultipleDataIter(DataIter):
    def __init__(self, rec_path, batch_size, num_parts, image_shape, data_nthread):
        self.batch_size = batch_size
        self.num_parts = num_parts
        self.image_shape = image_shape
        self.iters = self.getMultipleIter(rec_path, batch_size, num_parts, image_shape, data_nthread)

    def getMultipleIter(self, rec_path, batch_size, num_parts, image_shape, data_nthreads):
        train_iters = []
        for i in xrange(num_parts):
            train_iters.append(mx.io.ImageRecordIter(
                path_imgrec=rec_path,
                label_width=1,
                data_name='data',
                label_name='softmax_label',
                data_shape=image_shape,
                batch_size=batch_size//num_parts,
                pad=0,
                fill_value=127,
                random_resized_crop=True,
                max_random_area=1.0,
                min_random_area=0.08,
                max_aspect_ratio=4.0 / 3.0,
                min_aspect_ratio=3.0 / 4.0,
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                mean_r=123.68,
                mean_g=116.28,
                mean_b=103.53,
                std_r=58.395,
                std_g=57.12,
                std_b=57.375,
                pca_noise=0.1,
                scale=1,
                inter_method=2,
                rand_mirror=True,
                shuffle=True,
                shuffle_chunk_size=4096,
                preprocess_threads=data_nthreads,
                prefetch_buffer=16,
                num_parts=num_parts,
                part_index=i))
        return train_iters

    def mergeIters(self):
        dataBatchs = []
        for iter_elem in self.iters:
            dataBatchs.append(next(iter_elem))
        total_data = dataBatchs[0].data[0]
        total_label = dataBatchs[0].label[0]
        if len(dataBatchs) > 1:
            for i in xrange(1, len(dataBatchs)):
                total_data = mx.nd.concat(total_data, dataBatchs[i].data[0], dim=0)
                total_label = mx.nd.concat(total_label, dataBatchs[i].label[0], dim=0)
        total_data_batch = DataBatch(data=(total_data,),
                                     label=(total_label,),
                                     pad=0,
                                     index=None,
                                     provide_data=self.provide_data,
                                     provide_label=self.provide_label)
        return total_data_batch


    def __iter__(self):
        return self

    @property
    def provide_data(self):
        return [mx.io.DataDesc('data', (self.batch_size,) + self.image_shape, np.float32)]

    @property
    def provide_label(self):
        return [mx.io.DataDesc('softmax_label', (self.batch_size,), np.float32)]

    def next(self):
        dataBatchs = []
        for iter_elem in self.iters:
            dataBatchs.append(next(iter_elem))
        total_data = dataBatchs[0].data[0]
        total_label = dataBatchs[0].label[0]
        if len(dataBatchs) > 1:
            for i in xrange(1, len(dataBatchs)):
                total_data = mx.nd.concat(total_data, dataBatchs[i].data[0], dim=0)
                total_label = mx.nd.concat(total_label, dataBatchs[i].label[0], dim=0)
        total_data_batch = DataBatch(data=(total_data,),
                                     label=(total_label,),
                                     pad=0,
                                     index=None,
                                     provide_data=self.provide_data,
                                     provide_label=self.provide_label)
        return total_data_batch

    def __next__(self):
        return self.next()

    def reset(self):
        for iter_elem in self.iters:
            iter_elem.reset()


def imagenet_iterator(data_dir, batch_size, kv, image_shape):
    num_examples = 1281167

    if config.benchmark is not None and config.benchmark is True:
        data_shape = (batch_size,) + image_shape
        train = SyntheticDataIter(config.num_classes, data_shape, 5005, np.float32)
        return (train, None, num_examples)

    train = mx.io.ImageRecordIter(
            path_imgrec         = os.path.join(data_dir, "train.rec"),
            label_width         = 1,
            data_name           = 'data',
            label_name          = 'softmax_label',
            data_shape          = image_shape,
            batch_size          = batch_size,
            pad                 = 0,
            fill_value          = 127,
            random_resized_crop = True,
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
            preprocess_threads  = config.data_nthreads,
            prefetch_buffer     = 16,
            num_parts           = kv.num_workers,
            part_index          = kv.rank)
    
    val = mx.io.ImageRecordIter(
            path_imgrec         = os.path.join(data_dir, "val.rec"),
            label_width         = 1,
            data_name           = 'data',
            label_name          = 'softmax_label',
            resize              = 256,
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
            preprocess_threads  = 8,
            num_parts           = kv.num_workers,
            part_index          = kv.rank)

    return train, val, num_examples


def multiple_imagenet_iterator(data_dir, batch_size, num_parts, image_shape, data_nthread):
    num_examples = 1281167
    train = MultipleDataIter(os.path.join(data_dir, "train.rec"), batch_size, num_parts, image_shape, data_nthread)
    val = mx.io.ImageRecordIter(
            path_imgrec         = os.path.join(data_dir, "val.rec"),
            label_width         = 1,
            data_name           = 'data',
            label_name          = 'softmax_label',
            resize              = 256,
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
            preprocess_threads  = 8,
            num_parts           = 1,
            part_index          = 0)
    return train, val, num_examples
