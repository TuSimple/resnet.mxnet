import os
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.mxnet import DALIClassificationIterator

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus, db_folder):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.MXNetReader(path=[os.path.join(db_folder, "train.rec")], index_path=[os.path.join(db_folder, "train.idx")],
                                     random_shuffle=True, shard_id=device_id, num_shards=num_gpus)
        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        self.rrc = ops.RandomResizedCrop(device="gpu", size=(224, 224))
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(224, 224),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name = "Reader")
        images = self.decode(self.jpegs)
        images = self.rrc(images)
        output = self.cmnp(images, mirror = rng)
        return [output, self.labels]

class  HybridValPipe (Pipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus, db_folder):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.MXNetReader(path=[os.path.join(db_folder, "val.rec")], index_path=[os.path.join(db_folder, "val.idx")],
                                     random_shuffle=False, shard_id=device_id, num_shards=num_gpus)
        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        self.rs = ops.Resize(device="gpu", resize_shorter=256)
        # self.rrc = ops.RandomResizedCrop(device="gpu", size=(224, 224))
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(224, 224),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.rs(images)
        # images = self.rrc(images)
        output = self.cmnp(images)
        return [output, self.labels]


def get_dali_iter(data_dir, batch_size, kv, image_shape, num_gpus):
    num_examples = 1281167
    trainpipes = [HybridTrainPipe(batch_size=batch_size//num_gpus, num_threads=2, device_id=i, num_gpus=num_gpus, db_folder=data_dir) for i in range(num_gpus)]
    valpipes = [HybridValPipe(batch_size=batch_size//num_gpus, num_threads=2, device_id=i, num_gpus=num_gpus, db_folder=data_dir) for i in range(num_gpus)]

    trainpipes[0].build()
    valpipes[0].build()

    print("Training pipeline epoch size: {}".format(trainpipes[0].epoch_size("Reader")))
    print("Validation pipeline epoch size: {}".format(valpipes[0].epoch_size("Reader")))

    dali_train_iter = DALIClassificationIterator(trainpipes, trainpipes[0].epoch_size("Reader"))
    dali_val_iter = DALIClassificationIterator(valpipes, valpipes[0].epoch_size("Reader"))

    return dali_train_iter, dali_val_iter, num_examples

