from easydict import EasyDict as edict

config = edict()

# mxnet version: https://github.com/huangzehao/incubator-mxnet-bk
config.mxnet_path = '../mxnet/python/'
config.gpu_list = [0, 1, 2, 3, 4, 5, 6, 7]
config.dataset = "imagenet"
config.model_prefix = "resnet50"
config.network = "resnet"
config.depth = 50
config.model_load_prefix = config.model_prefix
config.model_load_epoch = 0
config.retrain = False

# data
config.data_dir = '/data/ILSVRC2012'
config.batch_per_gpu = 64
config.batch_size = config.batch_per_gpu * len(config.gpu_list)
config.kv_store = 'local'

# optimizer
config.lr = 3.2
config.wd = 0.0001
config.momentum = 0.9
config.multi_precision = True
if config.dataset == "imagenet":
    config.lr_step = [30, 60, 90]
else:
    config.lr_step = [120, 160, 240]
config.lr_factor = 0.1
config.begin_epoch = config.model_load_epoch if config.retrain else 0
config.num_epoch = 100
config.frequent = 20
# for distributed training
config.warmup = True
config.warmup_lr = 0.1
config.warm_epoch = 5
config.lr_scheduler = 'poly'
config.optimizer = 'sgd'
# set image_shape for io and network
config.image_shape = [3, 224, 224]
config.benchmark = False
config.num_group = 64
config.data_type = 'float32'
config.grad_scale = 1.0
config.data_nthreads = 16
config.use_multiple_iter = False
config.use_dali_iter = False
config.memonger = False





# network config
if config.dataset == "imagenet":
    config.num_classes = 1000
    config.units_dict = {"18": [2, 2, 2, 2],
                  "34": [3, 4, 6, 3],
                  "50": [3, 4, 6, 3],
                  "101": [3, 4, 23, 3],
                  "152": [3, 8, 36, 3]}
    config.units = config.units_dict[str(config.depth)]
    if config.depth >= 50:
        config.filter_list = [64, 256, 512, 1024, 2048]
        config.bottle_neck = True
    else:
        config.filter_list = [64, 64, 128, 256, 512]
        config.bottle_neck = False
    config.num_stage = 4
