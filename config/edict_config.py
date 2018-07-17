from easydict import EasyDict as edict

config = edict()

# mxnet version: https://github.com/huangzehao/incubator-mxnet-bk
config.mxnet_path = '../mxnet/python/'
config.gpu_list = [0, 1, 2, 3, 4, 5, 6, 7]
config.dataset = "imagenet"
config.model_prefix = "resnext50"
config.network = "resnext"
config.depth = 50
config.model_load_prefix = config.model_prefix
config.model_load_epoch = 0
config.retrain = False

# data
config.data_dir = '/data/ILSVRC2012'
config.batch_size = 16
config.batch_size *= len(config.gpu_list)
config.kv_store = 'dist_sync'

# optimizer
config.lr = 0.2
config.wd = 0.0001
config.momentum = 0.9
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
# set image_shape for io and network
config.image_shape = [3, 320, 320]



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
