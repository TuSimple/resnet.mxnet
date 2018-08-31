import logging, os
import argparse
# import config
from config.edict_config import config
import mxnet as mx
from core.scheduler import multi_factor_scheduler
from core.solver import Solver
from core.memonger_v2 import search_plan_to_layer
from core.callback import DetailSpeedometer
from data import *
from symbol import *
import datetime
import pprint
from core.scheduler import WarmupMultiFactorScheduler


def main(config):
    # log file
    log_dir = "./log"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='{}/{}.log'.format(log_dir, config.model_prefix),
                        filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s %(levelname)s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    # model folder
    model_dir = "./model"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # set up environment
    devs = [mx.gpu(int(i)) for i in config.gpu_list]
    kv = mx.kvstore.create(config.kv_store)

    # set up iterator and symbol
    # iterator
    if config.use_multiple_iter is True:
        train, val, num_examples = multiple_imagenet_iterator(data_dir=config.data_dir,
                                                    batch_size=config.batch_size,
                                                    num_parts=2,
                                                    image_shape=tuple(config.image_shape),
                                                    data_nthread=config.data_nthreads)
    # elif config.use_dali_iter is True:
    #     train, val, num_examples = get_dali_iter(data_dir=config.data_dir,
    #                                              batch_size=config.batch_size,
    #                                              kv=kv,
    #                                              image_shape=tuple(config.image_shape),
    #                                              num_gpus=len(devs))
    else:
        train, val, num_examples = imagenet_iterator(data_dir=config.data_dir,
                                                     batch_size=config.batch_size,
                                                     kv=kv,
                                                     image_shape=tuple(config.image_shape))
    print train
    print val
    data_names = ('data',)
    label_names = ('softmax_label',)
    data_shapes = [('data', tuple([config.batch_size] + config.image_shape))]
    label_shapes = [('softmax_label', (config.batch_size,))]

    if config.network == 'resnet':
        symbol = eval(config.network)(units=config.units,
                                      num_stage=config.num_stage,
                                      filter_list=config.filter_list,
                                      num_classes=config.num_classes,
                                      data_type=config.data_type,
                                      bottle_neck=config.bottle_neck,
                                      grad_scale=config.grad_scale,
                                      memonger=config.memonger)
    elif config.network == 'resnet_mxnet':
        symbol = eval(config.network)(units=config.units,
                                      num_stage=config.num_stage,
                                      filter_list=config.filter_list,
                                      num_classes=config.num_classes,
                                      data_type=config.data_type,
                                      bottle_neck=config.bottle_neck,
                                      grad_scale=config.grad_scale)
    elif config.network == 'resnext' or config.network == 'resnext_cyt':
        symbol = eval(config.network)(units=config.units,
                                      num_stage=config.num_stage,
                                      filter_list=config.filter_list,
                                      num_classes=config.num_classes,
                                      data_type=config.data_type,
                                      num_group=config.num_group,
                                      bottle_neck=config.bottle_neck)
    elif config.network == 'vgg16' or config.network == 'mobilenet' or config.network == 'shufflenet':
        symbol = eval(config.network)(num_classes=config.num_classes)

    # mx.viz.print_summary(symbol, {'data': (1, 3, 224, 224)})

    # memonger
    if config.memonger:
        # infer shape
        data_shape_dict = dict(train.provide_data + train.provide_label)
        per_gpu_data_shape_dict = {}
        for key in data_shape_dict:
            per_gpu_data_shape_dict[key] = (config.batch_per_gpu,) + data_shape_dict[key][1:]

        # if config.network == 'resnet':
        #     last_block = 'conv3_1_relu'
        #     if kv.rank == 0:
        #         print("resnet do memonger up to {}".format(last_block))
        # else:
        #     last_block = None
        last_block = 'stage4_unit1_sc'
        input_dtype = {k: 'float32' for k in per_gpu_data_shape_dict}
        symbol = search_plan_to_layer(symbol, last_block, 1000, type_dict=input_dtype, **per_gpu_data_shape_dict)

    # train
    epoch_size = max(int(num_examples / config.batch_size / kv.num_workers), 1)
    if 'dist' in config.kv_store and not 'async' in config.kv_store \
            and config.use_multiple_iter is False and config.use_dali_iter is False:
        logging.info('Resizing training data to %d batches per machine', epoch_size)
        # resize train iter to ensure each machine has same number of batches per epoch
        # if not, dist_sync can hang at the end with one machine waiting for other machines
        train = mx.io.ResizeIter(train, epoch_size)

    if config.warmup is not None and config.warmup is True:
        lr_epoch = [int(epoch) for epoch in config.lr_step]
        lr_epoch_diff = [epoch - config.begin_epoch for epoch in lr_epoch if epoch > config.begin_epoch]
        lr = config.lr * (config.lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
        lr_iters = [int(epoch * epoch_size) for epoch in lr_epoch_diff]
        print 'warmup lr', config.warmup_lr, 'warm_epoch', config.warm_epoch, 'warm_step', int(config.warm_epoch * epoch_size)

        if config.lr_scheduler == 'poly':
            print 'PolyScheduler lr', lr
            lr_scheduler = mx.lr_scheduler.PolyScheduler(int(epoch_size*config.num_epoch), base_lr=lr, pwr=2, final_lr=0,
                                                         warmup_steps=int(config.warm_epoch * epoch_size),
                                                         warmup_begin_lr=0, warmup_mode='linear')
        else:
            print 'WarmupMultiFactorScheduler lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters
            lr_scheduler = WarmupMultiFactorScheduler(base_lr=lr, step=lr_iters, factor=config.lr_factor,
                                                  warmup=True, warmup_type='gradual',
                                                  warmup_lr=config.warmup_lr, warmup_step=int(config.warm_epoch * epoch_size))
    elif config.lr_step is not None:
        lr_scheduler = multi_factor_scheduler(config.begin_epoch, epoch_size, step=config.lr_step,
                                              factor=config.lr_factor)
    else:
        lr_scheduler = None

    optimizer_params = {
        'learning_rate': lr,
        'wd': config.wd,
        'lr_scheduler': lr_scheduler,
        'multi_precision': config.multi_precision}
    # Only a limited number of optimizers have 'momentum' property
    has_momentum = {'sgd', 'dcasgd', 'nag', 'signum', 'lbsgd'}
    if config.optimizer in has_momentum:
        optimizer_params['momentum'] = config.momentum
    # A limited number of optimizers have a warmup period
    has_warmup = {'lbsgd', 'lbnag'}
    if config.optimizer in has_warmup:
        optimizer_params['updates_per_epoch'] = epoch_size
        optimizer_params['begin_epoch'] = config.begin_epoch
        optimizer_params['batch_scale'] = 1.0
        optimizer_params['warmup_strategy'] = 'lars'
        optimizer_params['warmup_epochs'] = config.warm_epoch # not work whne warmup_strategy is 'lars'
        optimizer_params['num_epochs'] = config.num_epoch

    eval_metric = ['acc']
    if config.dataset == "imagenet":
        eval_metric.append(mx.metric.create('top_k_accuracy', top_k=5))

    solver = Solver(symbol=symbol,
                    data_names=data_names,
                    label_names=label_names,
                    data_shapes=data_shapes,
                    label_shapes=label_shapes,
                    logger=logging,
                    context=devs)
    epoch_end_callback = mx.callback.do_checkpoint("./model/" + config.model_prefix)
    batch_end_callback = mx.callback.Speedometer(config.batch_size, config.frequent)
    # batch_end_callback = DetailSpeedometer(config.batch_size, config.frequent)
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2)
    arg_params = None
    aux_params = None
    if config.retrain:
        _, arg_params, aux_params = mx.model.load_checkpoint("model/{}".format(config.model_load_prefix),
                                                             config.model_load_epoch)
    solver.fit(train_data=train,
               eval_data=val,
               eval_metric=eval_metric,
               epoch_end_callback=epoch_end_callback,
               batch_end_callback=batch_end_callback,
               initializer=initializer,
               arg_params=arg_params,
               aux_params=aux_params,
               optimizer=config.optimizer,
               optimizer_params=optimizer_params,
               begin_epoch=config.begin_epoch,
               num_epoch=config.num_epoch,
               kvstore=kv)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN network')
    # general
    parser.add_argument('--network', help='network name', default=config.network, type=str)
    parser.add_argument('--dataset', help='dataset name', default=config.dataset, type=str)
    parser.add_argument('--dataset_dir', help='dataset path', default=config.dataset_dir, type=str)
    # training
    parser.add_argument('--frequent', help='frequency of logging', default=config.frequent, type=int)
    parser.add_argument('--kv_store', help='the kv-store type', default=config.kv_store, type=str)
    parser.add_argument('--resume', help='continue training', action='store_true')
    parser.add_argument('--gpus', help='GPU device to train with', default='-1', type=str)
    parser.add_argument('--model_prefix', help='pretrained model prefix', default=config.model_prefix, type=str)
    parser.add_argument('--model_load_epoch', help='pretrained model epoch', default=config.model_load_epoch, type=int)
    parser.add_argument('--begin_epoch', help='begin epoch of training, use with resume', default=config.begin_epoch, type=int)
    parser.add_argument('--num_epoch', help='end epoch of training', default=config.num_epoch, type=int)
    parser.add_argument('--lr', help='base learning rate', default=config.lr, type=float)
    parser.add_argument('--lr_scheduler', help='lr scheduler', default=config.lr_scheduler, type=str)
    parser.add_argument('--optimizer', help='optimizer', default=config.optimizer, type=str)
    parser.add_argument('--data_type', help='data type', default=config.data_type, type=str)
    parser.add_argument('--grad_scale', help='grad scale for fp16', default=config.grad_scale, type=float)
    parser.add_argument('--batch_per_gpu', help='batch size per gpu', default=config.batch_per_gpu, type=int)
    # memory
    parser.add_argument('--memonger', help='use memonger to put more images on a single GPU', default=config.memonger, type=int)
    args = parser.parse_args()
    return args

def set_config(args):
    config.network = args.network
    config.dataset = args.dataset
    config.dataset_dir = args.dataset_dir
    config.frequent = args.frequent
    config.kv_store = args.kv_store
    if args.resume:
        config.retrain = True
    if args.gpus != '-1':
        config.gpu_list = [mx.gpu(int(devs_id)) for devs_id in args.gpus.split(',')]
    config.model_prefix = args.model_prefix
    config.model_load_epoch = args.model_load_epoch
    config.begin_epoch = args.begin_epoch
    config.num_epoch = args.num_epoch
    config.lr = args.lr
    config.lr_scheduler = args.lr_scheduler
    config.optimizer = args.optimizer
    config.data_type = args.data_type
    config.grad_scale = args.grad_scale
    config.batch_per_gpu = args.batch_per_gpu
    config.batch_size = config.batch_per_gpu * len(config.gpu_list)
    config.memonger = args.memonger


if __name__ == '__main__':
    args = parse_args()
    print 'Called with argument:', args
    now = datetime.datetime.now()
    date = '{}_{:0>2}_{:0>2}'.format(now.year, now.month, now.day)

    set_config(args)

    pprint.pprint(config)
    main(config)
