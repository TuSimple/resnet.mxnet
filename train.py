import logging, os

# import config
from config.edict_config import config
import mxnet as mx
from core.scheduler import multi_factor_scheduler
from core.solver import Solver
from core.callback import DetailSpeedometer
from data import *
from symbol import *

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

    if config.network == 'resnet' or config.network == 'resnext' or config.network == 'resnext_cyt':
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

    # train
    epoch_size = max(int(num_examples / config.batch_size / kv.num_workers), 1)
    if 'dist' in config.kv_store and not 'async' in config.kv_store:
        logging.info('Resizing training data to %d batches per machine', epoch_size)
        # resize train iter to ensure each machine has same number of batches per epoch
        # if not, dist_sync can hang at the end with one machine waiting for other machines
        train = mx.io.ResizeIter(train, epoch_size)

    if config.warmup is not None and config.warmup is True:
        lr_epoch = [int(epoch) for epoch in config.lr_step]
        lr_epoch_diff = [epoch - config.begin_epoch for epoch in lr_epoch if epoch > config.begin_epoch]
        lr = config.lr * (config.lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
        lr_iters = [int(epoch * epoch_size) for epoch in lr_epoch_diff]
        print 'lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters
        print 'warmup lr', config.warmup_lr, 'warm_epoch', config.warm_epoch, 'warm_step', int(config.warm_epoch * epoch_size)
        lr_scheduler = WarmupMultiFactorScheduler(base_lr=lr, step=lr_iters, factor=config.lr_factor,
                                                  warmup=True, warmup_type='gradual',
                                                  warmup_lr=config.warmup_lr, warmup_step=int(config.warm_epoch * epoch_size))
    elif config.lr_step is not None:
        lr_scheduler = multi_factor_scheduler(config.begin_epoch, epoch_size, step=config.lr_step,
                                              factor=config.lr_factor)
    else:
        lr_scheduler = None
    optimizer_params = {'learning_rate': config.lr,
                        'lr_scheduler': lr_scheduler,
                        'wd': config.wd,
                        'momentum': config.momentum,
                        'multi_precision': config.multi_precision}
    optimizer = "nag"
    #optimizer = 'sgd'
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
    # batch_end_callback = mx.callback.Speedometer(config.batch_size, config.frequent)
    batch_end_callback = DetailSpeedometer(config.batch_size, config.frequent)
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
               optimizer=optimizer,
               optimizer_params=optimizer_params,
               begin_epoch=config.begin_epoch,
               num_epoch=config.num_epoch,
               kvstore=kv)


if __name__ == '__main__':
    pprint.pprint(config)
    main(config)
