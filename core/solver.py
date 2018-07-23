import time
import logging

import mxnet as mx
from mxnet.module import Module
from mxnet import metric
#from mxnet.model import BatchEndParam
from callback import BatchEndParam

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def _as_list(obj):
    if isinstance(obj, list):
        return obj
    else:
        return [obj]


class Solver(object):
    def __init__(self, symbol, data_names, label_names,
                 data_shapes, label_shapes, logger=logging,
                 context=mx.cpu(), work_load_list=None, fixed_param_names=None):
        self.symbol = symbol
        self.data_names = data_names
        self.label_names = label_names
        self.data_shapes = data_shapes
        self.label_shapes = label_shapes
        self.context = context
        self.work_load_list = work_load_list
        self.fixed_param_names = fixed_param_names

        if logger is None:
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
        self.logger = logger
        self.module = Module(symbol=self.symbol, data_names=self.data_names,
                             label_names=self.label_names, logger=self.logger,
                             context=self.context, work_load_list=self.work_load_list,
                             fixed_param_names=self.fixed_param_names)

    def fit(self, train_data, eval_data=None,
            eval_metric='acc', validate_metric=None,
            work_load_list=None, epoch_end_callback=None,
            batch_end_callback=None, fixed_param_prefix=None,
            initializer=None, arg_params=None,
            aux_params=None, allow_missing=False,
            optimizer=None, optimizer_params=None,
            begin_epoch=0, num_epoch=None,
            kvstore='device'):

        self.module.bind(data_shapes=self.data_shapes, label_shapes=self.label_shapes, for_training=True)
        self.module.init_params(initializer=initializer,
                                arg_params=arg_params,
                                aux_params=aux_params,
                                allow_missing=allow_missing)
        self.module.init_optimizer(kvstore=kvstore,
                                   optimizer=optimizer,
                                   optimizer_params=optimizer_params)

        if validate_metric is None:
            validate_metric = eval_metric
        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)

        temp_count = 0
        # training loop
        for epoch in range(begin_epoch, num_epoch):

            train_time = AverageMeter()
            kvstore_sync_time = AverageMeter()
            get_data_time = AverageMeter()
            iter_total_time = AverageMeter()

            tic = time.time()
            eval_metric.reset()
            nbatch = 0
            data_iter = iter(train_data)
            end_of_batch = False
            next_data_batch = next(data_iter)
            while not end_of_batch:
            # while temp_count <= 1000:
                # ndarray.waitall()
                start_time = time.time()
                data_batch = next_data_batch

                self.module.forward(data_batch, is_train=True)
                self.module.backward()

                # ndarray.waitall()
                train_time.update(time.time() - start_time)

                self.module.update()

                # ndarray.waitall()
                kvstore_sync_time.update(time.time() - start_time)

                try:
                    next_data_batch = next(data_iter)
                except StopIteration:
                    end_of_batch = True

                # ndarray.waitall()
                get_data_time.update(time.time() - start_time)

                self.module.update_metric(eval_metric, data_batch.label)

                # ndarray.waitall()
                iter_total_time.update(time.time() - start_time)

                if batch_end_callback is not None:
                    # batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                    #                                  eval_metric=eval_metric,
                    #                                  locals=locals())

                    batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                     eval_metric=eval_metric,
                                                     locals=locals(),
                                                     rank=kvstore.rank, total_iter=temp_count,
                                                     cur_data_time=get_data_time.val, avg_data_time=get_data_time.avg,
                                                     cur_batch_time=train_time.val, avg_batch_time=train_time.avg,
                                                     cur_kvstore_sync_time=kvstore_sync_time.val,
                                                     avg_kvstore_sync_time=kvstore_sync_time.avg,
                                                     cur_iter_total_time=iter_total_time.val,
                                                     avg_iter_total_time=iter_total_time.avg
                                                     )
                    for callback in _as_list(batch_end_callback):
                        callback(batch_end_params)
                nbatch += 1
                temp_count += 1

            for name, val in eval_metric.get_name_value():
                self.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc - tic))

            arg_params, aux_params = self.module.get_params()
            self.module.set_params(arg_params, aux_params)

            if epoch_end_callback is not None and kvstore.rank == 0:
                for callback in _as_list(epoch_end_callback):
                    callback(epoch, self.symbol, arg_params, aux_params)
            if eval_data:
                res = self.module.score(eval_data, validate_metric,
                                        score_end_callback=None,
                                        batch_end_callback=None,
                                        reset=True,
                                        epoch=epoch)
                for name, val in res:
                    self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)

            train_data.reset()
