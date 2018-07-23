import sys
import config

sys.path.insert(0, config.mxnet_path)
import mxnet as mx
import numpy as np
eps = 1e-5


def xresidual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True,
                   num_group=32, bn_mom=0.9, workspace=256, memonger=False):
    if num_group == 32:
        multip_factor = 0.5
    elif num_group == 64:
        multip_factor = 1.0
    if bottle_neck:
        conv1 = mx.sym.Convolution(data=data, num_filter=int(num_filter * multip_factor), kernel=(1, 1), stride=(1, 1),
                                   pad=(0, 0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')

        conv2 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * multip_factor), num_group=num_group, kernel=(3, 3),
                                   stride=stride, pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=eps, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')

        conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=eps, momentum=bn_mom, name=name + '_bn3')

        if dim_match:
            shortcut = data
        else:
            shortcut_conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride,
                                               no_bias=True,
                                               workspace=workspace, name=name + '_sc')
            shortcut = mx.sym.BatchNorm(data=shortcut_conv, fix_gamma=False, eps=eps, momentum=bn_mom,
                                        name=name + '_sc_bn')

        if memonger:
            shortcut._set_attr(mirror_stage='True')
        eltwise = bn3 + shortcut
        return mx.sym.Activation(data=eltwise, act_type='relu', name=name + '_relu')
    else:
        conv1 = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=eps, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')

        conv2 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=eps, name=name + '_bn2')

        if dim_match:
            shortcut = data
        else:
            shortcut_conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride,
                                               no_bias=True,
                                               workspace=workspace, name=name + '_sc')
            shortcut = mx.sym.BatchNorm(data=shortcut_conv, fix_gamma=False, eps=eps, momentum=bn_mom,
                                        name=name + '_sc_bn')

        if memonger:
            shortcut._set_attr(mirror_stage='True')
        eltwise = bn2 + shortcut
        return mx.sym.Activation(data=eltwise, act_type='relu', name=name + '_relu')


def resnext(units, num_stage, filter_list, num_classes, data_type, num_group=32,
            bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    num_unit = len(units)
    assert (num_unit == num_stage)

    data = mx.sym.Variable(name='data')
    if data_type == 'float32':
        data = mx.sym.identity(data=data, name='id')
    elif data_type == 'float16':
        data = mx.sym.Cast(data=data, dtype=np.float16)

    conv1_1 = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                 no_bias=True, name="conv1_1", workspace=workspace)
    conv1_1 = mx.sym.BatchNorm(data=conv1_1, fix_gamma=False, eps=eps, momentum=bn_mom, name='bn1_1')
    conv1_1 = mx.sym.Activation(data=conv1_1, act_type='relu', name='relu1_1')

    conv1_2 = mx.sym.Convolution(data=conv1_1, num_filter=filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                 no_bias=True, name="conv1_2", workspace=workspace)
    conv1_2 = mx.sym.BatchNorm(data=conv1_2, fix_gamma=False, eps=eps, momentum=bn_mom, name='bn1_2')
    conv1_2 = mx.sym.Activation(data=conv1_2, act_type='relu', name='relu1_2')

    conv1_3 = mx.sym.Convolution(data=conv1_2, num_filter=filter_list[0], kernel=(3, 3), stride=(2, 2), pad=(1, 1),
                                 no_bias=True, name="conv1_3", workspace=workspace)
    conv1_3 = mx.sym.BatchNorm(data=conv1_3, fix_gamma=False, eps=eps, momentum=bn_mom, name='bn1_3')
    conv1_3 = mx.sym.Activation(data=conv1_3, act_type='relu', name='relu1_3')
    body = mx.symbol.Pooling(data=conv1_3, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')

    for i in range(num_stage):
        body = xresidual_unit(body, filter_list[i + 1], (1 if i == 0 else 2, 1 if i == 0 else 2), False,
                              name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, num_group=num_group,
                              bn_mom=bn_mom, workspace=workspace, memonger=memonger)
        for j in range(units[i] - 1):
            body = xresidual_unit(body, filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                  bottle_neck=bottle_neck, num_group=num_group, bn_mom=bn_mom,
                                  workspace=workspace, memonger=memonger)
    pool1 = mx.symbol.Pooling(data=body, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_classes, name='fc1')
    cls = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return cls
