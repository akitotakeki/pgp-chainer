#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
from models_imagenet.resnet_fb import ResNet_fb
from models_imagenet.resnet_fb_dconv import ResNet_fb_DConv
from models_imagenet.resnet_fb_pgp import ResNet_fb_PGP
from models_imagenet.shufflenetv2 import ShuffleNetV2
from models_imagenet.shufflenetv2_pgp import ShuffleNetV2_PGP


model_dict = {
    'ResNet50_fb': lambda args: L.Classifier(ResNet_fb(50, args.nclasses)),
    'ResNet50_fb_DConv': lambda args: L.Classifier(ResNet_fb_DConv(50, args.nclasses)),
    'ResNet50_fb_PGP': lambda args: FuseTrainWrapper(
        ResNet_fb_PGP(50, args.nclasses)),
    'ResNet101_fb': lambda args: L.Classifier(ResNet_fb(101, args.nclasses)),
    'ResNet101_fb_DConv': lambda args: L.Classifier(ResNet_fb_DConv(101, args.nclasses)),
    'ResNet101_fb_PGP': lambda args: FuseTrainWrapper(
        ResNet_fb_PGP(101, args.nclasses)),
    'ShuffleNetV2_0.5x': lambda args: L.Classifier(ShuffleNetV2(50, args.nclasses, 0.5)),
    'ShuffleNetV2_0.5x_PGP': lambda args: FuseTrainWrapper(
        ShuffleNetV2_PGP(50, args.nclasses, 0.5), n_kernel=256),
    'ShuffleNetV2': lambda args: L.Classifier(ShuffleNetV2(50, args.nclasses)),
}


class FuseTrainWrapper(chainer.Chain):

    def __init__(self, predictor, n_kernel=16, mode=None, sm_fuse=False):
        super().__init__()
        with self.init_scope():
            self.predictor = predictor
        self.n_kernel = n_kernel
        self.sm_fuse = sm_fuse

    def __call__(self, x, t):
        y_list = self.predictor(x)
        _len, _cls = y_list.shape
        if self.sm_fuse:
            _sm = F.reshape(F.log_softmax(y_list), (self.n_kernel, _len // self.n_kernel, _cls))
            ave_y = F.average(_sm, axis=0)
            loss = - F.average(F.select_item(ave_y, t))
        else:
            loss = F.average(F.softmax_cross_entropy(y_list, F.tile(t, self.n_kernel)))

        conf = F.average(
            F.reshape(y_list, (self.n_kernel, _len // self.n_kernel, _cls)), axis=0)
        chainer.report(
            {'loss': loss, 'accuracy': F.accuracy(conf, t)}, self)
        return loss
