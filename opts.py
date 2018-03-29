#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
from models.allconv import AllConv
from models.allconv_dconv import AllConv_DConv
from models.allconv_pgp import AllConv_PGP
from models.preresnet import PreResNet
from models.preresnet_dconv import PreResNet_DConv
from models.preresnet_pgp import PreResNet_PGP
from models.wideresnet import WideResNet
from models.wideresnet_dconv import WideResNet_DConv
from models.wideresnet_pgp import WideResNet_PGP
from models.resnext import ResNeXt
from models.resnext_dconv import ResNeXt_DConv
from models.resnext_pgp import ResNeXt_PGP
from models.pyramidnet import PyramidNet
from models.pyramidnet_dconv import PyramidNet_DConv
from models.pyramidnet_pgp import PyramidNet_PGP
from models.densenet import DenseNetBC
from models.densenet_dconv import DenseNetBC_DConv
from models.densenet_pgp import DenseNetBC_PGP
from models.shakeshake import ShakeShake

model_dict = {
    'AllConv': lambda args: L.Classifier(
        AllConv(args.nclasses)),
    'AllConv_DConv': lambda args: L.Classifier(
        AllConv_DConv(args.nclasses)),
    'AllConv_PGP': lambda args: FuseTrainWrapper(
        AllConv_PGP(args.nclasses)),
    'PreResNet164': lambda args: L.Classifier(
        PreResNet(164, args.nclasses)),
    'PreResNet164_DConv': lambda args: L.Classifier(
        PreResNet_DConv(164, args.nclasses)),
    'PreResNet164_PGP': lambda args: FuseTrainWrapper(
        PreResNet_PGP(164, args.nclasses)),
    'DenseNetBC100': lambda args: L.Classifier(
        DenseNetBC(args.nclasses, (16, 16, 16), 12)),
    'DenseNetBC100_DConv': lambda args: L.Classifier(
        DenseNetBC_DConv(args.nclasses, (16, 16, 16), 12)),
    'DenseNetBC100_PGP': lambda args: FuseTrainWrapper(
        DenseNetBC_PGP(args.nclasses, (16, 16, 16), 12)),
    'WideResNet28-10': lambda args: L.Classifier(
        WideResNet(28, args.nclasses, 10)),
    'WideResNet28-10_DConv': lambda args: L.Classifier(
        WideResNet_DConv(28, args.nclasses, 10)),
    'WideResNet28-10_PGP': lambda args: FuseTrainWrapper(
        WideResNet_PGP(28, args.nclasses, 10)),
    'ResNeXt29_8x64d': lambda args: L.Classifier(
        ResNeXt(29, args.nclasses)),
    'ResNeXt29_8x64d_DConv': lambda args: L.Classifier(
        ResNeXt_DConv(29, args.nclasses)),
    'ResNeXt29_8x64d_PGP': lambda args: FuseTrainWrapper(
        ResNeXt_PGP(29, args.nclasses)),
    'PyramidNetB164': lambda args: L.Classifier(
        PyramidNet(164, args.nclasses)),
    'PyramidNetB164_DConv': lambda args: L.Classifier(
        PyramidNet_DConv(164, args.nclasses)),
    'PyramidNetB164_PGP': lambda args: FuseTrainWrapper(
        PyramidNet_PGP(164, args.nclasses)),
    'Shake-Shake26_2x64d': lambda args: L.Classifier(
        ShakeShake(26, args.nclasses, k=64)),
}


class FuseTrainWrapper(chainer.Chain):

    def __init__(self, predictor, n_kernel=16, sm_fuse=False):
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
