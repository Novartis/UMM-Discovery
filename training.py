# Copyright 2020 Novartis Institutes for BioMedical Research Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file incorporates work covered by the following copyright and  
# permission notice:
#
#   Copyright (c) 2017-present, Facebook, Inc.
#   All rights reserved.
#
#   This source code is licensed as Creative Commons Attribution-Noncommercial 
#   and can be found under https://creativecommons.org/licenses/by-nc/4.0/.

import os
import time
import numpy as np

import torch
import torch.optim

from util import AverageMeter

def train(loader, model, crit, opt, epoch, args):
    """Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
    if args.verbose:
        print('Start training')

    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()

    # switch to train mode
    model.train()

    # create an optimizer for the last fc layer
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=args.lr,
        weight_decay=10 ** args.wd,
    )

    end = time.time()
    for i, (input_tensor, target) in enumerate(loader):
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input_tensor.cuda())
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = crit(output, target_var)

        # record loss
        losses.update(loss.data, input_tensor.size(0))

        # compute gradient and do SGD step
        opt.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        optimizer_tl.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(loader), batch_time=batch_time,
                          data_time=data_time, loss=losses))

    # save epoch checkpoint
    path = os.path.join(args.output_path, 'checkpoints', 'checkpoint_epoch' + str(epoch + 1) + '.pth.tar')
    if args.verbose:
        print('Save checkpoint at: {0}'.format(path))

    torch.save({
        'epoch': epoch + 1,
        'n_input_dim': args.n_input_dim,
        'n_features': args.n_features,
        'state_dict': model.state_dict(),
        'optimizer': opt.state_dict()
    }, path)
    return losses.avg


def compute_features(dataloader, model, N, args):
    if args.verbose:
        print('Compute features')

    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor, _) in enumerate(dataloader):
        input_var = torch.autograd.Variable(input_tensor.cuda())
        with torch.no_grad():
            aux = model(input_var).data.cpu().numpy().astype('float32')  

        if i == 0:
            features = np.zeros((N, aux.shape[1])).astype('float32')

        if i < len(dataloader) - 1:
            features[i * args.batch: (i + 1) * args.batch] = aux
        else:
            # special treatment for final batch
            features[i * args.batch:] = aux

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))
    return features
