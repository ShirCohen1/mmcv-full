# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

import mmcv
from .base_runner import BaseRunner
from .builder import RUNNERS
from .checkpoint import save_checkpoint
from .utils import get_host_info

class early_stopping():

    def __init__(self, patience):
        self.patience = patience
        self.counter = 0

    def check_stop_condition(self, val_metrics):
        if len(val_metrics) > 1:
            if val_metrics[-1] > val_metrics[-2]:
                self.counter += 1
            elif val_metrics[-1] <= val_metrics[-2]:
                self.counter = 0
                # resets each time if there is a decrease in loss

        if self.counter == self.patience:
            return True  # if true should exit training loop

        else:
            return False

@RUNNERS.register_module()
class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def run_iter(self, data_batch: Any, train_mode: bool, **kwargs) -> None:
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            # returns losses from model base in dict format
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
            # print(f'In validation mode: returning outputs: {outputs["loss"]}!!!!!!!!!!!!!!!')

        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
#             if i == 0:
#                 outputs_to_avg = self.outputs['loss']
#             if i > 0:
#                 outputs_to_avg = torch.cat((outputs_to_avg.reshape(-1) , self.outputs['loss'].reshape(1)),dim=0)
            self.call_hook('after_train_iter')
            del self.data_batch
            self._iter += 1
        #dana added
        self.save_checkpoint(self.work_dir, f'epoch_{self._epoch}.pth')

        self.call_hook('after_train_epoch')
        self._epoch += 1
        
#         self.meta['train_metrics'].append(outputs_to_avg.mean())


    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition

        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            if i == 0:
                outputs_to_avg = self.outputs['loss']
            if i > 0:
                outputs_to_avg = torch.cat((outputs_to_avg.reshape(-1) , self.outputs['loss'].reshape(1)),dim=0)
            self.call_hook('after_val_iter')
            del self.data_batch
        self.call_hook('after_val_epoch')
        self.meta['val_metrics'].append(outputs_to_avg.mean())

    def run(self,
            data_loaders: List[DataLoader],
            workflow: List[Tuple[str, int]],
            max_epochs: Optional[int] = None,
            **kwargs) -> None:
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)

#         self.meta['train_metrics'] = list()
        self.meta['val_metrics'] = list()
        patience = self.meta['patience']
        early_stopper = early_stopping(patience)

        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

#         results = pd.DataFrame(columns = ['epoch', 'train_loss', 'val_loss'])

        while self.epoch < self._max_epochs :
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break

                    epoch_runner(data_loaders[i], **kwargs) # Runs one epoch and multiple iterations in batches
            
#             ##save history
#             epoch = self.epoch
#             train_loss = self.meta['train_metrics'][self.epoch-1]
            
#             print(loss_train)
#             print(type(loss_train))
#             print(loss_train.cpu())
#             print(type(loss_train))

#             val_loss = self.meta['val_metrics'][self.epoch-1]

#             tmp_result = pd.DataFrame([epoch, train_loss, val_loss], columns = ['epoch', 'train_loss', 'val_loss'])
#             results = pd.concat([tmp_result, results])
#             print(results)

            if early_stopper.check_stop_condition(self.meta['val_metrics']):
                break

#         print(f"train losses: {self.meta['train_metrics']}")
        print(f"val losses: {self.meta['val_metrics']}")
#         results = np.array([self.meta['train_metrics'].numpy(),self.meta['val_metrics'].numpy()])
#         results = self.meta['train_metrics'].numpy()

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')
        return 

    def save_checkpoint(self,
                        out_dir: str,
                        filename_tmpl: str = 'epoch_{}.pth',
                        save_optimizer: bool = True,
                        meta: Optional[Dict] = None,
                        create_symlink: bool = True) -> None:
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)


@RUNNERS.register_module()
class Runner(EpochBasedRunner):
    """Deprecated name of EpochBasedRunner."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            'Runner was deprecated, please use EpochBasedRunner instead',
            DeprecationWarning)
        super().__init__(*args, **kwargs)
