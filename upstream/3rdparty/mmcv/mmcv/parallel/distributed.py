# ----------------------------------------------------------------------------
# Portions of this file have been modified by Ash in 2025.
# Original source: OpenMMLab/mmcv (Apache License 2.0).
# Modifications were made to support the working environment of TV3S.
# ----------------------------------------------------------------------------

# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.nn.parallel.distributed import (DistributedDataParallel,
                                           _find_tensors)

from mmcv import print_log
from mmcv.utils import TORCH_VERSION, digit_version
from .scatter_gather import scatter_kwargs


def _conv_pre_hook(module, inputs):
    """Make Conv2d input contiguous (NCHW) before forward."""
    return (inputs[0].contiguous(),) if not inputs[0].is_contiguous() else inputs


def _conv_output_hook(module, inputs, output):
    """Register a backward hook on the Conv2d output tensor.

    This fires when dy arrives at the Conv2d backward node — BEFORE cuDNN
    computes dw = conv_backward_weight(x, dy). Making dy contiguous here
    ensures cuDNN uses an NCHW algorithm for dw, so the weight gradient is
    always contiguous when DDP copies it into the NCCL bucket.
    """
    if output.requires_grad:
        output.register_hook(
            lambda g: g.contiguous() if not g.is_contiguous() else g
        )
    return output


class MMDistributedDataParallel(DistributedDataParallel):
    """The DDP module that supports DataContainer.

    MMDDP has two main differences with PyTorch DDP:

    - It supports a custom type :class:`DataContainer` which allows more
      flexible control of input data.
    - It implement two APIs ``train_step()`` and ``val_step()``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Blackwell (sm_120) crash fix: cuDNN picks an NHWC kernel if EITHER
        # the Conv2d input (x) OR the output gradient (dy) is channels-last,
        # producing a non-contiguous weight gradient. DDP's C++ AccumulateGrad
        # hook copies that gradient into the NCCL bucket before any Python
        # param.register_hook can fix it — causing an illegal memory access.
        #
        # Two-part fix:
        #   pre-hook:     make x contiguous before the forward pass
        #   output hook:  make dy contiguous before conv_backward_weight(x, dy)
        #                 (output tensor hooks fire before AccumulateGrad)
        for m in self.module.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.register_forward_pre_hook(_conv_pre_hook)
                m.register_forward_hook(_conv_output_hook)

    def to_kwargs(self, inputs, kwargs, device_id):
        # Use `self.to_kwargs` instead of `self.scatter` in pytorch1.8
        # to move all tensors to device_id
        return scatter_kwargs(inputs, kwargs, [device_id], dim=self.dim)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def train_step(self, *inputs, **kwargs):
        """train_step() API for module wrapped by DistributedDataParallel.

        This method is basically the same as
        ``DistributedDataParallel.forward()``, while replacing
        ``self.module.forward()`` with ``self.module.train_step()``.
        It is compatible with PyTorch 1.1 - 1.5.
        """

        # In PyTorch >= 1.7, ``reducer._rebuild_buckets()`` is moved from the
        # end of backward to the beginning of forward.
        if ('parrots' not in TORCH_VERSION
                and digit_version(TORCH_VERSION) >= digit_version('1.7')
                and self.reducer._rebuild_buckets()):
            print_log(
                'Reducer buckets have been rebuilt in this iteration.',
                logger='mmcv')

        if ('parrots' not in TORCH_VERSION
                and digit_version(TORCH_VERSION) >= digit_version('1.11.0')):
            if self._check_sync_bufs_pre_fwd():
                self._sync_buffers()
        else:
            if (getattr(self, 'require_forward_param_sync', False)
                    and self.require_forward_param_sync):
                self._sync_params()

        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                output = self.module.train_step(*inputs[0], **kwargs[0])
            else:
                outputs = self.parallel_apply(
                    self._module_copies[:len(inputs)], inputs, kwargs)
                output = self.gather(outputs, self.output_device)
        else:
            output = self.module.train_step(*inputs, **kwargs)

        if ('parrots' not in TORCH_VERSION
                and digit_version(TORCH_VERSION) >= digit_version('1.11.0')):
            if self._check_sync_bufs_post_fwd():
                self._sync_buffers()

        if (torch.is_grad_enabled()
                and getattr(self, 'require_backward_grad_sync', False)
                and self.require_backward_grad_sync):
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            if ('parrots' not in TORCH_VERSION
                    and digit_version(TORCH_VERSION) > digit_version('1.2')):
                self.require_forward_param_sync = False
        return output

    def val_step(self, *inputs, **kwargs):
        """val_step() API for module wrapped by DistributedDataParallel.

        This method is basically the same as
        ``DistributedDataParallel.forward()``, while replacing
        ``self.module.forward()`` with ``self.module.val_step()``.
        It is compatible with PyTorch 1.1 - 1.5.
        """
        # In PyTorch >= 1.7, ``reducer._rebuild_buckets()`` is moved from the
        # end of backward to the beginning of forward.
        if ('parrots' not in TORCH_VERSION
                and digit_version(TORCH_VERSION) >= digit_version('1.7')
                and self.reducer._rebuild_buckets()):
            print_log(
                'Reducer buckets have been rebuilt in this iteration.',
                logger='mmcv')

        if ('parrots' not in TORCH_VERSION
                and digit_version(TORCH_VERSION) >= digit_version('1.11.0')):
            if self._check_sync_bufs_pre_fwd():
                self._sync_buffers()
        else:
            if (getattr(self, 'require_forward_param_sync', False)
                    and self.require_forward_param_sync):
                self._sync_params()

        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                output = self.module.val_step(*inputs[0], **kwargs[0])
            else:
                outputs = self.parallel_apply(
                    self._module_copies[:len(inputs)], inputs, kwargs)
                output = self.gather(outputs, self.output_device)
        else:
            output = self.module.val_step(*inputs, **kwargs)

        if ('parrots' not in TORCH_VERSION
                and digit_version(TORCH_VERSION) >= digit_version('1.11.0')):
            if self._check_sync_bufs_post_fwd():
                self._sync_buffers()

        if (torch.is_grad_enabled()
                and getattr(self, 'require_backward_grad_sync', False)
                and self.require_backward_grad_sync):
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            if ('parrots' not in TORCH_VERSION
                    and digit_version(TORCH_VERSION) > digit_version('1.2')):
                self.require_forward_param_sync = False
        return output
