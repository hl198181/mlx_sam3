from dataclasses import dataclass

import torch
import torch.nn.functional as F

from typing import Any, List, Optional, Union

MyTensor = Union[torch.Tensor, List[Any]]


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    if input.numel() == 0:
        out_shape = list(input.shape)
        if size is not None:
            out_shape[2] = size[0]
            out_shape[3] = size[1]
        elif scale_factor is not None:
            out_shape[2] = int(out_shape[2] * scale_factor)
            out_shape[3] = int(out_shape[3] * scale_factor)
        return torch.zeros(out_shape, dtype=input.dtype, device=input.device)

    return F.interpolate(input, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


@dataclass
class FindStage:
    img_ids: MyTensor
    img_ids__type = torch.int64
    text_ids: MyTensor
    text_ids__type = torch.int64

    input_boxes: MyTensor
    input_boxes__type = torch.float32
    input_boxes_mask: MyTensor
    input_boxes_mask__type = torch.bool
    input_boxes_label: MyTensor
    input_boxes_label__type = torch.int64

    input_points: MyTensor
    input_points__type = torch.float32
    input_points_mask: MyTensor
    input_points_mask__type = torch.bool

    # We track the object ids referred to by this query.
    # This is beneficial for tracking in videos without the need for pointers.
    object_ids: Optional[List[List]] = None  # List of objects per query
