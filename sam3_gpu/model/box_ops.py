# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""

from typing import Tuple

import torch


def unbind(x: torch.Tensor, dim):
    return x.unbind(dim)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = unbind(x, -1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_cxcywh_to_xywh(x):
    x_c, y_c, w, h = unbind(x, -1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (w), (h)]
    return torch.stack(b, dim=-1)


def box_xywh_to_xyxy(x):
    x, y, w, h = unbind(x, -1)
    b = [(x), (y), (x + w), (y + h)]
    return torch.stack(b, dim=-1)


def box_xywh_to_cxcywh(x):
    x, y, w, h = unbind(x, -1)
    b = [(x + 0.5 * w), (y + 0.5 * h), (w), (h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_xywh(x):
    x, y, X, Y = unbind(x, -1)
    b = [(x), (y), (X - x), (Y - y)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = unbind(x, -1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_area(boxes):
    """
    Batched version of box area. Boxes should be in [x0, y0, x1, y1] format.

    Inputs:
    - boxes: Tensor of shape (..., 4)

    Returns:
    - areas: Tensor of shape (...,)
    """
    x0, y0, x1, y1 = unbind(boxes, -1)
    return (x1 - x0) * (y1 - y0)


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32, device=masks.device)

    h, w = masks.shape[-2:]

    masks_bool = masks.to(torch.bool)
    masks_f = masks_bool.to(torch.float32)

    y = torch.arange(0, h, dtype=torch.float32, device=masks.device)[:, None]
    x = torch.arange(0, w, dtype=torch.float32, device=masks.device)[None, :]
    y = y.expand(h, w)
    x = x.expand(h, w)

    x_mask = masks_f * x[None, :, :]
    y_mask = masks_f * y[None, :, :]

    flat_x = x_mask.reshape(masks.shape[0], -1)
    flat_y = y_mask.reshape(masks.shape[0], -1)
    flat_m = masks_bool.reshape(masks.shape[0], -1)

    x_max = flat_x.max(dim=1).values + 1
    x_min = torch.where(flat_m, flat_x, torch.tensor(1e8, device=masks.device)).min(dim=1).values
    y_max = flat_y.max(dim=1).values + 1
    y_min = torch.where(flat_m, flat_y, torch.tensor(1e8, device=masks.device)).min(dim=1).values

    boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)
    has_any = flat_m.any(dim=1).to(boxes.dtype)
    boxes = boxes * has_any[:, None]
    return boxes


def box_iou(boxes1, boxes2):
    """
    Batched version of box_iou. Boxes should be in [x0, y0, x1, y1] format.

    Inputs:
    - boxes1: Tensor of shape (..., N, 4)
    - boxes2: Tensor of shape (..., M, 4)

    Returns:
    - iou, union: Tensors of shape (..., N, M)
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.maximum(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
    rb = torch.minimum(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])

    wh = torch.clamp((rb - lt), min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[..., None] + area2[..., None, :] - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Batched version of Generalized IoU from https://giou.stanford.edu/

    Boxes should be in [x0, y0, x1, y1] format

    Inputs:
    - boxes1: Tensor of shape (..., N, 4)
    - boxes2: Tensor of shape (..., M, 4)

    Returns:
    - giou: Tensor of shape (..., N, M)
    """
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.minimum(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
    rb = torch.maximum(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])

    wh = torch.clamp((rb - lt), min=0)
    area = wh[..., 0] * wh[..., 1]

    return iou - (area - union) / area


def fast_diag_generalized_box_iou(boxes1, boxes2):
    assert len(boxes1) == len(boxes2)
    box1_xy = boxes1[:, 2:]
    box1_XY = boxes1[:, :2]
    box2_xy = boxes2[:, 2:]
    box2_XY = boxes2[:, :2]

    area1 = torch.prod((box1_xy - box1_XY), dim=-1)
    area2 = torch.prod((box2_xy - box2_XY), dim=-1)

    lt = torch.maximum(box1_XY, box2_XY)
    lt2 = torch.minimum(box1_XY, box2_XY)
    rb = torch.minimum(box1_xy, box2_xy)
    rb2 = torch.maximum(box1_xy, box2_xy)

    inter = torch.prod(torch.clamp((rb - lt), min=0), dim=-1)
    tot_area = torch.prod(torch.clamp((rb2 - lt2), min=0), dim=-1)

    union = area1 + area2 - inter

    iou = inter / union

    return iou - (tot_area - union) / tot_area


def fast_diag_box_iou(boxes1, boxes2):
    assert len(boxes1) == len(boxes2)
    box1_xy = boxes1[:, 2:]
    box1_XY = boxes1[:, :2]
    box2_xy = boxes2[:, 2:]
    box2_XY = boxes2[:, :2]

    area1 = torch.prod((box1_xy - box1_XY), dim=-1)
    area2 = torch.prod((box2_xy - box2_XY), dim=-1)

    lt = torch.maximum(box1_XY, box2_XY)
    rb = torch.minimum(box1_xy, box2_xy)

    inter = torch.prod(torch.clamp((rb - lt), min=0), dim=-1)

    union = area1 + area2 - inter

    iou = inter / union

    return iou


def box_xywh_inter_union(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Assumes boxes in xywh format
    assert boxes1.shape[-1] == 4 and boxes2.shape[-1] == 4
    boxes1 = box_xywh_to_xyxy(boxes1)
    boxes2 = box_xywh_to_xyxy(boxes2)
    box1_tl_xy = boxes1[..., :2]
    box1_br_xy = boxes1[..., 2:]
    box2_tl_xy = boxes2[..., :2]
    box2_br_xy = boxes2[..., 2:]
    area1 = torch.prod((box1_br_xy - box1_tl_xy), dim=-1)
    area2 = torch.prod((box2_br_xy - box2_tl_xy), dim=-1)

    assert (area1 >= 0).all() and (area2 >= 0).all()
    tl = torch.maximum(box1_tl_xy, box2_tl_xy)
    br = torch.minimum(box1_br_xy, box2_br_xy)

    inter = torch.prod(torch.clamp((br - tl), min=0), dim=-1)
    union = area1 + area2 - inter

    return inter, union
