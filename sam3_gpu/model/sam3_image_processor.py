import time
from functools import partial
from typing import Dict, List
import PIL
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from sam3_gpu.model import box_ops
from sam3_gpu.model.data_misc import FindStage
from sam3_gpu.model.geometry_encoders import Prompt


def transform(image_path_or_pil, resolution, device=None):
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil).convert("RGB")
    else:
        img = image_path_or_pil.convert("RGB")
    img = img.resize((resolution, resolution), resample=Image.Resampling.LANCZOS)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = (img_np - 0.5) / 0.5
    tensor = torch.tensor(img_np).permute(2, 0, 1)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


class Sam3Processor:
    def __init__(self, model, resolution=1008, confidence_threshold=0.5):
        self.model = model
        self.resolution = resolution
        self.confidence_threshold = confidence_threshold
        self.device = next(model.parameters()).device
        self.transform = partial(transform, resolution=self.resolution, device=self.device)
        self.find_stage = FindStage(
            img_ids=torch.tensor([0], dtype=torch.int64, device=self.device),
            text_ids=torch.tensor([0], dtype=torch.int64, device=self.device),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None,
        )

    @torch.no_grad()
    def set_image(self, image, state=None):
        if state is None:
            state = {}
        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        else:
            raise ValueError("Image must be a PIL image")
        image = self.transform(image)[None]
        state["original_height"] = height
        state["original_width"] = width
        start = time.perf_counter()
        state["backbone_out"] = self.model.backbone.call_image(image)
        print(f"Backbone pass took {time.perf_counter() - start:.2f} Seconds")
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(sam2_backbone_out["backbone_fpn"][0])
            sam2_backbone_out["backbone_fpn"][1] = self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(sam2_backbone_out["backbone_fpn"][1])
        return state

    @torch.no_grad()
    def set_text_prompt(self, prompt, state):
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")
        text_outputs = self.model.backbone.call_text([prompt])
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()
        return self._call_grounding(state)

    @torch.no_grad()
    def add_geometric_prompt(self, box, label, state):
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")
        if "language_features" not in state["backbone_out"]:
            dummy_text_outputs = self.model.backbone.call_text(["visual"])
            state["backbone_out"].update(dummy_text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()
        boxes = torch.tensor(box, dtype=torch.float32, device=self.device).reshape(1, 1, 4)
        labels = torch.tensor([label], dtype=torch.bool, device=self.device).reshape(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)
        return self._call_grounding(state)

    def reset_all_prompts(self, state):
        if "backbone_out" in state:
            for key in ["language_features", "language_mask", "language_embeds"]:
                if key in state["backbone_out"]:
                    del state["backbone_out"][key]
        for key in ["geometric_prompt", "boxes", "masks", "masks_logits", "scores"]:
            if key in state:
                del state[key]

    @torch.no_grad()
    def _call_grounding(self, state):
        outputs = self.model.call_grounding(backbone_out=state["backbone_out"], find_input=self.find_stage, geometric_prompt=state["geometric_prompt"], find_target=None)
        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = torch.sigmoid(out_logits)
        presence_score = torch.sigmoid(outputs["presence_logit_dec"])[:, None]
        out_probs = (out_probs * presence_score).squeeze(-1)
        keep = out_probs > self.confidence_threshold
        indices = keep[0].nonzero(as_tuple=False).squeeze(-1)
        out_probs = out_probs[0][indices]
        out_masks = out_masks[0][indices]
        out_bbox = out_bbox[0][indices]
        seg_mask = outputs['semantic_seg']
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)
        boxes = boxes * scale_fct[None, :]
        out_masks = F.interpolate(out_masks[:, None].float(), size=(img_h, img_w), mode="bilinear", align_corners=False)
        out_masks = torch.sigmoid(out_masks)
        seg_mask = F.interpolate(seg_mask.float(), size=(img_h, img_w), mode="bilinear", align_corners=False)
        state["semantic_seg"] = seg_mask
        state["mask_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state
