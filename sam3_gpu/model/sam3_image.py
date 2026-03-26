from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3_gpu.model.box_ops import box_cxcywh_to_xywh, box_cxcywh_to_xyxy
from sam3_gpu.model.vl_combiner import SAM3VLBackbone
from sam3_gpu.model.geometry_encoders import Prompt
from sam3_gpu.model.model_misc import MLP, DotProductScoring, inverse_sigmoid

def _update_out(out, out_name, out_value, auxiliary=True, update_aux=True):
    out[out_name] = out_value[-1] if auxiliary else out_value
    if auxiliary and update_aux:
        if "aux_outputs" not in out:
            out["aux_outputs"] = [{} for _ in range(len(out_value) - 1)]
        assert len(out["aux_outputs"]) == len(out_value) - 1
        for aux_output, aux_value in zip(out["aux_outputs"], out_value[:-1]):
            aux_output[out_name] = aux_value

class Sam3Image(nn.Module):
    TEXT_ID_FOR_TEXT = 0
    TEXT_ID_FOR_VISUAL = 1
    TEXT_ID_FOR_GEOMETRIC = 2

    def __init__(self, backbone, transformer, input_geometry_encoder, segmentation_head=None, num_feature_levels=1, o2m_mask_predict=True, dot_prod_scoring=None, use_instance_query=True, multimask_otuput=True, use_act_checkpoint_seg_head=True, interactivity_in_encoder=True, matcher=None, use_dot_prod_scoring=True, supervise_joint_box_scores=False, detach_presence_in_joint_score=False, separate_scorer_for_instance=False, num_interactive_steps_val=0, inst_interactive_predictor=None, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.geometry_encoder = input_geometry_encoder
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.segmentation_head = segmentation_head
        self.o2m_mask_predict = o2m_mask_predict
        self.use_act_checkpiont_seg_head = use_act_checkpoint_seg_head
        self.matcher = matcher
        self.num_interactive_steps_val = num_interactive_steps_val
        self.use_dot_prod_scoring = use_dot_prod_scoring
        if self.use_dot_prod_scoring:
            assert dot_prod_scoring is not None
            self.dot_prod_scoring = dot_prod_scoring
            self.instance_dot_prod_scoring = None
            if separate_scorer_for_instance:
                # ... creates instance scoring
                pass
        else:
            self.class_embed = nn.Linear(self.hidden_dim, 1)
            self.instance_class_embed = None
            if separate_scorer_for_instance:
                self.instance_class_embed = nn.Linear(self.hidden_dim, 1)
        self.supervise_joint_box_scores = supervise_joint_box_scores
        self.detach_presence_in_joint_score = detach_presence_in_joint_score
        num_o2o_static = self.transformer.decoder.num_queries
        num_o2m_static = self.transformer.decoder.num_o2m_queries
        assert num_o2m_static == (num_o2o_static if self.transformer.decoder.dac else 0)
        self.dac = self.transformer.decoder.dac
        self.use_instant_query = use_instance_query
        self.multimask_output = multimask_otuput
        self.inst_interactive_predictor = inst_interactive_predictor

    def _get_img_feats(self, backbone_out, img_ids):
        if "backbone_fpn" in backbone_out:
            if "id_mapping" in backbone_out and backbone_out["id_mapping"] is not None:
                img_ids = backbone_out["id_mapping"][img_ids]
                assert (img_ids >= 0).all()
            vis_feats = backbone_out["backbone_fpn"][-self.num_feature_levels:]
            vis_pos_enc = backbone_out["vision_pos_enc"][-self.num_feature_levels:]
            vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]
            img_feats = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_feats]
            img_pos_embeds = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_pos_enc]
            return backbone_out, img_feats, img_pos_embeds, vis_feat_sizes

    def _encode_prompt(self, backbone_out, find_input, geometric_prompt, visual_prompt_embed=None, visual_prompt_mask=None, encode_text=True, prev_mask_pred=None):
        txt_ids = find_input.text_ids
        txt_feats = backbone_out["language_features"][:, txt_ids]
        txt_masks = backbone_out["language_mask"][txt_ids]
        feat_tuple = self._get_img_feats(backbone_out, find_input.img_ids)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple
        if prev_mask_pred is not None:
            img_feats = [img_feats[-1] + prev_mask_pred]
        geo_feats, geo_masks = self.geometry_encoder(geo_prompt=geometric_prompt, img_feats=img_feats, img_sizes=vis_feat_sizes, img_pos_embeds=img_pos_embeds)
        if visual_prompt_embed is None:
            visual_prompt_embed = torch.zeros((0, *geo_feats.shape[1:]), device=geo_feats.device)
            visual_prompt_mask = torch.zeros((*geo_masks.shape[:-1], 0), dtype=geo_masks.dtype, device=geo_masks.device)
        if encode_text:
            prompt = torch.cat([txt_feats, geo_feats, visual_prompt_embed], dim=0)
            prompt_mask = torch.cat([txt_masks, geo_masks, visual_prompt_mask], dim=1)
        else:
            prompt = torch.cat([geo_feats, visual_prompt_embed], dim=0)
            prompt_mask = torch.cat([geo_masks, visual_prompt_mask], dim=1)
        return prompt, prompt_mask, backbone_out

    def _run_encoder(self, backbone_out, find_input, prompt, prompt_mask, encoder_extra_kwargs=None):
        feat_tuple = self._get_img_feats(backbone_out, find_input.img_ids)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple
        prompt_pos_embed = torch.zeros_like(prompt)
        memory = self.transformer.encoder(src=img_feats, src_key_padding_mask=None, src_pos=img_pos_embeds, prompt=prompt, prompt_pos=prompt_pos_embed, prompt_key_padding_mask=prompt_mask, feat_sizes=vis_feat_sizes, encoder_extra_kwargs=encoder_extra_kwargs)
        encoder_out = {
            "encoder_hidden_states": memory["memory"],
            "pos_embed": memory["pos_embed"],
            "padding_mask": memory["padding_mask"],
            "level_start_index": memory["level_start_index"],
            "spatial_shapes": memory["spatial_shapes"],
            "valid_ratios": memory["valid_ratios"],
            "vis_feat_sizes": vis_feat_sizes,
            "prompt_before_enc": prompt,
            "prompt_after_enc": memory.get("memory_text", prompt),
            "prompt_mask": prompt_mask
        }
        return backbone_out, encoder_out, feat_tuple

    def _run_decoder(self, pos_embed, memory, src_mask, out, prompt, prompt_mask, encoder_out):
        bs = memory.shape[1]
        query_embed = self.transformer.decoder.query_embed.weight
        tgt = query_embed[:, None].expand(-1, bs, -1).contiguous()
        apply_dac = self.transformer.decoder.dac and self.training
        hs, reference_boxes, dec_presence_out, dec_presence_feats = self.transformer.decoder(
            tgt=tgt, memory=memory, memory_key_padding_mask=src_mask, pos=pos_embed,
            reference_boxes=None,
            level_start_index=encoder_out["level_start_index"],
            spatial_shapes=encoder_out["spatial_shapes"],
            valid_ratios=encoder_out["valid_ratios"],
            tgt_mask=None, memory_text=prompt, text_attention_mask=prompt_mask, apply_dac=apply_dac,
        )
        hs = hs.permute(0, 2, 1, 3)
        reference_boxes = reference_boxes.permute(0, 2, 1, 3)
        if dec_presence_out is not None:
            dec_presence_out = dec_presence_out.permute(0, 2, 1)
        out["presence_feats"] = dec_presence_feats
        self._update_scores_and_boxes(out, hs, reference_boxes, prompt, prompt_mask, dec_presence_out=dec_presence_out)
        return out, hs

    def _update_scores_and_boxes(self, out, hs, reference_boxes, prompt, prompt_mask, dec_presence_out=None, is_instance_prompt=False):
        apply_dac = self.transformer.decoder.dac and self.training
        num_o2o = (hs.shape[2] // 2) if apply_dac else hs.shape[2]
        num_o2m = hs.shape[2] - num_o2o
        out["queries"] = hs[-1][:, :num_o2o]
        if self.use_dot_prod_scoring:
            dot_prod_scoring_head = self.dot_prod_scoring
            if is_instance_prompt and self.instance_dot_prod_scoring is not None:
                dot_prod_scoring_head = self.instance_dot_prod_scoring
            outputs_class = dot_prod_scoring_head(hs, prompt, prompt_mask)
        else:
            class_embed_head = self.class_embed
            if is_instance_prompt and self.instance_class_embed is not None:
                class_embed_head = self.instance_class_embed
            outputs_class = class_embed_head(hs)
        box_head = self.transformer.decoder.bbox_embed
        if is_instance_prompt and self.transformer.decoder.instance_bbox_embed is not None:
            box_head = self.transformer.decoder.instance_bbox_embed
        anchor_box_offsets = box_head(hs)
        reference_boxes_inv_sig = inverse_sigmoid(reference_boxes)
        outputs_coord = torch.sigmoid(reference_boxes_inv_sig + anchor_box_offsets)
        outputs_boxes_xyxy = box_cxcywh_to_xyxy(outputs_coord)
        if dec_presence_out is not None:
            _update_out(out, "presence_logit_dec", dec_presence_out, update_aux=self.training)
        if self.supervise_joint_box_scores:
            assert dec_presence_out is not None
            prob_dec_presence_out = torch.sigmoid(dec_presence_out)
            if self.detach_presence_in_joint_score:
                prob_dec_presence_out = prob_dec_presence_out.detach()
            outputs_class = torch.clamp(inverse_sigmoid(outputs_class.sigmoid() * prob_dec_presence_out[:,:,None]), -10.0, 10.0)
        _update_out(out, "pred_logits", outputs_class[:, :, :num_o2o], update_aux=self.training)
        _update_out(out, "pred_boxes", outputs_coord[:, :, :num_o2o], update_aux=self.training)
        _update_out(out, "pred_boxes_xyxy", outputs_boxes_xyxy[:, :, :num_o2o], update_aux=self.training)
        if num_o2m > 0 and self.training:
            _update_out(out, "pred_logits_o2m", outputs_class[:, :, num_o2o:], update_aux=self.training)
            _update_out(out, "pred_boxes_o2m", outputs_coord[:, :, :num_o2o], update_aux=self.training)
            _update_out(out, "pred_boxes_xyxy_o2m", outputs_boxes_xyxy[:, :, num_o2o:], update_aux=self.training)

    def _run_segmentation_heads(self, out, backbone_out, img_ids, vis_feat_sizes, encoder_hidden_states, prompt, prompt_mask, hs):
        apply_dac = self.transformer.decoder.dac and self.training
        if self.segmentation_head is not None:
            num_o2o = (hs.shape[2] // 2) if apply_dac else hs.shape[2]
            num_o2m = hs.shape[2] - num_o2o
            obj_queries = hs if self.o2m_mask_predict else hs[:, :, :num_o2o]
            seg_head_outputs = self.segmentation_head(backbone_feats=backbone_out["backbone_fpn"], obj_queries=obj_queries, image_ids=img_ids, encoder_hidden_states=encoder_hidden_states, prompt=prompt, prompt_mask=prompt_mask)
            aux_masks = False
            for k, v in seg_head_outputs.items():
                if k in self.segmentation_head.instance_keys:
                    _update_out(out, k, v[:, :num_o2o], auxiliary=aux_masks)
                    if self.o2m_mask_predict and num_o2m > 0:
                        _update_out(out, f"{k}_o2m", v[:, num_o2o], auxiliary=aux_masks)
                else:
                    out[k] = v
        else:
            backbone_out.pop("backbone_fpn", None)

    def call_grounding(self, backbone_out, find_input, find_target, geometric_prompt):
        prompt, prompt_mask, backbone_out = self._encode_prompt(backbone_out, find_input, geometric_prompt)
        backbone_out, encoder_out, _ = self._run_encoder(backbone_out, find_input, prompt, prompt_mask)
        out = {"encoder_hidden_states": encoder_out["encoder_hidden_states"], "prev_encoder_out": {"encoder_out": encoder_out, "backbone_out": backbone_out}}
        out, hs = self._run_decoder(memory=out["encoder_hidden_states"], pos_embed=encoder_out["pos_embed"], src_mask=encoder_out["padding_mask"], out=out, prompt=prompt, prompt_mask=prompt_mask, encoder_out=encoder_out)
        self._run_segmentation_heads(out=out, backbone_out=backbone_out, img_ids=find_input.img_ids, vis_feat_sizes=encoder_out["vis_feat_sizes"], encoder_hidden_states=out["encoder_hidden_states"], prompt=prompt, prompt_mask=prompt_mask, hs=hs)
        return out

    def _get_dummy_prompt(self, num_prompts=1):
        device = next(self.parameters()).device
        geometric_prompt = Prompt(
            box_embeddings=torch.zeros((0, num_prompts, 4), device=device),
            box_mask=torch.zeros((num_prompts, 0), dtype=torch.bool, device=device),
        )
        return geometric_prompt

    def forward(self):
        pass
