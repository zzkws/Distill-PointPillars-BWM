import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict
from ...config import cfg_from_yaml_file
from .pointpillar import PointPillar
from .centerpoint import CenterPoint
from .detector3d_template import Detector3DTemplate
from ...utils import common_utils  
# exp 3
class ConsistencyDistillPP(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg, num_class, dataset)

        # 1. Build student model
        self.student = PointPillar(model_cfg, num_class, dataset)
        
        # 2. Build teacher model
        teacher_cfg_path = model_cfg.TEACHER_CFG_FILE 
        teacher_cfg = EasyDict()
        cfg_from_yaml_file(teacher_cfg_path, teacher_cfg)
        
        self.teacher = CenterPoint(teacher_cfg.MODEL, num_class, dataset)

        logger = common_utils.create_logger(log_file=None, rank=0)
        logger.info(f"==> Loading Teacher Weights from: {model_cfg.TEACHER_CKPT}")

        self.teacher.load_params_from_file(
            filename=model_cfg.TEACHER_CKPT, 
            logger=logger,  
            to_cpu=False
        )
        
        # Freeze Teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def forward(self, batch_dict):
        student_outs = self.student(batch_dict)

        s_feat = batch_dict.get('spatial_features_2d', None)
        
        if not self.training:
            return student_outs

        ret_dict, tb_dict, disp_dict = student_outs

        # Obtain teacher features
        with torch.no_grad():
            self.teacher.eval()
            self.teacher(batch_dict)
            t_feat = batch_dict['spatial_features_2d']

        # Feature-level LaserMix 
        if s_feat is not None and s_feat.shape[0] >= 2:
            
            
            def laser_mix_op(features):
                B, C, H, W = features.shape
                
                mask_even = torch.arange(H, device=features.device) % 2 == 0
                mask_even = mask_even.view(1, 1, H, 1).float() 
                mask_odd = 1.0 - mask_even
                
                features_rolled = torch.roll(features, shifts=-1, dims=0)
                
                return features * mask_even + features_rolled * mask_odd

            s_feat_mix = laser_mix_op(s_feat)
            
            t_feat_mix_target = laser_mix_op(t_feat)
            
            # Calculate losses
            # Distillation Loss
            kd_loss = F.mse_loss(s_feat, t_feat)
            
            # Consistency Loss
            cons_loss = F.mse_loss(s_feat_mix, t_feat_mix_target)
            
            w_kd = self.model_cfg.get('KD_LOSS_WEIGHT', 10.0)
            w_cons = self.model_cfg.get('CONS_WEIGHT', 5.0)

            distill_total_loss = (kd_loss * w_kd) + (cons_loss * w_cons)
            
            # Return losses
            ret_dict['loss'] += distill_total_loss

            tb_dict.update({
                'kd_loss': kd_loss.item(),
                'cons_loss': cons_loss.item(),
                'distill_total': distill_total_loss.item()
            })

        return ret_dict, tb_dict, disp_dict
