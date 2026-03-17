import torch
import torch.nn as nn
import torch.nn.functional as F
from .detector3d_template import Detector3DTemplate
from .pointpillar import PointPillar
from .centerpoint import CenterPoint
from ...config import cfg_from_yaml_file
from easydict import EasyDict
from ...utils import common_utils

class DistillPointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg, num_class, dataset)
        
        # 1. Build student model
        self.student = PointPillar(model_cfg, num_class, dataset)
        
        # 2. Build teacher model
        teacher_cfg_path = model_cfg.TEACHER_CFG_FILE
        teacher_ckpt_path = model_cfg.TEACHER_CKPT
        
        logger = common_utils.create_logger(log_file=None, rank=0)
        logger.info(f"==> Loading Teacher Config from: {teacher_cfg_path}")
        
        teacher_cfg_root = EasyDict()
        cfg_from_yaml_file(teacher_cfg_path, teacher_cfg_root)
        
        self.teacher = CenterPoint(teacher_cfg_root.MODEL, num_class, dataset)
        
        logger.info(f"==> Loading Teacher Weights from: {teacher_ckpt_path}")
        self.teacher.load_params_from_file(filename=teacher_ckpt_path, logger=logger, to_cpu=False)
        
        # Freeze Teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def forward(self, batch_dict):
        student_outs = self.student(batch_dict)
       
        if self.training:
            ret_dict, tb_dict, disp_dict = student_outs
            
            if 'spatial_features_2d' in batch_dict:
                student_feature = batch_dict['spatial_features_2d']
            else:
                student_feature = None
            
            # Running the teacher model & calculating distillation loss
            if student_feature is not None:
                with torch.no_grad():
                    self.teacher.eval()
                    # Run the teacher model to extract features
                    self.teacher(batch_dict)
                    teacher_feature = batch_dict['spatial_features_2d']
                
                distill_loss = F.mse_loss(student_feature, teacher_feature)

                kd_loss_weight = self.model_cfg.get('KD_LOSS_WEIGHT', 10.0)
                distill_loss = distill_loss * kd_loss_weight
                
                ret_dict['loss'] += distill_loss
               
                tb_dict['kd_loss'] = distill_loss.item()
            
                tb_dict['student_task_loss'] = ret_dict['loss'].item() - distill_loss.item()
            
            return ret_dict, tb_dict, disp_dict
            
        else:
            return student_outs

    def get_training_loss(self):
        return self.student.get_training_loss()