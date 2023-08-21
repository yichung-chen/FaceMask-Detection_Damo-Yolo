#!/usr/bin/env python3

import os

from damo.config import Config as MyConfig
import yaml
import sys,os
from easydict import EasyDict as easydict
import ast

sys.path.append(os.getcwd())

class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()


        with open('/home/ubuntu/Mask-Det_Damo_Yolo/customized/hyperparameters/parameters.yaml', 'r') as para:
            hyperparameters = yaml.load(para, Loader=yaml.FullLoader)
            hyperparameters = hyperparameters['hyperparameters']['training']
            hyperparameters = easydict(hyperparameters)

        self.miscs.exp_name = os.path.split(
            os.path.realpath(__file__))[1].split('.')[0]
        self.miscs.eval_interval_epochs = 10
        self.miscs.ckpt_interval_epochs = 100
        # optimizer
        self.train.batch_size = hyperparameters.default_para.batch.value
        self.train.total_epochs = hyperparameters.default_para.epochs.value

        self.train.base_lr_per_img = hyperparameters.advanced_para.learning_rate.value / 64
        self.train.min_lr_ratio = hyperparameters.advanced_para.min_lr_ratio.value
        self.train.weight_decay = hyperparameters.advanced_para.weight_decay.value
        self.train.momentum = hyperparameters.advanced_para.weight_decay.value
        self.train.no_aug_epochs = hyperparameters.advanced_para.no_aug_epochs.value
        self.train.warmup_epochs = hyperparameters.advanced_para.warmup_epochs.value
        self.train.height_image =  hyperparameters.advanced_para.image_size.Height.value
        self.train.width_image =  hyperparameters.advanced_para.image_size.Width.value
        self.train.image_size = self.train.height_image if self.train.height_image > self.train.width_image else self.train.width_image
        # augment
        self.train.augment.transform.image_max_range = (self.train.image_size, self.train.image_size)
        self.train.augment.mosaic_mixup.mixup_prob = hyperparameters.advanced_para.mixup_prob.value
        self.train.augment.mosaic_mixup.degrees = hyperparameters.advanced_para.degrees.value
        self.train.augment.mosaic_mixup.translate = hyperparameters.advanced_para.translate.value
        self.train.augment.mosaic_mixup.shear = hyperparameters.advanced_para.shear.value
        self.train.augment.mosaic_mixup.mosaic_scale = (0.1, 2.0)


        # dataset
        self.dataset.train_ann = ('coco_2017_train', )
        self.dataset.val_ann = ('coco_2017_val', )

        with open('/home/ubuntu/Mask-Det_Damo_Yolo/customized/results/weights/leadtek.names','r') as cls:
            cls_info = cls.readlines()
            nc = cls_info[0].splitlines()[0].split(':')[1]
            classes_name = cls_info[1].split(':')[1]
            classes_name = ast.literal_eval(classes_name)

        self.dataset.class_names = classes_name
        self.dataset.nc = nc


        self.train.finetune_path='/home/ubuntu/Mask-Det_Damo_Yolo/customized/models/pretrained/damoyolo_tinynasL35_M.pth'
        # backbone
        structure = self.read_structure(
            '/home/ubuntu/Mask-Det_Damo_Yolo/customized/models/damo-yolo/damo/base_models/backbones/nas_backbones/tinynas_L35_kxkx.txt')
        TinyNAS = {
            'name': 'TinyNAS_csp',
            'net_structure_str': structure,
            'out_indices': (2, 3, 4),
            'with_spp': True,
            'use_focus': True,
            'act': 'silu',
            'reparam': True,
        }

        self.model.backbone = TinyNAS

        GiraffeNeckV2 = {
            'name': 'GiraffeNeckV2',
            'depth': 1.5,
            'hidden_ratio': 1.0,
            'in_channels': [128, 256, 512],
            'out_channels': [128, 256, 512],
            'act': 'silu',
            'spp': False,
            'block_name': 'BasicBlock_3x3_Reverse',
        }

        self.model.neck = GiraffeNeckV2

        ZeroHead = {
            'name': 'ZeroHead',
            'num_classes': int(self.dataset.nc),
            'in_channels': [128, 256, 512],
            'stacked_convs': 0,
            'reg_max': 16,
            'act': 'silu',
            'nms_conf_thre': 0.05,
            'nms_iou_thre': 0.7,
            'legacy': False,
        }
        self.model.head = ZeroHead
        self.dataset.class_names = classes_name
        #self.dataset.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
