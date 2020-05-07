MODEL_LIST = ['faster_rcnn_nas_coco_2018_01_28',  # OOM
              'faster_rcnn_nas_lowproposals_coco_2018_01_28',  # OOM
              'faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28',  # OOM
              'faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28',  # OOM
#              'mask_rcnn_resnet101_atrous_coco_2018_01_28',  # No acepta batch
#              'mask_rcnn_resnet50_atrous_coco_2018_01_28',  # No acepta batch
#              'mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28',  # No acepta batch
#              'mask_rcnn_inception_v2_coco_2018_01_28',  # No acepta batch
              'ssd_mobilenet_v1_coco_2018_01_28',
              'ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03',
              'ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03',
              'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03',
              'ssd_mobilenet_v2_coco_2018_03_29',
              'ssdlite_mobilenet_v2_coco_2018_05_09',
              'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03',
              'ssd_inception_v2_coco_2018_01_28',
              'faster_rcnn_resnet50_coco_2018_01_28',
              'faster_rcnn_resnet50_lowproposals_coco_2018_01_28',
              'faster_rcnn_resnet101_coco_2018_01_28',
              'faster_rcnn_resnet101_lowproposals_coco_2018_01_28',
              'faster_rcnn_inception_v2_coco_2018_01_28',
              'rfcn_resnet101_coco_2018_01_28']
#              'ssd_mobilenet_v2_quantized_300x300_coco_2018_09_14',  # (TFLITE)
#              'ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18',  # (TFLITE)
#              'ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18',  # (TFLITE)

MODEL_NAMES = ['faster_rcnn_nas', # OOM
              'faster_rcnn_nas_lowproposals',  # OOM
              'faster_rcnn_inception_resnet_v2_atrous',  # OOM
              'faster_rcnn_inception_resnet_v2_atrous_lowproposals',  # OOM
#              'mask_rcnn_resnet101_atrous',  # No acepta batch
#              'mask_rcnn_resnet50_atrous',  # No acepta batch
#              'mask_rcnn_inception_resnet_v2_atrous',  # No acepta batch
#              'mask_rcnn_inception_v2',  # No acepta batch
              'ssd_mobilenet_v1',
              'ssd_mobilenet_v1_0.75_depth_300x300',
              'ssd_mobilenet_v1_ppn_shared_box_predictor_300x300',
              'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640',
              'ssd_mobilenet_v2',
              'ssdlite_mobilenet_v2',
              'ssd_resnet50_v1_fpn_shared_box_predictor_640x640',
              'ssd_inception_v2',
              'faster_rcnn_resnet50',
              'faster_rcnn_resnet50_lowproposals',
              'faster_rcnn_resnet101',
              'faster_rcnn_resnet101_lowproposals',
              'faster_rcnn_inception_v2',
              'rfcn_resnet101']
#              'ssd_mobilenet_v2_quantized_300x300',  # (TFLITE)
#              'ssd_mobilenet_v1_quantized_300x300',  # (TFLITE)
#              'ssd_mobilenet_v1_0.75_depth_quantized_300x300',  # (TFLITE)