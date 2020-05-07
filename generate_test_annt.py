import os
import json
import time
import numpy as np
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

SYSTEM = 'windows'  # 'windows' 'linux'
DATASET = 'coco' # 'bdd','coco'
#####################################################################################
# Windows:
if SYSTEM == 'windows':
    # BDD100k
    if DATASET == 'bdd':
        annt_file = "F:/BDBerkeley/bdd100k/labels/bdd100k_labels_images_val.json"
        images_dir = "F:/BDBerkeley/bdd100k/images/100k/val"
        category_index = label_map_util.create_category_index_from_labelmap('bdd_label_map.pbtxt',
                                                                            use_display_name=False)
    # COCO2017
    elif DATASET == 'coco':
        annt_file = "F:/COCO2017/annotations/instances_val2017.json"
        images_dir = "F:/COCO2017/val2017"
        category_index = label_map_util.create_category_index_from_labelmap('mscoco_label_map.pbtxt',
                                                                            use_display_name=True)
    else:
        print('DATASET error. Seleccione "coco" o "bdd"')

#####################################################################################
# LINUX:
if SYSTEM == 'linux':
    # BDD100k
    if DATASET == 'bdd':
        annt_file = "/home/export/pfc/jsanexp/bdd100k/labels/bdd100k_labels_images_val.json"
        images_dir = "/home/export/pfc/jsanexp/bdd100k/images/100k/val"
        category_index = label_map_util.create_category_index_from_labelmap('bdd_label_map.pbtxt',
                                                                            use_display_name=False)
    # COCO2017
    elif DATASET == 'coco':
        annt_file = "/home/export/pfc/jsanexp/COCO2017/annotations/instances_val2017.json"
        images_dir = "/home/export/pfc/jsanexp/COCO2017/val2017"
        category_index = label_map_util.create_category_index_from_labelmap('mscoco_label_map.pbtxt',
                                                                            use_display_name=True)
    else:
        print('DATASET error. Seleccione "coco" o "bdd"')

#####################################################################################
num_img = 5

####################################################################################
gt_data = json.load(open(annt_file, 'r'))
image_list = []
image_paths = []
gt_val = []

if DATASET == 'bdd':
    for i in range(num_img):
        frame = gt_data[i]
        image_list.append(frame['name'])
        image_paths.append(images_dir + '/' + frame['name'])

        # Initialize labels
        labels = []
        for j, label in enumerate(frame['labels']):
            if 'box2d' not in label:
                continue
            else:
                label['score'] = float(1)
                labels.append(label)

        annot = {'name': frame['name'],
                  'labels': labels}

        num_labels = len(annot['labels'])
        tensor_dict = {}
        tensor_dict['categories'] = np.zeros(num_labels, dtype=int)
        tensor_dict['scores'] = np.zeros(num_labels)
        tensor_dict['boxes'] = np.zeros((num_labels, 4))
        for j, label in enumerate(annot['labels']):
            (x1, y1, x2, y2) = (label['box2d']['x1'], label['box2d']['y1'], label['box2d']['x2'], label['box2d']['y2'])
            for k in range(len(category_index)):
                if category_index[k+1]['name'] == label['category']:
                    category_id = category_index[k+1]['id']
                    break
            tensor_dict['categories'][j] = int(category_id)
            tensor_dict['scores'][j] = float(1)
            tensor_dict['boxes'][j, :] = (y1, x1, y2, x2)  # Tensorflow format

        gt_val.append(annot)

        # Save Detections on Image file
        # Visualization of the results of a detection.
        img = Image.open(image_paths[i])
        (im_width, im_height) = img.size
        image_np = np.array(img.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            tensor_dict['boxes'],
            tensor_dict['categories'],
            tensor_dict['scores'],
            category_index,
            max_boxes_to_draw=100,
            use_normalized_coordinates=False,
            line_thickness=2,
            skip_scores=True)

        result_image = Image.fromarray(image_np)
        if not os.path.isdir(os.getcwd() + '/gt_bdd'):
            os.makedirs(os.getcwd() + '/gt_bdd')
        result_image.save(os.getcwd() + '/gt_bdd/{}'.format(image_list[i]))

    with open("bdd_test_images_annt.json", "w+") as test_json:
        json.dump(gt_val, test_json, indent=4)


if DATASET == 'coco':
    images = gt_data['images']
    annotations_index = {}

    if 'annotations' in gt_data:
        for annotation in gt_data['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations_index:
                annotations_index[image_id] = []
            annotations_index[image_id].append(annotation)

    for image in images:
        image_id = image['id']
        if image_id not in annotations_index:
            annotations_index[image_id] = []

    for i in range(num_img):
        image = images[i]
        image_id = image['id']
        annotations_list = annotations_index[image['id']]
        labels = []
        tensor_dict = {}
        tensor_dict['categories'] = np.zeros(len(annotations_list), dtype=int)
        tensor_dict['scores'] = np.zeros(len(annotations_list))
        tensor_dict['boxes'] = np.zeros((len(annotations_list), 4))
        for j, annotation in enumerate(annotations_list):
            (x, y, width, height) = tuple(annotation['bbox'])
            box2d = {'x1': x,
                     'y1': y,
                     'x2': x + width,
                     'y2': y + height}
            category_id = annotation['category_id']
            category = category_index[category_id]['name']
            label_ = {'category': category,
                      'box2d': box2d,
                      'score': float(1)
                      }
            tensor_dict['categories'][j] = int(category_id)
            tensor_dict['scores'][j] = float(1)
            tensor_dict['boxes'][j, :] = (y, x, y + height, x + width) # Tensorflow format
            labels.append(label_)

        annot = {'name': image['file_name'],
                 'labels': labels}

        gt_val.append(annot)

        # Save Detections on Image file
        # Visualization of the results of a detection.
        img = Image.open(images_dir + '/' + image['file_name'])
        (im_width, im_height) = img.size
        image_np = np.array(img.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            tensor_dict['boxes'],
            tensor_dict['categories'],
            tensor_dict['scores'],
            category_index,
            max_boxes_to_draw=100,
            use_normalized_coordinates=False,
            line_thickness=2,
            skip_scores=True)

        result_image = Image.fromarray(image_np)
        if not os.path.isdir(os.getcwd() + '/gt_coco'):
            os.makedirs(os.getcwd() + '/gt_coco')
        result_image.save(os.getcwd() + '/gt_coco/{}'.format(image['file_name']))

    with open("coco_test_images_annt.json", "w+") as test_json:
        json.dump(gt_val, test_json, indent=4)


# # Generate Structure:
# box2d = {'x1': None,
#          'y1': None,
#          'x2': None,
#          'y2': None}
# label_ = {'category': None,
#           'box2d': box2d}
# labels = []
# labels.append(label_)
#
# annot_ = {'name': None,
#           'labels': labels}
#
# annot = []
# annot.append(annot_)

# #Obtain class from class name:
# for id in range(10):
#     id += 1 #First id is 1 not 0
#     if (category_index[id]['name'] == label['category']):
#         break
#     id -= 1 #Restore iterand

#det_result = json.load(open('results.json', 'w+'))

# det = {'name': image_path,
#        'label': {
#            'category': label['category'],
#            'bbox': [xy['x1'], xy['y1'], xy['x2'], xy['y2']],
#            'score': 1}
#        }
#
# output_dict['detection_boxes']
# output_dict['detection_classes']
# output_dict['detection_scores']
#
# PATH_TO_IMAGES_DIR = '/home/export/pfc/jsanexp/bdd100k/images/100k/val'
# num_img = 5
# list_images = os.listdir(PATH_TO_IMAGES_DIR)
# TEST_IMAGE_PATHS=[]
# for count in range(num_img):
#     TEST_IMAGE_PATHS.append(PATH_TO_IMAGES_DIR + '/' + list_images[count])