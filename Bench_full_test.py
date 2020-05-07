# Benchmark for object detection models
"""
Benchmark for testing inference time and accuracy of several models using
Tensorflow Object Detection API

Author: Jorge Sánchez-Alor Expósito
Based on code from Tensorflow Object Detection API Tutorial
and bdd-data github repositories:
https://github.com/tensorflow/models/tree/master/research/object_detection
https://github.com/ucbdrive/bdd-data
"""
# coding: utf-8
# IMPORTS
from __future__ import division, print_function
import os
import json
import tarfile
import time

import numpy as np
import six.moves.urllib as urllib
import tensorflow as tf

from distutils.version import StrictVersion
from collections import defaultdict

from matplotlib import pyplot as plt
from PIL import Image
import cv2

from convert_detections import convert_detections

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

#OBJECT DETECTION IMPORTS
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


import sys
import argparse

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box

from model import yolov3
from convert_detections import convert_detections


parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--class_name_path", type=str, default="./data/coco.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt",
                    help="The path of the weights to restore.")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)

#HELPER FUNCTIONS CODE:

#LOAD IMAGE INTO NP ARRAY
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

#LOAD IMAGE IN A BATCH
def load_images(img_names, model_size):
    """Loads images in a 4D array.

    Args:
        img_names: A list of images names.
        model_size: The input size of the model.
        data_format: A format for the array returned
            ('channels_first' or 'channels_last').

    Returns:
        A 4D NumPy array.
    """
    imgs = []

    for img_name in img_names:
        img = Image.open(img_name)
        img = img.resize(size=model_size)
        img = np.array(img, dtype=np.float32)
        img = np.expand_dims(img, axis=0)
        imgs.append(img)

    imgs = np.concatenate(imgs)

    return imgs

# INFERENCE FUNCTION
def run_inference_for_single_image(image, tensor_dict, image_tensor):
    # Run inference
    iterations = 5
    inf_times = np.zeros(iterations)
    for i in range(iterations):
        start = time.time()
        output_dict = sess.run(tensor_dict,
                               feed_dict={image_tensor: np.expand_dims(image, 0)})
        end = time.time()
        elapsed = (end - start)*1000
        inf_times[i] = elapsed
        print('Detection time ({}):'.format(i), elapsed, ' ms')

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict, inf_times

# DETECTION IN VIDEO
def detect_in_video(detection_graph, category_index):
    # VideoWriter is the responsible of creating a copy of the video
    # used for the detections but with the detections overlays. Keep in
    # mind the frame size has to be the same as original video.
    #out = cv2.VideoWriter('/home/export/pfc/jsanexp/bdd100k/videos/Detection.avi', cv2.VideoWriter_fourcc(
    #    'm', 'p', 'g', '2'), 30, (720, 1280))
    out = cv2.VideoWriter('/home/export/pfc/jsanexp/bdd100k/videos/Detection.avi', cv2.VideoWriter_fourcc(
        'm', 'p', 'g', '2'), 30, (720, 1280))

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object
            # was detected.
            detection_boxes = detection_graph.get_tensor_by_name(
                'detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class
            # label.
            detection_scores = detection_graph.get_tensor_by_name(
                'detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name(
                'detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            cap = cv2.VideoCapture('/home/export/pfc/jsanexp/bdd100k/videos/cabc30fc-e7726578.mov')

            while (cap.isOpened()):
                # Read the frame
                ret, frame = cap.read()

                # Recolor the frame. By default, OpenCV uses BGR color space.
                # This short blog post explains this better:
                # https://www.learnopencv.com/why-does-opencv-use-bgr-color-format/
                color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_np_expanded = np.expand_dims(color_frame, axis=0).transpose([0, 2, 1, 3])

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores,
                     detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # Visualization of the results of a detection.
                # note: perform the detections using a higher threshold
                vis_util.visualize_boxes_and_labels_on_image_array(
                    color_frame.transpose([1, 0, 2]),
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8,
                    min_score_thresh=.20)

                # cv2.imshow('frame', color_frame)
                output_rgb = cv2.cvtColor(color_frame.transpose([1, 0, 2]), cv2.COLOR_RGB2BGR)
                out.write(output_rgb)

            out.release()
            cap.release()
            # cv2.destroyAllWindows()

# EVALUATION FUNCTIONS:
def evaluate_detection(gt, pred):

    cat_gt = group_by_key(gt, 'category')
    cat_pred = group_by_key(pred, 'category')
    cat_list = sorted(cat_gt.keys())
    thresholds = [0.75]
    aps = np.zeros((len(thresholds), len(cat_list)))
    for i, cat in enumerate(cat_list):
        if cat in cat_pred:
            r, p, ap = cat_pc(cat_gt[cat], cat_pred[cat], thresholds)
            aps[:, i] = ap
    aps *= 100
    mAP = np.mean(aps)
    return mAP, aps.flatten().tolist()

def get_ap(recalls, precisions):
    # correct AP calculation
    # first append sentinel values at the end
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))

    # compute the precision envelope
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(recalls[1:] != recalls[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
    return ap

def group_by_key(detections, key):
    groups = defaultdict(list)
    for d in detections:
        groups[d[key]].append(d)
    return groups

def cat_pc(gt, predictions, thresholds):
    """
    Implementation refers to https://github.com/rbgirshick/py-faster-rcnn
    """
    num_gts = len(gt)
    image_gts = group_by_key(gt, 'name')
    image_gt_boxes = {k: np.array([[float(z) for z in b['bbox']]
                                   for b in boxes])
                      for k, boxes in image_gts.items()}
    image_gt_checked = {k: np.zeros((len(boxes), len(thresholds)))
                        for k, boxes in image_gts.items()}
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)

    # go down dets and mark TPs and FPs
    nd = len(predictions)
    tp = np.zeros((nd, len(thresholds)))
    fp = np.zeros((nd, len(thresholds)))
    for i, p in enumerate(predictions):
        box = p['bbox']
        ovmax = -np.inf
        jmax = -1
        try:
            gt_boxes = image_gt_boxes[p['name']]
            gt_checked = image_gt_checked[p['name']]
        except KeyError:
            gt_boxes = []
            gt_checked = None

        if len(gt_boxes) > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(gt_boxes[:, 0], box[0])
            iymin = np.maximum(gt_boxes[:, 1], box[1])
            ixmax = np.minimum(gt_boxes[:, 2], box[2])
            iymax = np.minimum(gt_boxes[:, 3], box[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((box[2] - box[0] + 1.) * (box[3] - box[1] + 1.) +
                   (gt_boxes[:, 2] - gt_boxes[:, 0] + 1.) *
                   (gt_boxes[:, 3] - gt_boxes[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        for t, threshold in enumerate(thresholds):
            if ovmax > threshold:
                if gt_checked[jmax, t] == 0:
                    tp[i, t] = 1.
                    gt_checked[jmax, t] = 1
                else:
                    fp[i, t] = 1.
            else:
                fp[i, t] = 1.

    # compute precision recall
    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)
    recalls = tp / float(num_gts)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = np.zeros(len(thresholds))
    for t in range(len(thresholds)):
        ap[t] = get_ap(recalls[:, t], precisions[:, t])

    return recalls, precisions, ap


#########################################################################
#########################################################################
#Detection Results structure:
box2d = {'x1': None,
         'y1': None,
         'x2': None,
         'y2': None}
labels = [{'category': None,
          'box2d': box2d}]
annot = [{'name': None,
         'labels': labels}]

#IMAGE SET

images_dir = 'F:/COCO2017/val2017'
#images_dir = "/home/export/pfc/jsanexp/COCO2017/val2017"
annt_file = "./coco_test_images_annt.json"
gt_data = json.load(open(annt_file, 'r'))
image_list = []
TEST_IMAGE_PATHS = []
for frame in gt_data:
    image_list.append(frame['name'])
    TEST_IMAGE_PATHS.append(images_dir + '/' + frame['name'])

#MODEL PREPARATION
#VARIABLES
from model_list_selected import MODEL_LIST
#from model_list import MODEL_LIST
models_path = 'D:/Anaconda/envs/tf12/Projects/models/'
#models_path = '/home/export/pfc/jsanexp/Projects/models/'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './mscoco_label_map.pbtxt'

#LOADING LABEL MAP
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

frozen_list = []
for model in MODEL_LIST:
    # What model to download.
    MODEL_NAME = model

    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = models_path + MODEL_NAME + '/frozen_inference_graph.pb'
    frozen_list.append(PATH_TO_FROZEN_GRAPH)

    #DOWNLOAD MODEL
    if not os.path.isfile(PATH_TO_FROZEN_GRAPH):
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, models_path + MODEL_FILE)
        tar_file = tarfile.open(models_path + MODEL_FILE)
        tar_file.extractall(models_path)

#DETECTION
# Prepare save data
results_file = open('Results_coco.txt', 'w+')
y_axis_mAP = np.zeros(len(frozen_list)+1)
y_axis_mIT = np.zeros(len(frozen_list)+1)
y_axis_mAPperIT = np.zeros(len(frozen_list)+1)

#LOAD A (FROZEN) TENSORFLOW MODEL INTO MEMORY
for idx, frozen in enumerate(frozen_list):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(frozen, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')


            #DETECTION LOOP
            model = MODEL_LIST[frozen_list.index(frozen)]
            print('Runing Inference for model {}...'.format(model))
            if not os.path.isdir(os.getcwd() + '/results_coco/{}'.format(model)):
                os.makedirs(os.getcwd() + '/results_coco/{}'.format(model))

            # Initialize arrays:
            annot = []
            first_it = np.zeros(len(TEST_IMAGE_PATHS))
            mean_it = np.zeros(len(TEST_IMAGE_PATHS))
            total_it = np.zeros(len(TEST_IMAGE_PATHS))

            # Run detection for every image:
            for index, image_path in enumerate(TEST_IMAGE_PATHS):
                image = Image.open(image_path)
                im_width, im_height = image.size
                image_name = image_list[index]

                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                start = time.time()
                output_dict, inf_times = run_inference_for_single_image(image_np, tensor_dict, image_tensor)
                end = time.time()
                total_time = (end-start)*1000
                print('Total test time for ' + model + ': ', total_time,' ms')

                # Time registering:
                first_it[index] = inf_times[0]
                mean_it[index] = np.mean(inf_times[1:])
                total_it[index] = total_time

                # Detection Evaluation:
                # Initialize labels
                labels = []
                # Adapt format
                for i in range(output_dict['num_detections']):
                    # Obtain class name from class id:
                    class_id = output_dict['detection_classes'][i]
                    category = category_index[class_id]['name']
                    # Obtain box coordinates
                    ymin, xmin, ymax, xmax = output_dict['detection_boxes'][i]
                    # Undo Normalization
                    box2d = {'x1': xmin * im_width,
                             'y1': ymin * im_height,
                             'x2': xmax * im_width,
                             'y2': ymax * im_height}
                    # Obtain score
                    score = output_dict['detection_scores'][i]
                    label_ = {'category': category,
                              'box2d': box2d,
                              'score': float(score)} # float because float32 is not json serializable
                    labels.append(label_)

                annot_ = {'name': image_list[index],
                          'labels': labels}

                annot.append(annot_)

                # Save Detections on Image file
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    line_thickness=5)

                result_image = Image.fromarray(image_np)
                result_image.save(os.getcwd() + '/results_coco/{}/{}'.format(model, image_name))

            # Save detections from image set, and run evaluation:
            with open(os.getcwd() + '/results_coco/' + model + '/' + '{}_results_coco_images_annt.json'.format(model),
                "w+") as results_json:
                json.dump(annot, results_json, indent=4)
            detections_file = './results_coco/' + model + '/' + '{}_results_coco_images_annt.json'.format(model)
            gt_det = convert_detections(annt_file)
            rs_det = convert_detections(detections_file)
            mAP, aps = evaluate_detection(gt_det, rs_det)

            print('{}    {}    {}    {}    {}'.format(np.mean(first_it), np.mean(mean_it), np.mean(total_it),
                                                      mAP, model))
            print('{}    {}    {}    {}    {}'.format(np.mean(first_it), np.mean(mean_it), np.mean(total_it),
                                                      mAP, model), file=results_file)

            # Save Info for representation:
            y_axis_mAP[idx] = mAP
            y_axis_mIT[idx] = np.mean(mean_it)
            y_axis_mAPperIT[idx] = mAP/np.mean(mean_it)

#LOAD YOLOv3 MODEL
with tf.Session() as sess:
    input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
    yolo_model = yolov3(args.num_class, args.anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

    pred_scores = pred_confs * pred_probs
    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=50, score_thresh=0.4,
                                    iou_thresh=0.5)

    saver = tf.train.Saver()
    saver.restore(sess, args.restore_path)

    # DETECTION LOOP FOR ALL IMAGES:
    annot = []
    first_it = np.zeros(len(TEST_IMAGE_PATHS))
    mean_it = np.zeros(len(TEST_IMAGE_PATHS))
    total_it = np.zeros(len(TEST_IMAGE_PATHS))

    for index, input_image in enumerate(TEST_IMAGE_PATHS):
        # DETECTION
        print('Runing Inference for model YOLOv3...'.format(model))
        iterations = 5
        inf_times = np.zeros(iterations)
        start = time.time()
        for i in range(iterations):
            img_ori = cv2.imread(input_image)
            height_ori, width_ori = img_ori.shape[:2]
            # Start counting time:
            t0 = time.time()
            # We need to resize the images manually (in the API is included inside the model)
            # RESIZE
            img = cv2.resize(img_ori, tuple(args.new_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(img, np.float32)
            img = img[np.newaxis, :] / 255.

            boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
            inf_times[i] = (time.time() - t0)*1000
            print('Detection time ({}):'.format(i), inf_times[i], ' ms')

        end = time.time()
        total_time = (end - start) * 1000
        print('Total test time for YOLOv3: ', total_time, ' ms')

        # Time registering:
        first_it[index] = inf_times[0]
        mean_it[index] = np.mean(inf_times[1:])
        total_it[index] = total_time

        # Rescale the coordinates to the original image
        boxes_[:, 0] *= (width_ori / float(args.new_size[0]))
        boxes_[:, 2] *= (width_ori / float(args.new_size[0]))
        boxes_[:, 1] *= (height_ori / float(args.new_size[1]))
        boxes_[:, 3] *= (height_ori / float(args.new_size[1]))

        # Detection Evaluation
        # Initialize labels
        label = []
        for i in range(len(labels_)):
            # Obtain class name from class id:
            class_id = labels_[i]
            category = args.classes[labels_[i]]
            # Obtain box coordinates
            xmin, ymin, xmax, ymax = boxes_[i]
            # Fill detection dict
            box2d = {'x1': float(xmin),  # float because float32 is not json serializable
                     'y1': float(ymin),
                     'x2': float(xmax),
                     'y2': float(ymax)}
            # Obtain score
            score = scores_[i]
            label_ = {'category': category,
                      'box2d': box2d,
                      'score': float(score)}  # float because float32 is not json serializable
            label.append(label_)

        annot_ = {'name': image_list[index],
                  'labels': label}

        annot.append(annot_)

        # Draw boxes on each image:
        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i]
            plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]], color=color_table[labels_[i]])
        #    cv2.imshow('Detection result', img_ori) #Problem with OpenCv Implementation in lab

        # Save Image with boxes:
        if not os.path.isdir(os.getcwd() + '/results_coco/yolov3'):
            os.makedirs(os.getcwd() + '/results_coco/yolov3')
        cv2.imwrite(os.getcwd() + '/results_coco/yolov3/{}.jpg'.format(image_list[index]), img_ori)
        #cv2.waitKey(0) #Problem with OpenCv Implementation in lab

    # Save detections from image set, and run evaluation:
    with open(os.getcwd() + '/results_coco/yolov3/yolov3_results_coco_images_annt.json', "w+") as results_json:
        json.dump(annot, results_json, indent=4)
    detections_file = './results_coco/yolov3/yolov3_results_coco_images_annt.json'
    gt_det = convert_detections(annt_file)
    rs_det = convert_detections(detections_file)
    mAP, aps = evaluate_detection(gt_det, rs_det)

    print('{}    {}    {}    {}    {}'.format(np.mean(first_it), np.mean(mean_it), np.mean(total_it),
                                              mAP, 'YOLOv3'))
    print('{}    {}    {}    {}    {}'.format(np.mean(first_it), np.mean(mean_it), np.mean(total_it),
                                              mAP, 'YOLOv3'), file=results_file)
    y_axis_mAP[idx+1] = mAP
    y_axis_mIT[idx+1] = np.mean(mean_it)
    y_axis_mAPperIT[idx+1] = mAP/np.mean(mean_it)

#SAVE INFROMATION ON TEXT FILE
results_file.close()

# REPRESENTATION:
from model_list_selected import MODEL_NAMES
#from model_list import MODEL_NAMES
MODEL_NAMES.append('YOLOv3')
# Generate the comparison bar graph
plt.figure(figsize=(16,9))
x_axis = np.arange(len(MODEL_NAMES))
plt.bar(x_axis, y_axis_mIT, color=(0, 0, 1), rasterized=True)
plt.xlabel('Model', fontsize=10)
plt.ylabel('Mean Inference Time (ms)', fontsize=12)
plt.xticks(x_axis, MODEL_NAMES, fontsize=9, rotation=30)
plt.title('Inference Time for different Object Detection Models', fontsize=16)
plt.tight_layout()
plt.savefig('Inference_Time.eps')
plt.savefig('Inference_Time.png')

plt.clf()
plt.bar(x_axis, y_axis_mAP, color=(1, 0.5, 0), rasterized=True)
plt.xlabel('Model', fontsize=10)
plt.ylabel('Mean Average Precision', fontsize=12)
plt.xticks(x_axis, MODEL_NAMES, fontsize=9, rotation=30)
plt.title('mAP for Object Detection Models', fontsize=16)
plt.tight_layout()
plt.savefig('mAP.eps')
plt.savefig('mAP.png')

plt.clf()
plt.bar(x_axis, y_axis_mAPperIT, color=(0, 0.8, 0), rasterized=True)
plt.xlabel('Model', fontsize=10)
plt.ylabel('mAP/IT', fontsize=12)
plt.xticks(x_axis, MODEL_NAMES, fontsize=9, rotation=30)
plt.title('mAP/IT for Object Detection Models', fontsize=16)
plt.tight_layout()
plt.savefig('mAPperIT.eps')
plt.savefig('mAPperIT.png')

plt.clf()
plt.scatter(y_axis_mIT, y_axis_mAP, c=y_axis_mAP, s=500, marker='^', cmap='Set1')
plt.xlabel('mean IT', fontsize=12)
plt.ylabel('mAP', fontsize=12)
ax = plt.gca()
for i, txt in enumerate(MODEL_NAMES):
    ax.annotate(txt, (y_axis_mIT[i], y_axis_mAP[i]), fontsize=9)
#plt.xlim(0, 350)
plt.title('mAP against mean IT for Object Detection Models', fontsize=16)
plt.tight_layout()
plt.savefig('mAPvsIT.eps')
plt.savefig('mAPvsIT.png')


