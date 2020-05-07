import json
import time

def convert_detections(data_file):
    data = json.load(open(data_file, 'r'))
    converted_data = []
    for frame in data:
        labels = frame['labels']
        for label in labels:
            bbox = label['box2d']
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            det_dict = {'name': frame['name'],
                        "category": label['category'],
                        "bbox": [x1, y1, x2, y2],
                        "score": label['score']
                        }
            converted_data.append(det_dict)

    return converted_data