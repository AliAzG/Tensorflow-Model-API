import os, argparse
from io import BytesIO
import falcon
import random
import tensorflow as tf
import numpy as np
from PIL import Image
from werkzeug.http import parse_options_header
from werkzeug.formparser import MultiPartParser, default_stream_factory
from werkzeug.datastructures import FileStorage
from keras_retinanet.utils.image import  read_image_bgr,preprocess_image, resize_image
import json
import requests
import time
from falcon_multipart.middleware import MultipartMiddleware

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name=""
            )
    return graph
graph = load_graph('./model/new_model.pb') 
x = graph.get_tensor_by_name('input_1:0')
box = graph.get_tensor_by_name('filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0')
prob = graph.get_tensor_by_name("filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0")
cat = graph.get_tensor_by_name("filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0")
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
sess_config = tf.ConfigProto(gpu_options=gpu_options)
sess_config.gpu_options.allow_growth = True
persistent_sess = tf.Session(graph=graph, config=sess_config)
class TF_API(object):
    def __init__(self, persistent_sess):
        self._persistent_sess =  persistent_sess
    def on_post(self, request, response):
        start = time.time()
        im = request.get_param('file')
        # only if you need the image data
        raw = im.file.read()
        n = Image.open(BytesIO(raw)).convert('RGB') 
        open_cv_image = np.array(n) 
        image = preprocess_image(open_cv_image)
        image, scale = resize_image(image)
        boxes = self._persistent_sess.run(box, {x: np.expand_dims(image, axis=0)}).tolist()
        probability = self._persistent_sess.run(prob, {x: np.expand_dims(image, axis=0)}).tolist()
        category = self._persistent_sess.run(cat, {x: np.expand_dims(image, axis=0)}).tolist()
        pro = []
        final_boxes = []
        final_category = []
        area = []
        centers = []
        for i in range(0, len(probability[0])):
            if probability[0][0] < 0.5:
                return { 'Status': 'Nothing Found' }
            elif probability[0][i] >= 0.5:
                pro.append(probability[0][i])
                final_boxes.append([x / scale for x in boxes[0][i]])
                final_category.append(catogery_dict[category[0][i]])
        for j in range(0, len(final_boxes)):
            area.append([(final_boxes[j][2]-final_boxes[j][0])*(final_boxes[j][3]-final_boxes[j][1])])
            centers.append([(final_boxes[j][0]+(final_boxes[j][2]-final_boxes[j][0])/2), (final_boxes[j][1]+(final_boxes[j][3]-final_boxes[j][1])/2)])
        print("Time spent handling the request: %f" % (time.time() - start))
        result = {'box': final_boxes, 'category': final_category, 'probability': pro, 'area': area, 'centers': centers}
        response.media = result

api = falcon.API(middleware=[MultipartMiddleware()])
api.add_route('/api/tfapi', TF_API(persistent_sess))
