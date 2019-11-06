import os, argparse
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from PIL import Image
from keras_retinanet.utils.image import  read_image_bgr,preprocess_image, resize_image
import json
import requests
import time

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

app = Flask(__name__)
cors = CORS(app)
@app.route("/api/tf_api", methods=['POST'])
def tf_api():
    start = time.time()
    image = read_image_bgr(request.files['file'])
    image = preprocess_image(image)
    image, scale = resize_image(image)
    boxes = persistent_sess.run(box, {x: np.expand_dims(image, axis=0)}).tolist()
    probability = persistent_sess.run(prob, {x: np.expand_dims(image, axis=0)}).tolist()
    category = persistent_sess.run(cat, {x: np.expand_dims(image, axis=0)}).tolist()
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
    return {'box': final_boxes, 'category': final_category, 'probability': pro, 'area': area, 'centers': centers}





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="5002", type=str, help="Running port")
    parser.add_argument("--host", default="0.0.0.0", type=str, help="Running host")
    parser.add_argument("--model", default="./model/new_model.pb", type=str, help="Frozen model")
    args = parser.parse_args()
    graph = load_graph(args.model) 
    x = graph.get_tensor_by_name('input_1:0')
    box = graph.get_tensor_by_name('filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0')
    prob = graph.get_tensor_by_name("filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0")
    cat = graph.get_tensor_by_name("filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    sess_config.gpu_options.allow_growth = True
    persistent_sess = tf.Session(graph=graph, config=sess_config)
    app.run(host=args.host, port=args.port)
