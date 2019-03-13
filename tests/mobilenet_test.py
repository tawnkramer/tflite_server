import os
import sys
import json
from subprocess import Popen
import shlex
from io import BytesIO
import time

from PIL import Image
import numpy as np
import zmq

model_filename = "mobilenet_v2_1.0_224_quant.tflite"
model_url = "http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz"
class_filename = "labels.txt"
image_filename = "hammerhead.jpg"
image_url = "http://cdn.shopify.com/s/files/1/2281/5369/products/267929_3_1024x1024.jpg"
tflite_server_bin = '../build/tflite_serve'

port = 5555

def download_model_if_needed():
    if not os.path.exists(model_filename):
        os.system('wget %s' % model_url)
        os.system('tar xzf %s' % model_filename.replace('.tflite', '.tgz'))

def download_image_if_needed():
    if not os.path.exists(image_filename):
        os.system('wget %s' % image_url)
        os.system('mv 267929_3_1024x1024.jpg %s' % image_filename)

def img_to_binary(img, format='jpeg'):
    '''
    accepts: PIL image
    returns: binary stream (used to save to database)
    '''
    f = BytesIO()
    try:
        img.save(f, format=format)
    except Exception as e:
        raise e
    return f.getvalue()

def load_labels(filename):
    classes = {}
    with open(filename, "rt") as file:
        for line in file:
            tokens = line.strip().split(':')
            classes[int(tokens[0])] = tokens[1]
    return classes

class TFLiteServer():
    def __init__(self, path, model, port=5555):
        command = '%s --model %s --port=%d' % (path, model, port)
        args =  shlex.split(command)
        self.proc = Popen(args)        

    def stop(self):
        if self.proc is not None:
            print("stopping tflite server")
            self.proc.terminate()
            self.proc = None

    def __del__(self):
        self.stop()


download_model_if_needed()
download_image_if_needed()

if not os.path.exists(model_filename):
    print(model_filename, " wasn't available.")
    exit(-1)

classes = load_labels(class_filename)

dest_size = (224, 224)
img = Image.open(image_filename)
img = img.resize(dest_size)
arr = np.array(img)
img_bytes = arr.tobytes()

server = TFLiteServer(tflite_server_bin, model_filename, port)

context = zmq.Context()
print ("Connecting to tflite server...")
socket = context.socket(zmq.REQ)
socket.connect ("tcp://localhost:%d" % port)

start = time.time()
socket.send(arr)
message = socket.recv()
duration = time.time() - start
obj = json.loads(message.decode('UTF-8'))
if obj['err'] != "none":
    print(obj['err'])
    exit(-1)

pred_arr = np.array(obj['result'][0])
class_pred = np.argmax(pred_arr)
print('prediction:', classes[class_pred], "inference time: %0.2f" % duration)
