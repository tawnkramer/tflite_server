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

#from https://git.codingcafe.org/Mirrors/tensorflow/tensorflow/blob/a46412b9384b82593f69df04df49bfbe2c7f245c/tensorflow/lite/g3doc/models.md
model_filename = "mobilenet_v2_1.0_224_quant.tflite"
model_url = "http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz"
class_filename = "labels.txt"
image_filename = "hammerhead.jpg"
tflite_server_bin = '~/bin/tflite_serve'

port = 5555

def download_model_if_needed():
    if not os.path.exists(model_filename):
        os.system('wget %s' % model_url)
        os.system('tar xzf %s' % model_filename.replace('.tflite', '.tgz'))

def img_to_binary(img, format='jpeg'):
    f = BytesIO()
    img.save(f, format=format)
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

if not os.path.exists(model_filename):
    print(model_filename, " wasn't available.")
    exit(-1)

classes = load_labels(class_filename)

dest_size = (224, 224)
img = Image.open(image_filename)
img = img.resize(dest_size)
arr = np.array(img)
img_bytes = arr.tobytes()

if not os.path.exists(tflite_server_bin):
    print('tflite server not found at', tflite_server_bin)
    exit(-1)
    
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
print('prediction:', classes[class_pred], "inference time: %0.2f seconds" % duration)
