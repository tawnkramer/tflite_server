import zmq
import numpy as np
import time
import json

port = 5555
context = zmq.Context()
print ("Connecting to server...")
socket = context.socket(zmq.REQ)
socket.connect ("tcp://localhost:%d" % port)


def make_empy_arr(size_bytes):
  elements = []
  for i in range(size_bytes):
      elements.append(0)

  arr = bytes(elements)
  return arr

size_bytes = 150528
arr = make_empy_arr(size_bytes)

socket.send(arr)
message = socket.recv()
obj = json.loads(message.decode('UTF-8'))
if obj['err'] != "none":
    print(obj['err'])
    err = obj['err'].split(' ')
    if err[2] == 'expected':
      size_bytes = int(err[3])
      arr = make_empy_arr(size_bytes)

i = 0
start = time.time()
numIter = 50
while i < numIter:
  i += 1
  socket.send(arr)
  message = socket.recv()
  obj = json.loads(message.decode('UTF-8'))
  if obj['err'] != "none":
      print(obj['err'])

dur = time.time() - start
print('%d predictions took %.2f seconds.' % (numIter, dur))
print("Inference at %.2f FPS." % (float(numIter) / dur))

