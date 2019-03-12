import zmq
import numpy as np
import time

port = 5555
context = zmq.Context()
print ("Connecting to server...")
socket = context.socket(zmq.REQ)
socket.connect ("tcp://localhost:%s" % port)

size_bytes = 49152
elements = []
for i in range(size_bytes):
    elements.append(0)

arr = bytes(elements)

i = 0
start = time.time()
numIter = 50.0
while i < numIter:
  i += 1
  socket.send(arr)
  message = socket.recv()
  print(message)

dur = time.time() - start
print("inference at", numIter / dur, "FPS")

