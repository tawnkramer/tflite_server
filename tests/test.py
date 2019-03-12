import zmq

port = 5555
context = zmq.Context()
print ("Connecting to server...")
socket = context.socket(zmq.REQ)
socket.connect ("tcp://localhost:%s" % port)

i = 0
while i < 5:
  i += 1
  socket.send_string("Hi")
  message = socket.recv()
  print(message)

