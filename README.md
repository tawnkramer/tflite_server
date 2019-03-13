# tflite_server #

A lightweight executable that will serve TFlite model inferencing over zmq socket. 

### Goals ###

Create a small application that is quick to start up, and serves inferences needs of other processes that need low latency response. This starts up much faster than an equivalent python code and can be built easier on low cpu devices like the raspberry pi zero.

ZMQ is easy to use in a variety of languages, esp python, and makes for easy integration. No need to install tensorflow on the destination machine. This is designed primarily for RaspberryPi B/B+/A+/Zero.

With python you can launch this executable as a subprocess and open a socket to use it. Even serve a model from one machine to another.

*This is NOT intended as a production server.* 

### Setup ###

#### install dependencies ####

```
sudo apt-get install build-essential cmake libczmq-dev
```

#### clone and build tensorflow tflite library ####

These are the steps to build on the PI3 B.

from: https://www.tensorflow.org/lite/guide/build_rpi
```
cd ~/
git clone https://github.com/tensorflow/tensorflow.git --depth=1
```

Note* 3/12/2019 The latest tflite code was failing for me. If you have troubles, you can try:
```
cd ~/tensorflow
git pull --depth==100
git checkout 7273a08672c29739cee9f9aa91fb4d92ec1e2682
```

continue build ...
```
cd ~/tensorflow
./tensorflow/lite/tools/make/download_dependencies.sh
```

*Note: Currently there is a bug with the makefile that might be fixed. Also, enable openmp for better performance. Check :
    tensorflow/lite/tools/make/Makefile 
    * has BUILD_WITH_NNAPI=false
    * has CXXFLAGS := -O3 -DNDEBUG -fPIC -fopenmp


```
cd ~/tensorflow
./tensorflow/lite/tools/make/build_rpi_lib.sh
sudo cp tensorflow/lite/tools/make/gen/rpi_armv7l/lib/libtensorflow-lite.a /usr/local/lib
```


#### clone and build tflite_server ####

*Note: this wants to live a dir next to tensorflow. Or modify CMakeLists.txt for your Tensorflow location.
```
cd ~/
git clone https://github.com/tawnkramer/tflite_server
mkdir tflite_server/build
cd tflite_server/build
cmake ..
make
```
You should see the exeutable tflite_server in the tflite_server/build dir.

### Speed Test ###

Download a model https://www.tensorflow.org/lite/guide/hosted_models
```
cd ~/tflite_server/build
wget http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz
tar xzf mobilenet_v1_1.0_224_quant.tgz
export OMP_NUM_THREADS=4
./tflite_serve --model mobilenet_v1_1.0_224_quant.tflite --num_threads 4
```

Then in another shell, try the tester.

```
pip3 install zmq
cd ~/tflight_server/tests
python3 speed_test.py
```

Try setting --num_threads 1 and compare. Watch htop to see the processor use. On the Pi3 B, 4 threads were about 250% faster than 1 and pegged all four cores.

### Inference Test ###

```
cd tflite_serve/tests
python3 mobilenet_test.py
```

should output : "prediction: hammerhead, hammerhead shark inference time: 3.51 seconds"

### Message Protocol ###

This server accepts a ZQM.REQ type connection. It accepts a message and passes entire binary contents to the tflite model inference. The size of the binary payload must exactly match the size specified in the input tensors of the model. The server then sends a JSON reply with the results. If there was a problem, the json err member will contain details.