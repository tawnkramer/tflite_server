# tflite_server #

A lightweight executable that will serve TFlite model inferencing over zmq socket

### Goals ###

Create a small application that is quick to start up, and serves inferences needs of other processes that need low latency response. This starts up much faster than an equivalent python code and can be built easier on low cpu devices like the raspberry pi zero.

### Setup ###

#### clone and build tensorflow tflite library ####

from: https://www.tensorflow.org/lite/guide/build_rpi
```
git clone https://github.com/tensorflow/tensorflow.git --depth=1
cd tensorflow
./tensorflow/lite/tools/make/download_dependencies.sh
./tensorflow/lite/tools/make/build_rpi_lib.sh
sudo cp tensorflow/lite/tools/make/gen/rpi_armv7l/lib/libtensorflow-lite.a /usr/local/lib
```

*Note: currently there is a bug with the makefile that might be fixed. Check that tensorflow/lite/tools/make/Makefile has BUILD_WITH_NNAPI=false

#### install dependencies ####

```
sudo apt-get install build-essential cmake libczmq-dev
```

#### clone and build tflite_server ####

```
git clone https://github.com/tawnkramer/tflite_server
mkdir tflite_server/build
cd tflite_server/build
cmake ..
make
```
You should see the exeutable tflite_server in the tflite_server/build dir.

#### test ####

Download a model from https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md
```
cd tflite_server/build
wget http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz
tar xzf mobilenet_v1_1.0_224_quant.tgz
./tflite_server --model mobilenet_v1_1.0_224_quant.tflite
```

Then in another shell, try the tester.

```
cd tflight_server/tests
python test.py
```
