# tflite_server #

A lightweight executable that will serve TFlite model inferencing over zmq socket

### Goals ###

Create a small application that is quick to start up, and serves inferences needs of other processes that need low latency response. This starts up much faster than an equivalent python code and can be built easier on low cpu devices like the raspberry pi zero.

### Setup ###

#### clone and build tensorflow tflite library ####

from: https://www.tensorflow.org/lite/guide/build_rpi
```
cd ~/
git clone https://github.com/tensorflow/tensorflow.git --depth=1
cd tensorflow
./tensorflow/lite/tools/make/download_dependencies.sh
```

*Note: Currently there is a bug with the makefile that might be fixed. Also, enamble openmp for better performance. Check :
    tensorflow/lite/tools/make/Makefile 
    * has BUILD_WITH_NNAPI=false
    * has CXXFLAGS := -O3 -DNDEBUG -fPIC -fopenmp


```
./tensorflow/lite/tools/make/build_rpi_lib.sh
sudo cp tensorflow/lite/tools/make/gen/rpi_armv7l/lib/libtensorflow-lite.a /usr/local/lib
```

Note* 3/12/2019 The latest tflite code was failing for me. If you have troubles, you can try:
```
git pull --depth==100
git checkout 7273a08672c29739cee9f9aa91fb4d92ec1e2682
```
and build again.

#### install dependencies ####

```
sudo apt-get install build-essential cmake libczmq-dev
```

#### clone and build tflite_server ####

*Note: this wants to live a dir next to tensorflow
```
cd ~/
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
export OMP_NUM_THREADS=4
./tflite_serve --model mobilenet_v1_1.0_224_quant.tflite --num_threads 4
```

Then in another shell, try the tester.

```
cd tflight_server/tests
python test.py
```

Try setting --num_threads 1 and compare. Watch htop to see the processor use. On the Pi3 B, it was about 250% faster and pegged all four cores.
