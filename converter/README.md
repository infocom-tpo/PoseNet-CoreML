# PoseNet(TensorFlowJS) Wait Converter

From PoseNet(TensorFlowJS) to CoreML-Model

## Get PoseNet Wait 
```
$ python3 wget.py
```

## PoseNet TensorFlowJS to Python
```
$ python tfjs2python.py
```

## TensorFlow(Python) to Coreml
```
$ python convert.py
```

## Dependencies
```
$ pip install tfcoreml
$ pip install coremltools
$ pip install tensorflow==1.6.0
```

## Create Graph

Get: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/import_pb_to_tensorboard.py

```
$ cd converter
$ mkdir graph
$ python import_pb_to_tensorboard.py --model_dir=./models/frozen_model.pb --log_dir=./graph
$ tensorboard --logdir=./graph

// go : http://localhost:6006/
```
