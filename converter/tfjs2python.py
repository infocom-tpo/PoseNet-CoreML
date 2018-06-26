import json
import struct
import tensorflow as tf
import cv2
import numpy as np
import os
import yaml
import sys

f = open("config.yaml", "r+")
cfg = yaml.load(f)
checkpoints = cfg['checkpoints']
imageSize = cfg['imageSize']
chk = cfg['chk']
outputStride = cfg['outputStride']
chkpoint = checkpoints[chk]

if chkpoint == 'mobilenet_v1_050':
    mobileNetArchitectures = cfg['mobileNet50Architecture']
elif chkpoint == 'mobilenet_v1_075':
    mobileNetArchitectures = cfg['mobileNet75Architecture']
else:
    mobileNetArchitectures = cfg['mobileNet100Architecture']

width = imageSize
height = imageSize

def toOutputStridedLayers(convolutionDefinition, outputStride):
    currentStride = 1
    rate = 1
    blockId = 0
    buff = []
    for _a in convolutionDefinition:
        convType = _a[0]
        stride = _a[1]
        
        if (currentStride == outputStride):
            layerStride = 1
            layerRate = rate
            rate *= stride
        else:
            layerStride = stride
            layerRate = 1
            currentStride *= stride
        
        buff.append({
            'blockId': blockId,
            'convType': convType,
            'stride': layerStride,
            'rate': layerRate,
            'outputStride': currentStride
        })
        blockId += 1

    return buff

layers = toOutputStridedLayers(mobileNetArchitectures, outputStride)

f = open(os.path.join('./waits/', chkpoint, "manifest.json"))
variables = json.load(f)
f.close()

# with tf.variable_scope(None, 'MobilenetV1'):
for x in variables:
    filename = variables[x]["filename"]
    byte = open( os.path.join('./waits/', chkpoint, filename),'rb').read()
    fmt = str (int (len(byte) / struct.calcsize('f'))) + 'f'
    d = struct.unpack(fmt, byte) 
    # d = np.array(d,dtype=np.float32)
    d = tf.cast(d, tf.float32)
    d = tf.reshape(d,variables[x]["shape"])
    variables[x]["x"] = tf.Variable(d,name=x)

def read_imgfile(path, width, height):
    img = cv2.imread(path)
    img = cv2.resize(img, (width,height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(float)
    img = img * (2.0 / 255.0) - 1.0
    return img

def convToOutput(mobileNetOutput, outputLayerName):
    w = tf.nn.conv2d(mobileNetOutput,weights(outputLayerName),[1,1,1,1],padding='SAME')
    w = tf.nn.bias_add(w,biases(outputLayerName), name=outputLayerName)
    return w

def conv(inputs, stride, blockId):
    # w = tf.nn.conv2d(inputs,weights("Conv2d_" + str(blockId)), stride, padding='SAME')
    # w = tf.nn.bias_add(w,biases("Conv2d_" + str(blockId)))
    # w = tf.nn.relu6(w)
    # return w
    return tf.nn.relu6(
        tf.nn.conv2d(inputs,weights("Conv2d_" + str(blockId)), stride, padding='SAME') 
        + biases("Conv2d_" + str(blockId)))

def weights(layerName):
    return variables["MobilenetV1/" + layerName + "/weights"]['x']

def biases(layerName):
    return variables["MobilenetV1/" + layerName + "/biases"]['x']

def depthwiseWeights(layerName):
    return variables["MobilenetV1/" + layerName + "/depthwise_weights"]['x']

def separableConv(inputs, stride, blockID, dilations):
    if (dilations == None):
        dilations = [1,1]
    
    dwLayer = "Conv2d_" + str(blockID) + "_depthwise"
    pwLayer = "Conv2d_" + str(blockID) + "_pointwise"
    
    w = tf.nn.depthwise_conv2d(inputs,depthwiseWeights(dwLayer),stride, 'SAME',rate=dilations, data_format='NHWC')
    w = tf.nn.bias_add(w,biases(dwLayer))
    w = tf.nn.relu6(w)

    w = tf.nn.conv2d(w,weights(pwLayer), [1,1,1,1], padding='SAME')
    w = tf.nn.bias_add(w,biases(pwLayer))
    w = tf.nn.relu6(w)

    return w


image = tf.placeholder(tf.float32, shape=[1, imageSize, imageSize, 3],name='image')

x = image
rate = [1,1]
buff = []
# conv_res = {}
with tf.variable_scope(None, 'MobilenetV1'):
    
    for m in layers:
        strinde = [1,m['stride'],m['stride'],1]
        rate = [m['rate'],m['rate']]
        if (m['convType'] == "conv2d"):
            x = conv(x,strinde,m['blockId'])
            buff.append(x)
        elif (m['convType'] == "separableConv"):
            x = separableConv(x,strinde,m['blockId'],rate)
            buff.append(x)

# x = tf.identity(x, name="output")

heatmaps = convToOutput(x, 'heatmap_2')
offsets = convToOutput(x, 'offset_2')
displacementFwd = convToOutput(x, 'displacement_fwd_2')
displacementBwd = convToOutput(x, 'displacement_bwd_2')
heatmaps = tf.sigmoid(heatmaps,'heatmap')

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()

    ans = sess.run([heatmaps,offsets,displacementFwd,displacementBwd], feed_dict={
            image: [np.ndarray(shape=(width, height, 3),dtype=np.float32)]
        }
    )

    save_dir = './checkpoints'
    save_path = os.path.join(save_dir, 'model.ckpt')
    # if not os.path.exists(save_dir):
        # os.mkdir(save_dir)
    save_path = saver.save(sess, save_path)

    tf.train.write_graph(sess.graph,"./models/","model.pbtxt")

    # Result
    input_image = read_imgfile("./images/tennis_in_crowd.jpg",width,height)
    input_image = np.array(input_image,dtype=np.float32)
    input_image = input_image.reshape(1,width,height,3)
    mobileNetOutput = sess.run(x, feed_dict={ image: input_image } )

    heatmaps_result,offsets_result,displacementFwd_result,displacementBwd_result = sess.run(
        [heatmaps,offsets,displacementFwd,displacementBwd], feed_dict={ image: input_image } )

    print(input_image)
    print(input_image.shape)
    print(np.mean(input_image))

    count = 0
    for b in buff:
        conv_result = sess.run(b, feed_dict={ image: input_image } )
        print("========")
        print(count)
        print(conv_result[0:1, 0:1, :])
        print(conv_result.shape)
        print(np.mean(conv_result))
        count += 1


    print("========")
    print("mobileNetOutput")
    print(mobileNetOutput[0:1, 0:1, :])
    print(mobileNetOutput.shape)
    print(np.mean(mobileNetOutput))
    
    heatmaps_result = heatmaps_result[0]

    print("========")
    print("heatmaps")
    print(heatmaps_result[0:1, 0:1, :])
    print(heatmaps_result.shape)
    print(np.mean(heatmaps_result))
    
