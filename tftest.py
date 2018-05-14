import json
import struct
import tensorflow as tf
import cv2
import numpy as np

f = open("manifest.json")
variables = json.load(f)
f.close()

for x in variables:
    filename = variables[x]["filename"]
    byte = open('./waits/'+filename,'rb').read()
    # print(byte)
    fmt = str (len(byte) / struct.calcsize('f')) + 'f'
    d = struct.unpack(fmt, byte) 
    d = np.array(d,dtype=np.float64)
    d = d.reshape( variables[x]["shape"])
    variables[x]["x"] = d
    tf.Variable(d, name=filename)

def read_imgfile(path, width, height):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(float)
    img = img * (2.0 / 255.0) - 1.0
    return img

def conv(inputs, stride, blockId):
    # w = tf.nn.conv2d(inputs,weights("Conv2d_" + str(blockId)), [1,stride,stride,1], padding='SAME')
    # w = tf.nn.bias_add(w,biases("Conv2d_" + str(blockId)))
    # w = tf.nn.relu6(w)

    return tf.nn.relu6(
        tf.nn.conv2d(inputs,weights("Conv2d_" + str(blockId)), [1,stride,stride,1], padding='SAME') 
        + biases("Conv2d_" + str(blockId)))

def convToOutput(mobileNetOutput, outputLayerName):
    # w = tf.nn.conv2d(mobileNetOutput,weights(outputLayerName), [1,1,1,1], padding='SAME')
    # w = tf.nn.bias_add(w,biases(outputLayerName),name=outputLayerName)

    return tf.nn.bias_add(
        tf.nn.conv2d(mobileNetOutput,weights(outputLayerName), [1,1,1,1], padding='SAME')
        ,biases(outputLayerName),name=outputLayerName)

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
  
    w = tf.nn.depthwise_conv2d(inputs,depthwiseWeights(dwLayer),[1,stride,stride,1], 'SAME',rate=dilations, data_format='NHWC')
    w = tf.nn.bias_add(w,biases(dwLayer))
    w = tf.nn.relu6(w)

    w = tf.nn.conv2d(w,weights(pwLayer), [1,1,1,1], padding='SAME')
    w = tf.nn.bias_add(w,biases(pwLayer))
    w = tf.nn.relu6(w)

    # w = tf.nn.relu6(
    #     tf.nn.depthwise_conv2d_native(inputs,depthwiseWeights(dwLayer),[1,stride,stride,1], 'SAME', data_format='NHWC',dilations=[1,1,1,1])
    #     + biases(dwLayer))
    # w = tf.nn.relu6(
    #     tf.nn.conv2d(w,weights(pwLayer), [1,1,1,1], padding='SAME')
    #     + biases(pwLayer))

    return w

image = read_imgfile("./images/tennis_in_crowd.jpg",None,None)
image = np.array(image,dtype=np.float64)
image = image.reshape(1,513,513,3)

print("image")
print(image.shape)
print(np.mean(image))
print(image[0:1,0:1,:])

# image = np.load("./data/image.npy")
# image = np.array(image,dtype=np.float32)
# image = image.reshape(1,513,513,3)

# print("image")
# print(image.shape)
# print(np.mean(image))
# print(image[0:1,0:1,:])


inputs = tf.placeholder(tf.float64, shape=[None, 513, 513, 3],name='inputs')

# dilations = [1,1,1,1]
dilations = [1,1]
buff = {}
output = conv(inputs,2,0)
buff[0] = output
output = separableConv(output,1,1,dilations)
buff[1] = output
output = separableConv(output,2,2,dilations)
buff[2] = output
output = separableConv(output,1,3,dilations)
buff[3] = output
output = separableConv(output,2,4,dilations)
buff[4] = output
output = separableConv(output,1,5,dilations)
buff[5] = output
output = separableConv(output,2,6,dilations)
buff[6] = output

output = separableConv(output,1,7,dilations)
buff[7] = output
output = separableConv(output,1,8,dilations)
buff[8] = output
output = separableConv(output,1,9,dilations)
buff[9] = output
output = separableConv(output,1,10,dilations)
buff[10] = output
output = separableConv(output,1,11,dilations)
buff[11] = output
output = separableConv(output,1,12,dilations)
buff[12] = output
output = separableConv(output,1,13,[2,2])
buff[13] = output

heatmaps = convToOutput(output,'heatmap_2')
heatmapScores = tf.sigmoid(heatmaps,name="heatmap")
offsets = convToOutput(output,'offset_2')
displacementsFwd = convToOutput(output,'displacement_fwd_2')
displacementsBwd = convToOutput(output,'displacement_bwd_2')

with tf.Session() as sess:  
    sess = tf.Session()

    # for b in buff:
    #     res = sess.run(buff[b], feed_dict={ inputs: image})
        
    #     print(b)
    #     print(res.shape)
    #     print(np.mean(res))
    #     print(res[0:1,0:1,:])
    #     print("=========")

    # heatmapScores,offsets,displacementsFwd,displacementsBwd = sess.run([heatmapScores,offsets,displacementsFwd,displacementsBwd], feed_dict={
    #         inputs: [np.ndarray(shape=(513, 513, 3),dtype=np.float64)]
    #     }
    # )

    heatmapScores,offsets,displacementsFwd,displacementsBwd = sess.run([heatmapScores,offsets,displacementsFwd,displacementsBwd], feed_dict={
            inputs: image
        }
    )

    print(heatmapScores.shape)
    print(np.mean(heatmapScores))
    print(heatmapScores[0:1,0:1,0:1,:])

with open("heatmapScores.txt", "wb") as f:
    byte = np.array(heatmapScores,dtype=np.float32).tobytes()
    f.write(byte)

with open("offsets.txt", "wb") as f:
    byte = np.array(offsets,dtype=np.float32).tobytes()
    f.write(byte)

with open("displacementsFwd.txt", "wb") as f:
    byte = np.array(displacementsFwd,dtype=np.float32).tobytes()
    f.write(byte)

with open("displacementsBwd.txt", "wb") as f:
    byte = np.array(displacementsBwd,dtype=np.float32).tobytes()
    f.write(byte)

print(heatmapScores.shape)
print(offsets.shape)
print(displacementsFwd.shape)
print(displacementsBwd.shape)


