import tfcoreml as tf_converter
import numpy as np
from tensorflow.python.tools.freeze_graph import freeze_graph
from keras.preprocessing.image import load_img

import tfcoreml
import coremltools
import yaml

f = open("config.yaml", "r+")
cfg = yaml.load(f)
imageSize = cfg['imageSize']
checkpoints = cfg['checkpoints']
chk = cfg['chk']
chkpoint = checkpoints[chk]
versionName = chkpoint.lstrip('mobilenet_')

# Provide these to run freeze_graph:
# Graph definition file, stored as protobuf TEXT
graph_def_file = './models/model.pbtxt'
# Trained model's checkpoint name
checkpoint_file = './checkpoints/model.ckpt'
# Frozen model's output name
frozen_model_file = './models/frozen_model.pb'
# Output nodes. If there're multiple output ops, use comma separated string, e.g. "out1,out2".
output_node_names = 'heatmap,offset_2,displacement_fwd_2,displacement_bwd_2'
# output_node_names = 'Softmax' 

# Call freeze graph
freeze_graph(input_graph=graph_def_file,
             input_saver="",
             input_binary=False,
             input_checkpoint=checkpoint_file,
             output_node_names=output_node_names,
             restore_op_name="save/restore_all",
             filename_tensor_name="save/Const:0",
             output_graph=frozen_model_file,
             clear_devices=True,
             initializer_nodes="")

input_tensor_shapes = {"image:0":[1,imageSize, imageSize, 3]} 
coreml_model_file = './models/model.mlmodel'
# output_tensor_names = ['output:0']
output_tensor_names = ['heatmap:0','offset_2:0','displacement_fwd_2:0','displacement_bwd_2:0']

coreml_model = tfcoreml.convert(
        tf_model_path=frozen_model_file, 
        mlmodel_path=coreml_model_file, 
        input_name_shape_dict=input_tensor_shapes,
        image_input_names=['image:0'],
        output_feature_names=output_tensor_names,
        is_bgr=False,
        red_bias = -1, 
        green_bias = -1, 
        blue_bias = -1, 
        image_scale = 2./255)


coreml_model.author = 'Infocom TPO'
coreml_model.license = 'MIT'
coreml_model.short_description = 'Ver.0.0.1'

coreml_model.save('./models/posenet'+ str(imageSize) + '_' + versionName +'.mlmodel')

img = load_img("./images/tennis_in_crowd.jpg", target_size=(imageSize, imageSize))
print(img)
out = coreml_model.predict({'image__0': img})['heatmap__0']
print("#output coreml result.")

print(out.shape)
print(np.transpose(out))
print(out)
# print(out[:, 0:1, 0:1])
print(np.mean(out))
