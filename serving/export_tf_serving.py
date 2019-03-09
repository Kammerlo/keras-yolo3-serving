import json

from keras.models import load_model
import keras.backend as K
from keras.layers import Input
from keras import Model
from tensorflow.saved_model import builder as saved_model_builder
from tensorflow.saved_model.signature_def_utils import predict_signature_def
from tensorflow.saved_model import tag_constants
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils_impl
import argparse
import os
import cv2

from utils.base64layer import Base64DecodeLayer
from utils.decodeyololayer import DecodeYoloLayer
from utils.letterboxlayer import LetterBoxLayer


def export_h5_to_pb(path_to_h5,model_version, export_path,config_path):
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
    # Set the learning phase to Test since the model is already trained.
    K.set_learning_phase(0)


    input = Input(batch_shape=(1,),dtype=tf.string)
    b64 = Base64DecodeLayer()(input)
    preproc = LetterBoxLayer(net_size=416)(b64)


    keras_model = load_model(path_to_h5)


    x = keras_model(preproc[0])
    # Bypass the input image shape to get the exact bounding box coordinates
    # Keras doesn't like a list in a list as an input, so [x,preproc[1]] doesn't work.
    x.append(preproc[1])
    x = DecodeYoloLayer(anchors=config["model"]["anchors"], classes_num=len(config["model"]["labels"]), net_size=416)(x)
    model = Model(inputs=input,outputs=x)


    tensor_info_input = tf.saved_model.utils.build_tensor_info(model.input)
    tensor_info_output_boxes = tf.saved_model.utils.build_tensor_info(model.output[0])
    tensor_info_output_classes = tf.saved_model.utils.build_tensor_info(model.output[1])
    tensor_info_output_scores = tf.saved_model.utils.build_tensor_info(model.output[2])

    signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={"image": tensor_info_input},
            outputs={"box": tensor_info_output_boxes,
                     "class": tensor_info_output_classes,
                     "confidence": tensor_info_output_scores},
            method_name=signature_constants.PREDICT_METHOD_NAME)
    )

    valid_prediction_signature = tf.saved_model.signature_def_utils.is_valid_signature(signature)
    if(valid_prediction_signature == False):
        raise ValueError("Error: Prediction signature not valid!")

    builder = saved_model_builder.SavedModelBuilder(os.path.join(export_path,str(model_version)))

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(sess=sess,
                                             tags=[tag_constants.SERVING],
                                             signature_def_map={signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
        builder.save()
    print("Done exporting the model")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Export your yolo3 to use it with tensorflow serving.')
    argparser.add_argument('-m', '--model', help='path to keras model.',default="voc.h5")
    argparser.add_argument('-t', '--tensor', help='path to exported tensorflow model',default="models\\voc")
    argparser.add_argument('-c', '--config', help='path to training config file',default="..\\zoo\\config_voc.json")
    argparser.add_argument('-v', '--version', help='Model version',default=1)
    args = argparser.parse_args()

    export_h5_to_pb(args.model,args.version,args.tensor,args.config)