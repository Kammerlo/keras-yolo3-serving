from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras import Input
from tensorflow.python.keras import Model
from tensorflow.saved_model import builder as saved_model_builder
from tensorflow.saved_model.signature_def_utils import predict_signature_def
from tensorflow.saved_model import tag_constants
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils_impl
import argparse

from utils.custom_layer import PreprocessLayer


def export_h5_to_pb(path_to_h5,model_version, export_path):
    # Set the learning phase to Test since the model is already trained.
    K.set_learning_phase(0)
    input = Input(shape=(None,3))
    preproc = PreprocessLayer(net_size=416)(input)
    keras_model = load_model(path_to_h5)(preproc)
    model = Model(input,keras_model)
    signature = predict_signature_def(inputs={"image": model.input},
                                      outputs={"yolo1": model.output[0],
                                               "yolo2": model.output[1],
                                               "yolo3": model.output[2]})

    valid_prediction_signature = tf.saved_model.signature_def_utils.is_valid_signature(signature)
    if(valid_prediction_signature == False):
        raise ValueError("Error: Prediction signature not valid!")

    builder = saved_model_builder.SavedModelBuilder(export_path + "/" + str(model_version))

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[tag_constants.SERVING],
                                         signature_def_map={'predict': signature})
        builder.save()
    print("Done exporting the model")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Export your yolo3 to use it with tensorflow serving.')
    argparser.add_argument('-m', '--model', help='path to keras model.',default="voc.h5")
    argparser.add_argument('-t', '--tensor', help='path to exported tensorflow model',default="models/voc")
    argparser.add_argument('-v', '--version', help='Model version',default=1)
    args = argparser.parse_args()

    export_h5_to_pb(args.model,args.version,args.tensor)
