import base64
import time
import cv2
import numpy as np
import grpc
from tensorflow.contrib.util import make_ndarray,make_tensor_proto
from tensorflow_serving.apis import prediction_service_pb2_grpc, predict_pb2
import tensorflow as tf

from serving.http_client import draw_boxes

'''
This script should help you to test the maximum performance of your tf serving instance and model.
'''


tf.app.flags.DEFINE_integer('num_tests', 100, 'Number of calls to the grpc server as a test')
tf.app.flags.DEFINE_string('server', '127.0.0.1:8500', 'PredictionService host:port')
tf.app.flags.DEFINE_string('img', 'dog.jpg', 'Inferencing test img.')
tf.app.flags.DEFINE_string('model','voc','Model to use.')
tf.app.flags.DEFINE_boolean('vis',True,'Show Image prediction.')
FLAGS = tf.app.flags.FLAGS

def predictResponse_into_nparray(response, output_tensor_name):
    dims = response.outputs[output_tensor_name].tensor_shape.dim
    shape = tuple(d.size for d in dims)
    print(shape)
    return np.reshape(response.outputs[output_tensor_name].float_val, shape)


def main(_):
    channel = grpc.insecure_channel(FLAGS.server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = FLAGS.model
    request.model_spec.signature_name = "serving_default"

    img = cv2.imread(FLAGS.img)
    retr, buffer = cv2.imencode(".jpg", img)
    jpg_str = base64.urlsafe_b64encode(buffer).decode("utf-8")
    difference_sum = 0
    for i in range(FLAGS.num_tests):
        request.inputs['image'].CopyFrom(make_tensor_proto(jpg_str,shape=[1,]))
        start = time.time()
        response = stub.Predict(request,10)
        diff = time.time() - start
        print("Predict num: {} consumed time: {}".format(i,diff))
        difference_sum += diff
        if FLAGS.vis:
            classes = np.squeeze(make_ndarray(response.outputs["class"]))
            box = np.squeeze(make_ndarray(response.outputs["box"]))
            confidence = np.squeeze(make_ndarray(response.outputs["confidence"]))
            visimg = cv2.imread(FLAGS.img)
            for o in range(len(classes)):
                draw_boxes(visimg,box[o],confidence[o],classes[o])
            cv2.imshow("Vis",visimg)
            cv2.waitKey(1)
    print("Average time: {}".format(difference_sum / FLAGS.num_tests))

if __name__ == '__main__':
    print("Starting inferencing")
    tf.app.run()