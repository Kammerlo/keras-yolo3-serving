from keras.engine import Layer
import tensorflow as tf

'''
Decode a base64 image to an image tensor.
'''
class Base64DecodeLayer(Layer):

    def __init__(self,**kwargs):
        super(Base64DecodeLayer,self).__init__(**kwargs)

    # this will work only for single images passed to the layer
    def call(self,x):
        tf_b64 = tf.decode_base64(x[0])
        img = tf.image.decode_jpeg(tf_b64,channels=3)
        img = tf.cast(img,dtype=tf.float32)
        tf.Print(img,[tf.shape(img)],"Img Shape: ")

        return img




    def compute_output_shape(self,input_shape):
        return (1,None,None,3)