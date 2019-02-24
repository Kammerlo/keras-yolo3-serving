from keras.engine import Layer
import tensorflow as tf

class PreprocessLayer(Layer):

    def __init__(self,net_size,**kwargs):
        super(PreprocessLayer,self).__init__(**kwargs)
        self.net_size = net_size

    # this will work only for single images passed to the layer
    def call(self,x):
        # x = x[0]
        tf_b64 = tf.decode_base64(x[0])
        img = tf.image.decode_jpeg(tf_b64,channels=3)
        img = tf.cast(img,dtype=tf.float32)
        resized = tf.image.resize_images(
            img,
            (self.net_size,self.net_size),
            preserve_aspect_ratio=True,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        img_shape = tf.shape(resized)
        height_offset = (self.net_size - img_shape[0]) / 2
        width_offset = (self.net_size - img_shape[1]) / 2
        padded = tf.image.pad_to_bounding_box(
            image=resized,
            offset_height=tf.cast(height_offset,dtype=tf.int32),
            offset_width=tf.cast(width_offset,dtype=tf.int32),
            target_height=self.net_size,
            target_width=self.net_size) / 255.0
        expanded = tf.expand_dims(padded,0)

        return expanded

    def compute_output_shape(self,input_shape):
        return (1,self.net_size,self.net_size,3)