from keras.engine import Layer
import tensorflow as tf

class PreprocessLayer(Layer):

    def __init__(self,net_size,**kwargs):
        super(PreprocessLayer,self).__init__(**kwargs)
        self.net_size = net_size

    # this will work only for single images passed to the layer
    def call(self,img):
        ## This is just compatible with tf.__version__ >= 1.13
        # resized = tf.image.resize_images(
        #     img,
        #     (self.net_size,self.net_size),
        #     preserve_aspect_ratio=True,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


        ## For older versions use this
        prev_shape = tf.shape(img)
        max_ = tf.maximum(prev_shape[0],prev_shape[1])
        ratio = tf.cast(max_,tf.float32) / tf.constant(self.net_size,dtype=tf.float32)
        new_width = tf.cast(tf.cast(prev_shape[1],tf.float32) / ratio,tf.int32)
        new_height = tf.cast(tf.cast(prev_shape[0],tf.float32) / ratio,tf.int32)
        resized = tf.image.resize_images(img,(new_height,new_width))


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

        return [expanded,prev_shape]




    def compute_output_shape(self,input_shape):
        return [(1,self.net_size,self.net_size,3),(None,2)]