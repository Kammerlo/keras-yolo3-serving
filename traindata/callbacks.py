import io

from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
import tensorflow as tf
import numpy as np

from utils.bbox import draw_boxes
from utils.utils import get_yolo_boxes


def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)

class TensorBoardImage(Callback):
    def __init__(self, tag,labels,infer_model=None,valid_generator=None,anchors=None,log_dir= './logs',img_count=5):
        super().__init__()
        self.tag = tag
        self.obj_thresh, self.nms_thresh = 0.8, 0.6
        self.net_h, self.net_w = 416, 416
        self.valid_generator = valid_generator
        self.infer_model = infer_model
        self.anchors = anchors
        self.log_dir = log_dir
        self.labels = labels
        if len(self.valid_generator.instances) < img_count:
            self.img_count = len(self.valid_generator.instances)
        else:
            self.img_count = img_count

    def on_epoch_end(self, epoch, logs={}):
        writer = tf.summary.FileWriter(self.log_dir)
        for i in range(self.img_count):
            # Load image
            img = self.valid_generator.load_image(i)
            try:
                boxes = get_yolo_boxes(self.infer_model,[img],self.net_h,self.net_w,self.anchors,self.obj_thresh,self.nms_thresh)
                draw_boxes(img, boxes[0], self.labels, self.obj_thresh)
            except ZeroDivisionError:
                print("Zero")

            image = make_image(img)
            summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag + "_" + str(i), image=image)])

            writer.add_summary(summary, epoch)
        writer.close()

        return


class CustomTensorBoard(TensorBoard):
    """ to log the loss after each batch
    """    
    def __init__(self, log_every=1, **kwargs):
        super(CustomTensorBoard, self).__init__(**kwargs)
        self.log_every = log_every
        self.counter = 0
    
    def on_batch_end(self, batch, logs=None):
        self.counter+=1
        if self.counter%self.log_every==0:
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.counter)
            self.writer.flush()
        
        super(CustomTensorBoard, self).on_batch_end(batch, logs)

class CustomModelCheckpoint(ModelCheckpoint):
    """ to save the template model, not the multi-GPU model
    """
    def __init__(self, model_to_save, **kwargs):
        super(CustomModelCheckpoint, self).__init__(**kwargs)
        self.model_to_save = model_to_save

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model_to_save.save_weights(filepath, overwrite=True)
                        else:
                            self.model_to_save.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model_to_save.save_weights(filepath, overwrite=True)
                else:
                    self.model_to_save.save(filepath, overwrite=True)

        super(CustomModelCheckpoint, self).on_batch_end(epoch, logs)