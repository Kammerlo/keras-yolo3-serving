from keras.engine import Layer
import tensorflow as tf
import numpy as np

class PostprocessLayer(Layer):

    # Currently just with a fixed image size
    def __init__(self,anchors,classes_num,net_size, obj_threshold = 0.7, max_boxes = 20,  nms_threshold = 0.45,**kwargs):
        super(PostprocessLayer,self).__init__(**kwargs)
        self.net_size = net_size
        self.classes_num = classes_num
        self.obj_threshold = obj_threshold
        self.max_boxes = max_boxes
        self.anchors =  anchors
        self.anchors = np.array(self.anchors).reshape(-1, 2)
        self.anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.nms_threshold = nms_threshold

    def build(self, input_shape):
        assert isinstance(input_shape,list)
        super(PostprocessLayer,self).build(input_shape)

    def _get_feats(self,feats, anchors, num_classes, input_shape):

        num_anchors = 3
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])
        grid_size = tf.shape(feats)[1:3]
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])

        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis = -1)
        grid = tf.cast(grid, tf.float32)

        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)

        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        box_class_probs = tf.sigmoid(predictions[..., 5:])
        return box_xy, box_wh, box_confidence, box_class_probs

    def correct_boxes(self,box_xy, box_wh, input_shape, image_shape):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = tf.cast(input_shape, dtype = tf.float32)
        image_shape = tf.cast(image_shape, dtype = tf.float32)
        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = tf.concat([
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2]
        ], axis = -1)
        boxes *= tf.concat([image_shape, image_shape], axis = -1)
        return boxes

    def boxes_and_scores(self,feats, anchors, classes_num, input_shape, image_shape):

        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats, anchors, classes_num, input_shape)
        boxes = self.correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = tf.reshape(boxes, [-1, 4])
        box_scores = box_confidence * box_class_probs
        box_scores = tf.reshape(box_scores, [-1, classes_num])
        return boxes, box_scores

    # this will work only for single images passed to the layer
    def call(self,x):
        assert isinstance(x, list)
        shape_tensor = x[3]
        shape_tensor = tf.Print(shape_tensor,[shape_tensor],"img_shape")
        self.image_shape = (shape_tensor[0],shape_tensor[1])
        yolo_outputs = x[:3]
        boxes = []
        box_scores = []
        input_shape = tf.shape(yolo_outputs[0])[1 : 3] * 32
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        for i in range(len(yolo_outputs)):
            out = yolo_outputs[i]
            anch = self.anchors[anchor_mask[i]]
            classes = self.classes_num
            img_shape = self.image_shape

            _boxes, _box_scores = self.boxes_and_scores(out,anch , classes, input_shape, self.image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = tf.concat(boxes, axis = 0)
        box_scores = tf.concat(box_scores, axis = 0)

        mask = box_scores >= self.obj_threshold
        max_boxes_tensor = tf.constant(self.max_boxes, dtype = tf.int32)
        boxes_ = []
        scores_ = []
        classes_ = []
        for c in range(self.classes_num):
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold = self.nms_threshold)
            class_boxes = tf.gather(class_boxes, nms_index)
            class_box_scores = tf.gather(class_box_scores, nms_index)
            classes = tf.ones_like(class_box_scores, 'float32') * c
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)

        boxes_ = tf.concat(boxes_, axis = 0)
        scores_ = tf.concat(scores_, axis = 0)
        classes_ = tf.concat(classes_, axis = 0)

        # boxes_ = tf.Print(boxes_,[tf.shape(boxes_), tf.shape(scores_), tf.shape(classes_)],"boxes_ shape:")

        return [boxes_,classes_,scores_]

    def compute_output_shape(self,input_shape):
        return [(None, 3),(None, 1),(None, 1)]