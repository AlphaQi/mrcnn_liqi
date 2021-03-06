"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2018 Netease,Inc.
Licensed under the MIT License (see LICENSE for details)
Written by LiQi
"""

import os
import sys
import glob
import random
import math
import datetime
import json
import re
import logging
import numpy as np
import scipy.misc
import tensorflow as tf
import pickle as pkl
import setool
import config
import utils
import argparse
import coco_data_input
import visualize as vis
import cv2
sys.path.append('./coco/PythonAPI')

############################################################
#  Resnet Graph
############################################################

def identity_block(input_tensor, kernel_size, filters, stage, block, use_bias=True,  data_dict=None):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = setool.conv_op(input_op=input_tensor, name=conv_name_base + '2a',\
                       kh=1, kw=1,  n_out=nb_filter1,data_dict=data_dict)
    x = setool.batch_norm(x=x, conv=True, phase_train=False, name=bn_name_base + '2a',data_dict=data_dict)
    x = tf.nn.relu(x)

    x = setool.conv_op(input_op=x, name=conv_name_base + '2b',\
                       kh=kernel_size, kw=kernel_size,  n_out=nb_filter2,data_dict=data_dict)
    x = setool.batch_norm(x=x, conv=True, phase_train=False, name=bn_name_base + '2b',data_dict=data_dict)
    x = tf.nn.relu(x)

    x = setool.conv_op(input_op=x, name=conv_name_base + '2c',\
                       kh=1, kw=1,  n_out=nb_filter3,data_dict=data_dict)
    x = setool.batch_norm(x=x, conv=True, phase_train=False, name=bn_name_base + '2c',data_dict=data_dict)

    x = x+input_tensor
    x = tf.nn.relu(x, name='res' + str(stage) + block + '_out')
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block,use_bias=True,strides=2, data_dict=None):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = setool.conv_op(input_op=input_tensor, name=conv_name_base + '2a',\
                       kh=1, kw=1,  dh=strides, dw=strides, n_out=nb_filter1,data_dict=data_dict)
    x = setool.batch_norm(x=x, conv=True, phase_train=False, name=bn_name_base + '2a',data_dict=data_dict)
    x = tf.nn.relu(x)

    x = setool.conv_op(input_op=x, name=conv_name_base + '2b',\
                       kh=kernel_size, kw=kernel_size,  n_out=nb_filter2,data_dict=data_dict)
    x = setool.batch_norm(x=x, conv=True, phase_train=False, name=bn_name_base + '2b',data_dict=data_dict)
    x = tf.nn.relu(x)

    x = setool.conv_op(input_op=x, name=conv_name_base + '2c',\
                       kh=1, kw=1,  n_out=nb_filter3,data_dict=data_dict)
    x = setool.batch_norm(x=x, conv=True, phase_train=False, name=bn_name_base + '2c',data_dict=data_dict)

    shortcut = setool.conv_op(input_op=input_tensor, name=conv_name_base + '1',\
                       kh=1, kw=1,  dh=strides, dw =strides,
                       n_out=nb_filter3,data_dict=data_dict)
    shortcut = setool.batch_norm(x=shortcut, phase_train=False, \
                       name=bn_name_base + '1',data_dict=data_dict)
    x = x+shortcut
    x = tf.nn.relu(x, name='res' + str(stage) + block + '_out')
    return x

def resnet_graph(input_image, architecture, stage5=False, data_dict=None):
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = tf.pad(input_image, [[0, 0],[3,3],[3,3],[0,0]])
    x = setool.conv_op(input_op=x, name='conv1',\
                       kh=7, kw=7,  dh=2, dw=2, n_out=64,data_dict=data_dict, padding="VALID")
    x = setool.batch_norm(x=x, conv=True, phase_train=False, name='conv1',data_dict=data_dict)
    x = tf.nn.relu(x)

    x = setool.mpool_op(input_tensor=x, k=3, s=2)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=1, data_dict=data_dict)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', data_dict=data_dict)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', data_dict=data_dict)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', data_dict=data_dict)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', data_dict=data_dict)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', data_dict=data_dict)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', data_dict=data_dict)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', data_dict=data_dict)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98+i), data_dict=data_dict)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', data_dict=data_dict)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', data_dict=data_dict)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', data_dict=data_dict)
    else:
        C5 = None
    return C2, C3, C4, C5

############################################################
#  Proposal Layer
############################################################

def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w, base anchors
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result

def clip_boxes_graph(boxes, window):
    """
    boxes: [N, 4] each row is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    # Split corners
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    wy1 = tf.cast(wy1, tf.float32)
    wx1 = tf.cast(wx1, tf.float32)
    wy2 = tf.cast(wy2, tf.float32)
    wx2 = tf.cast(wx2, tf.float32)

    y1 = tf.cast(y1, tf.float32)
    x1 = tf.cast(x1, tf.float32)
    y2 = tf.cast(y2, tf.float32)
    x2 = tf.cast(x2, tf.float32)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped

class ProposalLayer():
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.

    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, proposal_count, nms_threshold, anchors, config=None):
        """
        anchors: [N, (y1, x1, y2, x2)] anchors defined in image coordinates
        """
        self.config = config
        self.proposal_count = proposal_count # 2000 if train
        self.nms_threshold = nms_threshold  # 0.7
        self.anchors = anchors.astype(np.float32)

    def getproposal(self, inputs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # Base anchors
        anchors = self.anchors

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = min(6000, self.anchors.shape[0])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="top_anchors").indices
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        anchors = utils.batch_slice(ix, lambda x: tf.gather(anchors, x),
                                    self.config.IMAGES_PER_GPU,
                                    names=["pre_nms_anchors"])

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = utils.batch_slice([anchors, deltas],
                                  lambda x, y: apply_box_deltas_graph(x, y),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors"])

        # Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
        height, width = self.config.IMAGE_SHAPE[:2]
        window = np.array([0, 0, height, width]).astype(np.float32)
        # the input boxes shape id (N, 4)
        boxes = utils.batch_slice(boxes,
                                  lambda x: clip_boxes_graph(x, window),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors_clipped"])

        # Normalize dimensions to range of 0 to 1.
        normalized_boxes = boxes / np.array([[height, width, height, width]])

        # Non-max suppression
        def nms(normalized_boxes, scores):
            indices = tf.image.non_max_suppression(
                normalized_boxes, scores, self.proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(normalized_boxes, indices)
            # Pad if neededs
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)

            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals
        proposals = utils.batch_slice([normalized_boxes, scores], nms,
                                      self.config.IMAGES_PER_GPU)
        return proposals

############################################################
#  Loss Functions
############################################################

def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typicallly: [N, 4], but could be any shape.
    """
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss

def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)

def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """
    # Squeeze last dim to simplify, like flatten
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = tf.cast(tf.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(tf.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Crossentropy loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=anchor_class,
                                                        logits=rpn_class_logits)
    result = tf.cond(tf.size(loss) > 0, lambda: tf.reduce_mean(loss), lambda: tf.constant(0.0))  
    return loss

def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.
    #target_bbox shiji 256*4, zheng yangben, rpn_match shiji fenlei label
    # rpn_bbox, predict 261888*4
    config: the model config object.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = tf.squeeze(rpn_match, -1)
    indices = tf.where(tf.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = tf.reduce_sum(tf.cast(tf.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts,
                                   config.IMAGES_PER_GPU)

    target_bbox = tf.cast(target_bbox, "float32") 
    rpn_bbox = tf.cast(rpn_bbox, "float32") 

    diff = tf.abs(target_bbox - rpn_bbox)
    less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)

    result = tf.cond(tf.size(loss) > 0, lambda: tf.reduce_mean(loss), lambda: tf.constant(0.0))  
    return loss

def mrcnn_class_loss_graph(target_class_ids, pred_class_logits, active_class_ids):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
    """
    target_class_ids = tf.cast(target_class_ids, 'int64')

    # Find predictions of classes that are not in the dataset.
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
    loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss

def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = tf.reshape(target_class_ids, (-1,))
    target_bbox = tf.reshape(target_bbox, (-1, 4))
    pred_bbox = tf.reshape(pred_bbox, (-1, tf.shape(pred_bbox)[2], 4))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indicies.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    # print("sssssssssssssssssssssss    ", tf.size(target_bbox))
    loss = tf.cond(tf.size(target_bbox) > 0, lambda:smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox), lambda:tf.constant(0.0))  
    # loss = tf.switch(tf.size(target_bbox) > 0,
    #                 smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
    #                 tf.constant(0.0))
    loss = tf.reduce_mean(loss)
    loss = tf.reshape(loss, [1, 1])
    return loss

def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = tf.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = tf.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = tf.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = tf.cond(tf.size(y_true) > 0, lambda:tf.keras.losses.binary_crossentropy(y_true, y_pred), \
                   lambda:tf.constant(0.0))  
    loss = tf.reduce_mean(loss)
    loss = tf.reshape(loss, [1, 1])
    return loss

############################################################
#  Detection Target Layer
############################################################
def trim_zeros_graph(boxes, name=None):
    """Often boxes are represented with matricies of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros

def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps

def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:
    proposals: [N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [MAX_GT_INSTANCES] int class IDs
    gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
    gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    deltas: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
            Class-specific bbox refinements.
    masks: [TRAIN_ROIS_PER_IMAGE, height, width). Masks cropped to bbox
           boundaries and resized to neural network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                         name="trim_gt_masks")

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    crowd_masks = tf.gather(gt_masks, crowd_ix, axis=2)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Compute overlaps with crowd boxes [anchors, crowds]
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    # Determine postive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.argmax(positive_overlaps, axis=1)
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # Assign positive ROIs to GT masks
    # Permute masks to [N, height, width, 1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    # Pick the right mask for each ROI
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    # Compute mask targets
    boxes = positive_rois
    if config.USE_MINI_MASK:
        # Transform ROI corrdinates from normalized image space
        # to normalized mini-mask space.
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                     box_ids,
                                     config.MASK_SHAPE)
    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=3)

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    # binary cross entropy loss.
    masks = tf.round(masks)

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

    return rois, roi_gt_class_ids, deltas, masks

class DetectionTargetLayer():
    """Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
                    (dy, dx, log(dh), log(dw), class_id)]
                   Class-specific bbox refinements.
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width)
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config):
        self.config = config

    def compute(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: detection_targets_graph(
                w, x, y, z, self.config),
            self.config.IMAGES_PER_GPU, names=names)
        return outputs

def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]   # (y1, x1, y2, x2) window of image in in pixels
    active_class_ids = meta[:, 8:]
    return [image_id, image_shape, window, active_class_ids]
  

############################################################
#  ROIAlign Layer
############################################################

def log2_graph(x):
    """Implementatin of Log2. TF doesn't have a native implemenation."""
    return tf.log(x) / tf.log(2.0)

class PyramidROIAlign():
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """
    def __init__(self, pool_shape, image_shape):
        self.pool_shape = tuple(pool_shape)
        self.image_shape = tuple(image_shape)

    def run(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[1:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(
            self.image_shape[0] * self.image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        #batch , 256, 1
        roi_level = tf.minimum(5, tf.maximum(2,   4 + tf.cast(tf.round(roi_level), tf.int32)  )  )
        #batch , 256, 1
        roi_level = tf.squeeze(roi_level, 2)
        #batch , 256

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        # 4 feature map level
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indicies for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 10000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        return pooled

############################################################
#  Feature Pyramid Network Heads
############################################################

def refine_detections_graph(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)] where
        coordinates are in image domain.
    """
    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(tf.shape(probs)[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices) * config.BBOX_STD_DEV
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = apply_box_deltas_graph(
        rois, deltas_specific)
    # Convert coordiates to image domain
    # TODO: better to keep them normalized until later
    height, width = config.IMAGE_SHAPE[:2]
    refined_rois *= tf.constant([height, width, height, width], dtype=tf.float32)
    # Clip boxes to image window
    refined_rois = clip_boxes_graph(refined_rois, window)
    # Round and cast to int since we're deadling with pixels now
    refined_rois = tf.to_int32(tf.rint(refined_rois))

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois,   keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
                tf.to_float(tf.gather(pre_nms_rois, ixs)),
                tf.gather(pre_nms_scores, ixs),
                max_output_size=config.DETECTION_MAX_INSTANCES,
                iou_threshold=config.DETECTION_NMS_THRESHOLD)
        # Map indicies
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]
    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are in image domain.
    detections = tf.concat([
        tf.to_float(tf.gather(refined_rois, keep)),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections

class DetectionLayer():
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are in image domain
    """
    def __init__(self, config=None):
        self.config = config

    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # Run detection refinement graph on each item in the batch
        _, _, window, _ = parse_image_meta_graph(image_meta)
        detections_batch = utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.config),
            self.config.IMAGES_PER_GPU)
        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
        return tf.reshape(
            detections_batch,
            [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

############################################################
#  Mask rnn class
############################################################
class MaskRCNN():

    def __init__(self, mode="training" , config=None):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = "./logs/"
        self.pretrain_model = pkl.load(open("./logs/eval.pkl", 'rb'))
        self.BATCH_SIZE = config.IMAGES_PER_GPU
        self.learning_rate = config.LEARNING_RATE
        self.MAX_STEP = 1000000

    def rpn_process(self, in_feature, data_dict=None):
        
        shared = setool.conv_rpn(inputs_list= in_feature, name='rpn_conv_shared', \
                                 output_num=512, data_dict=data_dict)
        for i in range(len(shared)):
            shared[i] = tf.nn.relu(shared[i])
        #classfication
        x = setool.conv_rpn(inputs_list=shared, name='rpn_class_raw', output_num=2*3, kernel_size=1,\
                            stride=self.config.RPN_ANCHOR_STRIDE, padding='VALID', data_dict=data_dict)
        #x is a list
        rpn_class_logits = []
        rpn_probs = []
        for i in range(len(shared)):
            conv_out = tf.reshape(x[i], [self.BATCH_SIZE, -1, 2])
            rpn_class_logits.append(conv_out)
            rpn_probs.append(tf.nn.softmax(conv_out))

        x = setool.conv_rpn(inputs_list=shared, name='rpn_bbox_pred', output_num=4*3, kernel_size=1,\
                            stride=self.config.RPN_ANCHOR_STRIDE, padding='VALID', data_dict=data_dict)
        rpn_bbox = []
        for i in range(len(shared)):
            rpn_bbox.append(tf.reshape(x[i], [self.BATCH_SIZE, -1, 4]))

        return rpn_class_logits, rpn_probs, rpn_bbox

    def build(self, net_input, mode="training"):

        assert mode in ['training', 'inference']
        if mode=="training":
            images = net_input[0]
            input_image_meta = net_input[3]
            input_rpn_classfi = net_input[1]
            input_rpn_bbox = net_input[2]
            input_gt_class_ids = net_input[4]
            gt_boxes = net_input[5]
            input_gt_masks = net_input[6]
        else:
            images = tf.cast(net_input[0],tf.float32)
            input_image_meta = tf.cast(net_input[1], tf.int32)
            # self.pretrain_model = None

        # Image size must be dividable by 2 multiple times
        h, w = self.config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        C2, C3, C4, C5 = resnet_graph(images, "resnet101", stage5=True, data_dict=self.pretrain_model)
        #128*4*4*256
        P5 = setool.conv_op(input_op=C5, name='fpn_c5p5',kh=1, kw=1,  n_out=256, data_dict=self.pretrain_model)
        P4 = setool.conv_op(input_op=C4, name='fpn_c4p4',kh=1, kw=1,  n_out=256, data_dict=self.pretrain_model) + \
                     tf.keras.layers.UpSampling2D(size=(2,2))(P5)
        P3= setool.conv_op(input_op=C3, name='fpn_c3p3',kh=1, kw=1,  n_out=256, data_dict=self.pretrain_model) + \
                     tf.keras.layers.UpSampling2D(size=(2,2))(P4)
        P2= setool.conv_op(input_op=C2, name='fpn_c2p2',kh=1, kw=1,  n_out=256, data_dict=self.pretrain_model) + \
                     tf.keras.layers.UpSampling2D(size=(2,2))(P3)

        if mode=='training':
            P2 = setool.conv_op(input_op=P2, name='ffpn_p2',n_out=256, data_dict=self.pretrain_model)
        else:
            P2 = setool.conv_op(input_op=P2, name='fpn_p2',n_out=256, data_dict=self.pretrain_model)
        P3 = setool.conv_op(input_op=P3, name='fpn_p3',n_out=256, data_dict=self.pretrain_model)
        P4 = setool.conv_op(input_op=P4, name='fpn_p4',n_out=256, data_dict=self.pretrain_model)
        P5 = setool.conv_op(input_op=P5, name='fpn_p5',n_out=256, data_dict=self.pretrain_model)
        P6 = setool.mpool_op(input_tensor=P5, k=1, s=2, name="fpn_p6")

        rpn_feature_maps = [P2, P3, P4, P5, P6]
     
        mrcnn_feature_maps = [P2, P3, P4, P5]
        #(32, 64, 128, 256, 512)  3, [256,128,64,32,16], [4, 8, 16, 32, 64], 1
        #all list
        logits, probs, bbox = self.rpn_process(in_feature=rpn_feature_maps, data_dict=self.pretrain_model)

        rpn_class_logits = tf.concat(logits, 1)
        rpn_class = tf.concat(probs, 1)
        rpn_bbox = tf.concat(bbox, 1)
            
        self.anchors = utils.generate_pyramid_anchors(self.config.RPN_ANCHOR_SCALES,
                                                      self.config.RPN_ANCHOR_RATIOS,
                                                      self.config.BACKBONE_SHAPES,
                                                      self.config.BACKBONE_STRIDES,
                                                      self.config.RPN_ANCHOR_STRIDE)

        proposal_count = self.config.POST_NMS_ROIS_TRAINING if mode == "training"\
        else self.config.POST_NMS_ROIS_INFERENCE
        proposallayer = ProposalLayer(proposal_count=proposal_count,
                                      nms_threshold=self.config.RPN_NMS_THRESHOLD,
                                      anchors=self.anchors,
                                      config=self.config)
        rpn_rois = proposallayer.getproposal([rpn_class, rpn_bbox])

        # loss_class_rpn = rpn_class_loss_graph(input_rpn_classfi, rpn_class_logits)
        # loss_bbox_rpn = rpn_bbox_loss_graph(self.config, input_rpn_bbox, input_rpn_classfi, rpn_bbox)

        # return [loss_class_rpn, loss_bbox_rpn, rpn_rois]
        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates

        if mode == "training":

            _, _, _, active_class_ids = parse_image_meta_graph(input_image_meta)#@@@@@@@@@@@@
            target_rois = rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            compute_stage2_labels = DetectionTargetLayer(self.config)
            
            rois, target_class_ids, target_bbox, target_mask = compute_stage2_labels.compute([
                        target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

        # return rpn_class_logits, rpn_bbox, rpn_class, rpn_rois, target_class_ids
            # Network Heads
            # for i in range(len(mrcnn_feature_maps)):
            #     mrcnn_feature_maps[i] = tf.stop_gradient(mrcnn_feature_maps[i])
            Aligned_class = PyramidROIAlign([self.config.POOL_SIZE, self.config.POOL_SIZE],\
                                             self.config.IMAGE_SHAPE).run([rois] + mrcnn_feature_maps)

            Aligned_mask = PyramidROIAlign([self.config.MASK_POOL_SIZE, self.config.MASK_POOL_SIZE],\
                                             self.config.IMAGE_SHAPE).run([rois] + mrcnn_feature_maps)

            mrcnn_class_logits, mrcnn_probs, mrcnn_bbox = self.fpn_classifier_graph(inputs=Aligned_class, \
                                                        pool_size=self.config.POOL_SIZE, \
                                                        num_classes=self.config.NUM_CLASSES)

            mrcnn_mask = self.build_fpn_mask_graph(inputs=Aligned_mask, pool_size=self.config.MASK_POOL_SIZE,\
                                              num_classes=self.config.NUM_CLASSES)

            loss_class_rpn = rpn_class_loss_graph(input_rpn_classfi, rpn_class_logits)
            loss_bbox_rpn = rpn_bbox_loss_graph(self.config, input_rpn_bbox, input_rpn_classfi, rpn_bbox)
            loss_class_mrcnn = mrcnn_class_loss_graph(target_class_ids, mrcnn_class_logits, active_class_ids)
            loss_bbox_mrcnn = mrcnn_bbox_loss_graph(target_bbox, target_class_ids, mrcnn_bbox)
            loss_mask_mrcnn = mrcnn_mask_loss_graph(target_mask, target_class_ids, mrcnn_mask)

            return [loss_class_rpn, loss_bbox_rpn, loss_class_mrcnn, loss_bbox_mrcnn, loss_mask_mrcnn,\
                    rpn_class, target_rois]
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            Aligned_class = PyramidROIAlign([self.config.POOL_SIZE, self.config.POOL_SIZE],\
                                         self.config.IMAGE_SHAPE).run([rpn_rois] + mrcnn_feature_maps)

            mrcnn_class_logits, mrcnn_probs, mrcnn_bbox = self.fpn_classifier_graph(inputs=Aligned_class, \
                                            pool_size=self.config.POOL_SIZE, num_classes=self.config.NUM_CLASSES)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in image coordinates
            detections = DetectionLayer(self.config).call([rpn_rois, mrcnn_probs, mrcnn_bbox, input_image_meta])
            h, w = self.config.IMAGE_SHAPE[:2]
            detection_boxes = detections[..., :4] / np.array([h, w, h, w])

            Aligned_mask = PyramidROIAlign([self.config.MASK_POOL_SIZE, self.config.MASK_POOL_SIZE],\
                                                  self.config.IMAGE_SHAPE).run([detection_boxes] + mrcnn_feature_maps)
            mrcnn_mask = self.build_fpn_mask_graph(inputs=Aligned_mask, pool_size=self.config.MASK_POOL_SIZE,\
                                              num_classes=self.config.NUM_CLASSES)

            return [detections, mrcnn_probs, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox]

    def train(self, train_dataset=None):

        train_generator = coco_data_input.data_generator(train_dataset, self.config, shuffle=True,
                                                        batch_size=self.config.BATCH_SIZE)
        #if use BATCH_SIZE instead of None, it can print the size
        images = tf.placeholder(tf.float32, [None, 1024, 1024, 3])
        input_rpn_classfi = tf.placeholder(tf.float32, [None, 261888, 1])
        input_rpn_bbox = tf.placeholder(tf.float64, [None, 256, 4])
        input_image_meta = tf.placeholder(tf.float32, [None, 89])
        input_gt_class_ids = tf.placeholder(tf.float32, [None, 100])
        gt_boxes = tf.placeholder(tf.float32, [None, 100, 4])
        input_gt_masks = tf.placeholder(tf.float32, [None, 56, 56, 100])

        net_input = [images, input_rpn_classfi, input_rpn_bbox, input_image_meta, \
                     input_gt_class_ids, gt_boxes, input_gt_masks]

        net_out_train = self.build(net_input, mode="training")
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        my_global_step = tf.Variable(0, name='global_step', trainable=False)
        minimize_target = []
        for i in range(5):
            name = optimizer.minimize(net_out_train[i],  global_step=my_global_step)
            minimize_target.append(name)
        saver = tf.train.Saver(tf.global_variables())
        #tensorboard
        # summary_op = tf.summary.merge_all()
        sess = tf.Session()
        # load pre trained model from yourself for continue
        if os.path.exists('/home/liqi/work/324logs/checkpoint'):
            ckpt = tf.train.get_checkpoint_state('/home/liqi/work/324logs/')
            if ckpt and ckpt.model_checkpoint_path:
                print("Continue training from the model {}".format(ckpt.model_checkpoint_path))
                saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            init = tf.global_variables_initializer()
            sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #tensorboard
        # summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        try:
            loss_avg = [0, 0, 0, 0, 0]
            loss_name = ["rpn_class_loss", "rpn_loss_bbox", "mrcnn_loss_class",\
                               "mrcnn_loss_bbox", "mrcnn_loss_mask"]
            show_step = 2000
            for step in np.arange(self.MAX_STEP):
                if coord.should_stop():
                        break
                inputs, _ = next(train_generator)

                _, _, _, _, _,rpn_class_loss, rpn_loss_bbox, mrcnn_loss_class,\
                             mrcnn_loss_bbox, mrcnn_loss_mask,rpn_soft_class, temp_rois = \
                                 sess.run( minimize_target + net_out_train,
                                           feed_dict={images: inputs[0], 
                                                      input_rpn_classfi: inputs[1],
                                                      input_rpn_bbox: inputs[2],
                                                      input_image_meta: inputs[3], 
                                                      input_gt_class_ids: inputs[4],
                                                      gt_boxes:inputs[5]/1024.0, 
                                                      input_gt_masks:inputs[6]})

                loss_accept = [rpn_class_loss, rpn_loss_bbox, mrcnn_loss_class,\
                               mrcnn_loss_bbox, mrcnn_loss_mask]
                for index in range(len(loss_accept)):
                    loss_avg[index] += loss_accept[index].mean()/(show_step*1.0)

                if step % show_step == 0:
                    print ('Step: %d' % (step))
                    for i in range(len(loss_avg)):
                        print (loss_name[i]+'  :  ', loss_avg[i])
                    loss_avg = [0, 0, 0, 0, 0]
        
                if step % 100 == 0 or (step + 1) == self.MAX_STEP:
                    checkpoint_path = os.path.join(self.log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                    
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
            
        coord.join(threads)
        sess.close()

    def mold_image(images):
        """Takes RGB images with 0-255 values and subtraces
        the mean pixel and converts it to float. Expects image
        colors in RGB order.
        """
        return images.astype(np.float32) - self.config.MEAN_PIXEL

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image to fit the model expected size
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                max_dim=self.config.IMAGE_MAX_DIM,
                padding=self.config.IMAGE_PADDING)
            molded_image = molded_image.astype(np.float32) - self.config.MEAN_PIXEL
            # Build image_meta
            image_meta = coco_data_input.compose_image_meta(
                0, image.shape, window,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)]
        mrcnn_mask: [N, height, width, num_classes]
        image_shape: [height, width, depth] Original size of the image before resizing
        window: [y1, x1, y2, x2] Box in the image where the real image is
                excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Compute scale and shift to translate coordinates to image domain.
        h_scale = image_shape[0] / (window[2] - window[0])
        w_scale = image_shape[1] / (window[3] - window[1])
        scale = min(h_scale, w_scale)
        shift = window[:2]  # y, x
        scales = np.array([scale, scale, scale, scale])
        shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

        # Translate bounding boxes to image domain
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

        # Filter out detections with zero area. Often only happens in early
        # stages of training when the network weights are still a bit random.
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty((0,) + masks.shape[1:3])

        return boxes, class_ids, scores, full_masks

    def fpn_classifier_graph(self, inputs, pool_size, num_classes):
        """Builds the computation graph of the feature pyramid network classifier
        and regressor heads.
        """
        # ROI Pooling
        # Shape: [batch, num_boxes, pool_height, pool_width, channels]
        x = setool.conv_op(input_op=inputs, name='mrcnn_class_conv1',kh=pool_size, \
                           kw=pool_size,  n_out=1024, padding="VALID", data_dict=self.pretrain_model)


        x = setool.batch_norm(x=x, conv=True, phase_train=True, name='mrcnn_class_bn1', data_dict=self.pretrain_model)
        x = tf.nn.relu(x)

        x = setool.conv_op(input_op=x, name='mrcnn_class_conv2',kh=1, \
                           kw=1,  n_out=1024, data_dict=self.pretrain_model)

        x = setool.batch_norm(x=x, conv=True, phase_train=True, name='mrcnn_class_bn2', data_dict=self.pretrain_model)
        x = tf.nn.relu(x)

        shared = tf.squeeze(tf.squeeze(x, 2), 1)
        # Classifier head
        mrcnn_class_logits = setool.fc_op(input_op=shared, name='mrcnn_class_logits', n_out=num_classes, data_dict=self.pretrain_model)
        mrcnn_probs = tf.nn.softmax(logits=mrcnn_class_logits)
        # BBox head
        # [batch, boxes, num_classes * (dy, dx, log(dh), log(dw))]
        x = setool.fc_op(input_op=shared, name='mrcnn_bbox_fc', n_out=num_classes*4, data_dict=self.pretrain_model)
        # Reshape to [batch, boxes, num_classes, (dy, dx, log(dh), log(dw))]
        mrcnn_bbox = tf.reshape(x ,(tf.shape(x)[0], num_classes, 4), name="mrcnn_bbox")

        mrcnn_class_logits = tf.reshape(mrcnn_class_logits,[self.BATCH_SIZE, -1, tf.shape(mrcnn_class_logits)[-1]])
        mrcnn_probs = tf.reshape(mrcnn_probs,[self.BATCH_SIZE, -1, tf.shape(mrcnn_probs)[-1]])
        mrcnn_bbox = tf.reshape(mrcnn_bbox,[self.BATCH_SIZE, -1, tf.shape(mrcnn_bbox)[-2], tf.shape(mrcnn_bbox)[-1]])

        return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox

    def build_fpn_mask_graph(self, inputs, pool_size, num_classes):
        """Builds the computation graph of the mask head of Feature Pyramid Network.
        Returns: Masks [batch, roi_count, height, width, num_classes]
        """
        # ROI Pooling
        # Shape: [batch, boxes, pool_height, pool_width, channels]
        # Conv layers
        x = setool.conv_op(input_op=inputs, name='mrcnn_mask_conv1', n_out=256, data_dict=self.pretrain_model)
        x = setool.batch_norm(x=x, conv=True, phase_train=True, name='mrcnn_mask_bn1',data_dict=self.pretrain_model)
        x = tf.nn.relu(x)

        x = setool.conv_op(input_op=x, name='mrcnn_mask_conv2', n_out=256, data_dict=self.pretrain_model)
        x = setool.batch_norm(x=x, conv=True, phase_train=True, name='mrcnn_mask_bn2',data_dict=self.pretrain_model)
        x = tf.nn.relu(x)

        x = setool.conv_op(input_op=x, name='mrcnn_mask_conv3', n_out=256, data_dict=self.pretrain_model)
        x = setool.batch_norm(x=x, conv=True, phase_train=True, name='mrcnn_mask_bn3',data_dict=self.pretrain_model)
        x = tf.nn.relu(x)

        x = setool.conv_op(input_op=x, name='mrcnn_mask_conv4', n_out=256, data_dict=self.pretrain_model)
        x = setool.batch_norm(x=x, conv=True, phase_train=True, name='mrcnn_mask_bn4',data_dict=self.pretrain_model)
        x = tf.nn.relu(x)

        #512, 14, 14, 256
        x = setool.de_conv_op(input_op=x, name="mrcnn_mask_deconv", 
                              output_shape=[tf.shape(x)[0],2*pool_size,2*pool_size,256],\
                                output_num=256, data_dict=self.pretrain_model)
        x = tf.nn.relu(x)

        x = setool.conv_op(input_op=x, name='mrcnn_mask', kh=1, kw=1, n_out=81, data_dict=self.pretrain_model)
        x = tf.nn.sigmoid(x)
        #512, 28, 28, 81
        x = tf.reshape(x,[self.BATCH_SIZE, -1, tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[-1]])
        return x

    def loaddata(self):
        images = tf.placeholder(tf.float32, [None, 1024, 1024, 3])
        input_image_meta = tf.placeholder(tf.float32, [None, 89])
        inputs = [images, input_image_meta]
        Net_out = self.build(inputs, mode='inference')

        return [Net_out, images, input_image_meta]

    def detect(self, images, dataprepare, sess):
        assert self.mode == "inference", "Create model in inference mode."
        assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)
        detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, \
            rois, rpn_class, rpn_bbox = sess.run(dataprepare[0], feed_dict={dataprepare[1]: molded_images, 
                                                      dataprepare[2]: image_metas})

        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

if __name__ == '__main__':
    config = config.Config()
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')

    args = parser.parse_args()
    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = coco_data_input.CocoDataset()
        dataset_train.load_coco(args.dataset, "train", year=2014)
        dataset_train.load_coco(args.dataset, "valminusminival", year=2014)
        dataset_train.prepare()
        dataset_val.load_coco(args.dataset, "val", year=2014)
        dataset_val.prepare()

    liqi = MaskRCNN(config=config)
    liqi.train(train_dataset=dataset_train)