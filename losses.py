__author__ = 'krishnch'

import tensorflow as tf
import numpy as np

class lossObj:

    #define possible loss functions like dice score, cross entropy, weighted cross entropy

    def __init__(self):
        print('loss init')

    def dice_loss_with_backgrnd(self, logits, labels, epsilon=1e-10, from_label=1, to_label=-1):
        '''
        Calculate a dice loss defined as `1-foreround_dice`. Default mode assumes that the 0 label
         denotes background and the remaining labels are foreground.
        :param logits: Network output before softmax
        :param labels: ground truth label masks
        :param epsilon: A small constant to avoid division by 0
        :param from_label: First label to evaluate
        :param to_label: Last label to evaluate
        :return: Dice loss
        '''

        with tf.name_scope('dice_loss'):

            prediction = tf.nn.softmax(logits)

            intersection = tf.multiply(prediction, labels)
            intersec_per_img_per_lab = tf.reduce_sum(intersection, axis=[1, 2])

            l = tf.reduce_sum(prediction, axis=[1, 2])
            r = tf.reduce_sum(labels, axis=[1, 2])

            dices_per_subj = 2 * intersec_per_img_per_lab / (l + r + epsilon)

            #loss = 1 - tf.reduce_mean(tf.slice(dices_per_subj, (0, from_label), (-1, to_label)))
            loss = 1 - tf.reduce_mean(dices_per_subj)
        #return loss,intersec_per_img_per_lab,l,r,dices_per_subj
        return loss

    def dice_loss_without_backgrnd(self, logits, labels, epsilon=1e-10, from_label=1, to_label=-1):
        '''
        Calculate a dice loss defined as `1-foreround_dice`. Default mode assumes that the 0 label
         denotes background and the remaining labels are foreground.
        :param logits: Network output before softmax
        :param labels: ground truth label masks
        :param epsilon: A small constant to avoid division by 0
        :param from_label: First label to evaluate
        :param to_label: Last label to evaluate
        :return: Dice loss
        '''

        with tf.name_scope('dice_loss'):

            prediction = tf.nn.softmax(logits)

            intersection = tf.multiply(prediction, labels)
            intersec_per_img_per_lab = tf.reduce_sum(intersection, axis=[1, 2])

            l = tf.reduce_sum(prediction, axis=[1, 2])
            r = tf.reduce_sum(labels, axis=[1, 2])

            dices_per_subj = 2 * intersec_per_img_per_lab / (l + r + epsilon)

            loss = 1 - tf.reduce_mean(tf.slice(dices_per_subj, (0, from_label), (-1, to_label)))
            #loss = 1 - tf.reduce_mean(dices_per_subj)
        return loss

    def dice_loss_with_backgrnd_wgt(self, logits, labels, class_wgts, n_classes, epsilon=1e-10, from_label=1, to_label=-1):
        '''
        Calculate a dice loss defined as `1-foreround_dice`. Default mode assumes that the 0 label
         denotes background and the remaining labels are foreground.
        :param logits: Network output before softmax
        :param labels: ground truth label masks
        :param epsilon: A small constant to avoid division by 0
        :param from_label: First label to evaluate
        :param to_label: Last label to evaluate
        :return: Dice loss
        '''

        with tf.name_scope('dice_loss'):

            prediction = tf.nn.softmax(logits)

            intersection = tf.multiply(prediction, labels)
            intersec_per_img_per_lab = tf.reduce_sum(intersection, axis=[1, 2])

            l = tf.reduce_sum(prediction, axis=[1, 2])
            r = tf.reduce_sum(labels, axis=[1, 2])

            dices_per_subj = 2 * intersec_per_img_per_lab / (l + r + epsilon)
            wgted_dices_per_subj = tf.multiply(dices_per_subj, class_wgts)
            loss = 1 - (n_classes * tf.reduce_mean(wgted_dices_per_subj))
        return loss

    #
    #def dice_loss_with_backgrnd_wgt(logits, labels, class_wgts, n_classes, epsilon=1e-10, from_label=1, to_label=-1):
    #    '''
    #    Calculate a dice loss defined as `1-foreround_dice`. Default mode assumes that the 0 label
    #     denotes background and the remaining labels are foreground.
    #    :param logits: Network output before softmax
    #    :param labels: ground truth label masks
    #    :param epsilon: A small constant to avoid division by 0
    #    :param from_label: First label to evaluate
    #    :param to_label: Last label to evaluate
    #    :return: Dice loss
    #    '''
    #
    #    with tf.name_scope('dice_loss'):
    #
    #        prediction = tf.nn.softmax(logits)
    #
    #        intersection = tf.multiply(prediction, labels)
    #        intersec_per_img_per_lab = tf.reduce_sum(intersection, axis=[1, 2])
    #
    #        l = tf.reduce_sum(prediction, axis=[1, 2])
    #        r = tf.reduce_sum(labels, axis=[1, 2])
    #        dices_per_subj = 2 * intersec_per_img_per_lab / (l + r + epsilon)
    #        wgted_dice_per_subj = tf.multiply(dices_per_subj,class_wgts)
    #        loss = 1 - (n_classes * tf.reduce_mean(wgted_dice_per_subj))
    #
    #    return loss
    #
    def pixel_wise_cross_entropy_loss(self, logits, labels):
        '''
        Simple wrapper for the normal tensorflow cross entropy loss
        '''

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        return loss


    def pixel_wise_cross_entropy_loss_weighted(self, logits, labels, class_weights):
        '''
        Weighted cross entropy loss, with a weight per class
        :param logits: Network output before softmax
        :param labels: Ground truth masks
        :param class_weights: A list of the weights for each class
        :return: weighted cross entropy loss
        '''

        n_class = len(class_weights)

        flat_logits = tf.reshape(logits, [-1, n_class])
        flat_labels = tf.reshape(labels, [-1, n_class])

        class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

        weight_map = tf.multiply(flat_labels, class_weights)
        weight_map = tf.reduce_sum(weight_map, axis=1)

        loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits, labels=flat_labels)
        weighted_loss = tf.multiply(loss_map, weight_map)

        loss = tf.reduce_mean(weighted_loss)

        return loss

    def pixel_wise_cross_entropy_loss_weighted_nn(self, logits, labels, class_weights):
        '''
        Weighted cross entropy loss, with a weight per class
        :param logits: Network output before softmax
        :param labels: Ground truth masks
        :param class_weights: A list of the weights for each class
        :return: weighted cross entropy loss
        '''

        # deduce weights for batch samples based on their true label
        weights = tf.reduce_sum(class_weights * labels, axis=3)

        # For weighted error
        # compute your (unweighted) softmax cross entropy loss
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels)
        # apply the weights, relying on broadcasting of the multiplication
        print("unweighted_losses",unweighted_losses)
        weighted_losses = unweighted_losses * weights
        # reduce the result to get your final loss
        loss = tf.reduce_mean(weighted_losses)

        return loss

    def pixel_wise_cross_entropy_loss_weighted_edge_enhance(self, logits, labels, weights):
        '''
        Weighted cross entropy loss, with a weight per class
        :param logits: Network output before softmax
        :param labels: Ground truth masks
        :param class_weights: A list of the weights for each class
        :return: weighted cross entropy loss
        '''

        # deduce weights for batch samples based on their true label
        #weights = tf.reduce_sum(class_weights * labels, axis=3)

        # For weighted error
        # compute your (unweighted) softmax cross entropy loss
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels)
        # apply the weights, relying on broadcasting of the multiplication
        print("unweighted_losses",unweighted_losses)
        weighted_losses = unweighted_losses * weights
        # reduce the result to get your final loss
        loss = tf.reduce_mean(weighted_losses)

        return loss
