"""
Neural network chatbot.  Adapted from stanford tensorflow course.

"""

from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import sys
import time
import numpy as np
import tensorflow as tf
from model import ChatBotModel
import config
import data
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def _get_random_bucket(train_bucket_scale):
    """
    Get a random bucket from which to choose a training sample
    :param train_bucket_scale:
    :return:
    """
    rand = random.random()
    return min([i for i in range(len(train_bucket_scale)) if train_bucket_scale[i] > rand])


def _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks):
    """
    Assert that the encoder inputs, decoder inputs, and decoder masks are of the expected lengths
    :param encoder_size:
    :param decoder_size:
    :param encoder_inputs:
    :param decoder_inputs:
    :param decoder_masks:
    :return:
    """
    if len(encoder_inputs) != encoder_size:
        raise ValueError("Encoder length must be equal to the one in bucket," 
                         "%d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
        raise ValueError("Decoder length must be equal to the one in bucket," 
                         "%d != %d." % (len(decoder_inputs), decoder_size))
    if len(decoder_masks) != decoder_size:
        raise ValueError("Weights length must be equal to the one in bucket," 
                         "%d != %d." % (len(decoder_masks), decoder_size))


def run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, forward_only):
    """
    Run one step in training.
    @forward_only: boolean value to decide whether a backward path should be created.
    forward_only is set to True when you just want to evaluate on the test, set, or when you want to run the bot in chat
    mode.
    :param sess:
    :param model:
    :param encoder_inputs:
    :param decoder_inputs:
    :param decoder_masks:
    :param bucket_id:
    :param forward_only:
    :return:
    """
    encoder_size, decoder_size = config.BUCKETS[bucket_id]
    _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks)

    # input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for step in range(encoder_size):
        input_feed[model.encoder_inputs[step].name] = encoder_inputs[step]
    for step in range(decoder_size):
        input_feed[model.decoder_inputs[step].name] = decoder_inputs[step]
        input_feed[model.decoder_masks[step].name] = decoder_masks[step]

    last_target = model.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([model.batch_size], dtype=np.int32)

    # output feed: depends on whether we do a backwards step or not.
    if not forward_only:
        output_feed = [model.train_ops[bucket_id],
                       model.gradient_norms[bucket_id],
                       model.losses[bucket_id]]
    else:
        output_feed = [model.losses[bucket_id]]
        for step in range(decoder_size):
            output_feed.append(model.outputs[bucket_id][step])

    outputs = sess.run(output_feed, input_feed)
    if not forward_only:
        return outputs[1], outputs[2], None # gradient norm, loss, no outputs.
    else:
        return None, outputs[0], outputs[1:] # no gradient norm, loss, outputs.


def _get_buckets():
    """
    Load the dataset into buckets based on their lengths. train_bucket_scale is the interval that'll help us choose a
    random bucket later on.
    :return:
    """
    test_buckets = data.load_data('test_ids.enc', 'test_ids.dec')
    data_buckets = data.load_data('train_ids.enc', 'train_ids.dec')
    train_bucket_sizes = [len(data_buckets[b]) for b in range(len(config.BUCKETS))]
    print("Number of samples n each bucket:\n", train_bucket_sizes)
    train_total_size = sum(train_bucket_sizes)
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in range(len(train_bucket_sizes))]
    print('Bucket scale:\n', train_buckets_scale)
    return test_buckets, data_buckets, train_buckets_scale


def _get_skip_step(iteration):
    """
    How many steps should the model train before it saves all the weights.
    :param iteration:
    :return:
    """
    if iteration < 100:
        return 30
    return 100


def _check_restore_parameters(sess, saver):
    """
    Restore the previously trained parameters if there are any.
    :param sess:
    :param saver:
    :return:
    """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.CPT_PATH + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading parameters for the ChatBot.")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Initializing fresh parameters for the ChatBot.")


def _eval_test_set(sess, model, test_buckets):
    """
    Evaluate on test set.
    :param sess:
    :param model:
    :param test_buckets:
    :return:
    """
    for bucket_id in range(len(config.BUCKETS)):
        if len(test_buckets[bucket_id]) == 0:
            print("  Test: empty bucket %d" % (bucket_id))
            continue
        start = time.time()
        encoder_inputs, decoder_inputs, decoder_masks = data.get_batch(test_buckets[bucket_id],
                                                                       bucket_id,
                                                                       batch_size=config.BATCH_SIZE)
        _, step_loss, _ = run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, True)
        print("Test bucket {0}: loss {1}, time {2}".format(bucket_id, step_loss, time.time() - start))


def train():
    """
    Train the bot.
    :return:
    """
    pass


def _get_user_input():
    """
    Get user's input, which will be transformed into encoder input later.
    :return:
    """
    print(">", end="")
    sys.stdout.flush()
    return sys.stdin.readline()


def find_right_bucket(length):
    """
    Find the proper bucket for an encoder input based on its length
    :param length:
    :return:
    """
    return min([b for b in range(len(config.BUCKETS)) if config.BUCKETS[b][0] >= length])


def _construct_reponse(output_logits, inv_dec_vocab):
    """
    Construct a reponse to the user's encoder input.
    @output_logits: the outputs from sequence to sequence to sequence wrapper.
    output_logits is decoder_size np array, each of dim 1 x DEC_VOCAB
    This is a greedy decoder - outputs are just argmaxes of output_logits.
    :param output_logits:
    :param inv_dec_vocab:
    :return:
    """
    pass


def chat():
    """
    in test mode, we don't create the backward path
    :return:
    """
    pass


def main():
    pass


if __name__ == '__main__':
    main()