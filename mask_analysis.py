#!/usr/bin/env python
# coding: utf-8

# In[216]:


from __future__ import print_function
from __future__ import division

from ast import literal_eval

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
import time
import h5py
import argparse
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

import masked_networks
from tf_plus import learning_phase, batchnorm_learning_phase
from tf_plus import sess_run_dict, add_classification_losses
from tf_plus import summarize_weights
from train_supermask import make_parser, read_input_data,     init_model, load_initial_weights, split_and_shape


# In[222]:


metaparser = argparse.ArgumentParser()
metaparser.add_argument('--experiment_name', type=str, required=True)
metaparser.add_argument('--pretrained_epochs', type=int, required=True)
meta_args = metaparser.parse_args()


# In[224]:


def build_input_dir(seed, meta_args):
    attempt_num = 0
    experiment_name = meta_args.experiment_name
    pretrained_epochs = meta_args.pretrained_epochs

    if experiment_name == "control_3":
        input_dir = "./results/iter_lot_fc_orig/learned_supermasks_seed_{seed}_attempt_{attempt_num}/run1".format(
            seed=seed,
            attempt_num=attempt_num)

    elif experiment_name == "pretrained_supermask":
        input_dir = "./results/iter_lot_fc_orig/learned_supermasks_pre_trained_{pretrained_epochs}_epochs_seed_{seed}_{attempt_num}/run1".format(
            pretrained_epochs=pretrained_epochs,
            seed=seed,
            attempt_num=attempt_num)
    return input_dir


parser = make_parser()
# Have a seed just to satisfy the requirements
seed = 1
input_dir = build_input_dir(seed, meta_args)

args_str = """--train_h5 ./data/mnist_train.h5 --test_h5 ./data/mnist_test.h5
--arch fc_mask --opt sgd --lr 100 --num_epochs 500 --print_every 220
--eval_every 220 --log_every 220 --save_weights --save_every 22000
--tf_seed {}
--init_weights_h5 {}
""".format(seed, input_dir).split()
args = parser.parse_args(args_str)


# In[3]:


train_x, train_y = read_input_data(args.train_h5)
test_x, test_y = read_input_data(args.test_h5) # used as val for now
images_scale = np.max(train_x)
if images_scale > 1:
    print('Normalizing images by a factor of {}'.format(images_scale))
    train_x = train_x / images_scale
    test_x = test_x / images_scale


if args.test_batch_size == 0:
    args.test_batch_size = test_y.shape[0]

print('Data shapes:', train_x.shape, train_y.shape, test_x.shape, test_y.shape)
if train_y.shape[0] % args.train_batch_size != 0:
    print("WARNING batch size doesn't divide train set evenly")
if train_y.shape[0] % args.large_batch_size != 0:
    print("WARNING large batch size doesn't divide train set evenly")
if test_y.shape[0] % args.test_batch_size != 0:
    print("WARNING batch size doesn't divide test set evenly")

# build model, masked networks
if args.arch == 'fc_mask':
    model = masked_networks.build_fc_supermask(args)
else:
    raise Exception("Not prepared for non fc_mask model")
        
init_model(model, args)

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())


# In[215]:


def sigmoid(x):
    return 1/(1+np.exp(-x))


def visualize_mask_weights(mask_layers, seed):
    num_bins = 20
    for i, mask_layer in enumerate(mask_layers):
        plt.subplot(len(mask_layers), 2, i * 2 + 1)
        plt.hist(mask_layer.flatten(), bins=num_bins)
        plt.xlabel("Raw mask values at layer {}".format(i))
        plt.subplot(len(mask_layers), 2, i * 2 + 2)
        plt.hist(sigmoid(mask_layer.flatten()), bins=num_bins)
        plt.xlabel("Sigmoided mask values at layer {}".format(i))
    plt.tight_layout()
    plt.savefig(os.path.join('results/iter_lot_fc_orig/figs',
        "mask_dists_epochs_{}_seed_{}.png".format(meta_args.pretrained_epochs, seed)))
    
def get_test_accs(run_dir):
    test_accs = []
    for filename in os.listdir(run_dir):
        if 'tfevents' in filename:
            for e in tf.compat.v1.train.summary_iterator(os.path.join(
                run_dir, filename
            )):
                for v in e.summary.value:
                    if v.tag == 'eval_test_acc':
                        test_accs.append(v.simple_value)
    return np.array(test_accs)


# In[225]:


def run_analysis_on_seed(seed, meta_args):
    seed_info = {
        "seed": seed,
        "regular_epochs": meta_args.pretrained_epochs,
        "supermask_epochs": 500,
        "experiment_name": meta_args.experiment_name,
        "has_supermask": 1,
        "has_lth": 0,
        "test_accuracy": None
    }
    
    args.init_weights_h5 = build_input_dir(seed, meta_args)
    if not args.init_weights_h5.endswith('/weights'):
        h5file = os.path.join(args.init_weights_h5, 'weights')
    else:
        h5file = args.init_weights_h5
    hf_weights = h5py.File(h5file, 'r')
    all_weights = hf_weights.get('all_weights')
    print("Number of weight copies:", len(all_weights))
    init_weights_flat = all_weights[0]
    final_weights_flat = all_weights[-1]
    current_mask = np.array(hf_weights.get('mask_values'))

    shapes = [literal_eval(s) for s in hf_weights.attrs['var_shapes'].decode('utf-8').split(';')]
    hf_weights.close()

    weight_values = split_and_shape(init_weights_flat, shapes)
    final_weight_values = split_and_shape(final_weights_flat, shapes)
    gk = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
    if len(gk) > 9:
        print("You need to restart the kernel - graphkeys have been replicated" +
              " and there's no going back")

    for i, w in enumerate(gk):
       #if 'mask' not in w.name: # HACK for biased masks
        print('loading weights for layer {}: {}'.format(i, w.name))
        w.load(weight_values[i], session=sess)


    mask_layers = final_weight_values[2::3]
    print("Basic info")
    all_mask_weights = []
    for mask_layer in mask_layers:
        print("Shape:", mask_layer.shape)
        print("Average fraction masked:", 1-sigmoid(mask_layer).mean())
        print("Min: {}, Max: {}\n".format(
            mask_layer.min(), mask_layer.max()))
        all_mask_weights.append(mask_layer.flatten())
    all_mask_weights = np.concatenate(all_mask_weights)
    print("Total average fraction masked:", 1-sigmoid(all_mask_weights).mean())
    
    # Visualize and save img of mask weight distribution
    visualize_mask_weights(mask_layers, seed)
    
    seed_info["test_accuracy"] = get_test_accs(args.init_weights_h5).max()
    return seed_info


# In[226]:


seed_infos = []

for seed in range(1, 11):
    seed_info = run_analysis_on_seed(seed, meta_args)
    seed_infos.append(seed_info)
    
df = pd.DataFrame(seed_infos)


# In[220]:


df


# In[231]:


df.to_csv('results/iter_lot_fc_orig/results_summary_{}_{}.csv'.format(meta_args.experiment_name, meta_args.pretrained_epochs))