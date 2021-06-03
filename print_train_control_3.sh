#!/bin/bash
attempt_num=1

for seed in 1 2 3 4 5 6 7 8 9 10
do
    echo $seed
    python train_supermask.py \
        --output_dir ./results/iter_lot_fc_orig/learned_supermasks_seed_${seed}_attempt_${attempt_num}/run1/ \
        --train_h5 ./data/mnist_train.h5 --test_h5 ./data/mnist_test.h5 \
        --arch fc_mask --opt sgd --lr 100 --num_epochs 100 --print_every 220 \
        --eval_every 220 --log_every 220 --save_weights --save_every 22000 \
        --tf_seed ${seed}
done
