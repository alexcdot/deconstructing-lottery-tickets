#!/bin/bash
attempt_num=0

for seed in 1 2 3 4 5 6 7 8 9 10
do
    for pretrained_epochs in 11 22 33 44 55
    do
        echo $seed $pretrained_epochs
        bash print_train_command.sh iter fc fc_lot_${pretrained_epochs}_epochs_seed_${seed}_${attempt_num} ${seed} t ${pretrained_epochs}
    done
done

for seed in 1 2 3 4 5 6 7 8 9 10
do
    for pretrained_epochs in 11 22 33 44 55
    do
        echo $seed $pretrained_epochs
        python train_supermask.py \
            --output_dir ./results/iter_lot_fc_orig/learned_supermasks_pre_trained_${pretrained_epochs}_epochs_seed_${seed}_${attempt_num}/run1/ \
            --train_h5 ./data/mnist_train.h5 --test_h5 ./data/mnist_test.h5 \
            --arch fc_mask --opt sgd --lr 100 --num_epochs 500 --print_every 220 \
            --eval_every 220 --log_every 220 --save_weights --save_every 22000 \
            --tf_seed ${seed} \
            --init_weights_h5 ./results/iter_lot_fc_orig/fc_lot_${pretrained_epochs}_epochs_seed_${seed}_${attempt_num}
    done
done
