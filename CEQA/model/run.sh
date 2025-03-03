#!/bin/sh

#SBATCH -p v
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem=60g
#SBATCH -o train_exp5.out


#project_dir=/home/u00483/repos/PDNC_deepspeed/code2 
#python evaluate.py >run_out/evaluate.out 2>&1
#accelerate launch --config_file zero2_character.yaml train.py --optimize_direction max --initial_optimal_value 0 --batch_size 4 --log_dir /home/u00483/repos/PDNC_deepspeed/log/exp2
python train.py --layer_num 1 --epoch 22 --lr 5e-5  --log_dir log/log_base/exp1 --train_file ./../finetune_data/PDNC_finetune/all_test.json  --validate_file ./../finetune_data/PDNC_finetune/all_test.json
