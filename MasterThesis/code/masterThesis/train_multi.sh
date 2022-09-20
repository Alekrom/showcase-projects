#!/bin/bash

command_list=()
log_path_list=()

log_path="test_run1"
log_path_list[0]=$log_path
command="python main.py --model resnet18_e1 --data-dir /data/imagenet1k -j 8 --weight-quantization dorefa --activation-method round --clip-threshold 1.3 --fp-downsample-sc --pool-downsample-sc --dataset imagenet --epochs 60 --batch-size 16 --log /test_run_1_log --log-interval 500 --gpus 0 --lr 0.002 --lr-mode cosine --bit-widths 32 8 4 2 1 --target-bits 1 --mode hybrid --optimizer radam --result_path ../test_run1/"
command_list[0]=$command

log_path="test_run2"
log_path_list[1]=$log_path
command="python main.py --model resnet18_e1 --data-dir /data/imagenet1k -j 8 --weight-quantization dorefa --activation-method round --clip-threshold 1.3 --fp-downsample-sc --pool-downsample-sc --dataset imagenet --epochs 60 --batch-size 16 --log test_run_2_log --log-interval 500 --gpus 0 --lr 0.002 --lr-mode cosine --bit-widths 32 8 4 2 1 --target-bits 1 --mode hybrid --optimizer radam --result_path ../test_run2/ --teacher ResNet18_v1 --kd-mode simple --kd-teacher-mode pretrained_external"
command_list[1]=$command

log_path="test_run3"
log_path_list[2]=$log_path
command="python main.py --model resnet18_e1 --weight-quantization dorefa --activation-method round --clip-threshold 1.3 --fp-downsample-sc --pool-downsample-sc --dataset cifar100 --epochs 150 --batch-size 128 --log test_run3 --log-interval 100 --gpus 0 --lr 0.01 --lr-mode cosine --bit-widths 32 8 4 2 --target-bits 4 --optimizer radam --result_path ../test_run3/ --wd 0.0001 --teacher cifar_resnet110_v2 --kd-mode progressive --kd-teacher-mode untrained_external --teacher-lr 0.1 --teacher-wd 0.0001 --teacher-lr-mode cosine --teacher-optimizer sgd"
command_list[2]=$command

log_path="test_run4"
log_path_list[3]=$log_path
command="python main.py --model resnet18_e1 --weight-quantization dorefa --activation-method round --clip-threshold 1.3 --fp-downsample-sc --pool-downsample-sc --dataset cifar100 --epochs 150 --batch-size 128 --log test_run4 --log-interval 100 --gpus 0 --lr 0.01 --lr-mode cosine --bit-widths 32 8 4 2 1 --target-bits 1 --optimizer radam --result_path ../test_run4/ --fp-weights"
command_list[3]=$command

log_path="test_run5"
log_path_list[4]=$log_path
command="python main.py --model resnet18_e1 --weight-quantization dorefa --activation-method round --clip-threshold 1.3 --fp-downsample-sc --pool-downsample-sc --dataset cifar100 --epochs 150 --batch-size 128 --log test_run5 --log-interval 100 --gpus 0 --lr 0.01 --lr-mode cosine --bit-widths 32 8 4 2 1 --target-bits 1 --optimizer radam --result_path ../test_run5/ --ensemble-teacher --resume ../teacher_params/ensemble_328421_kd_unext_progressive_resnet110_two_stage2021-07-26_02:53:43.032204_best_params.params --teacher-params ../teacher_params/ensemble_328421_kd_unext_progressive_resnet110_two_stage2021-07-26_02:53:43.032204_best_params.params"
command_list[4]=$command

# move to correct dir
cd ../home/alexander/code/masterThesis/


length=${#command_list[@]}
for (( i = 0; i < length; i++ )); do
  export MXNET_ENABLE_GPU_P2P=0
  echo ${command_list[i]}
  echo ${log_path_list[i]}

  echo ${command_list[i]} >> ../results/${log_path_list[i]}.txt
  eval "${command_list[i]}"
done



#cd ../home/alexander/code/masterThesis/analysis/
#eval "python verification_image.py --model resnet18_e1 --weight-quantization dorefa --activation-method round --clip-threshold 1.3 --fp-downsample-sc --pool-downsample-sc --dataset cifar100 --epochs 2 --batch-size 128 --log resnet56_v2_unext --gpus 0 --lr 0.1 --lr-mode cosine --bit-widths 32 8 4 2 1"

