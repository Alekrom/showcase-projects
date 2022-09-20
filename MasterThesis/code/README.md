# Code for "Training of Ensembles of Quantized Neural Networks with Shared Weights Using Knowledge Distillation"

## Setup

This section gives information on necessary setup steps, to run the Code. 

#### Install BMXNet
To run the code BMXNet needs to be installed. BMXNet\[1\] is a fork of [mxnet](https://mxnet.apache.org/versions/1.9.0/), a deep learning framework developed by Apache. 

The version of BMXNet used during this work is based on mxnet v1.8.0.

Setup of BMXNet according to official [BMXNet repository](https://gitlab.hpi.de/hpi-xnor/bmxnet-projects/bmxnet): 

We use CMake to build the project.
Make sure to install dependencies as listed on the official [MXNet page](https://mxnet.apache.org/get_started/build_from_source#installing-mxnet's-recommended-dependencies)
If you install CUDA 10, you will need CMake >=3.12.2
Adjust settings in cmake (build-type Release or Debug, configure CUDA, OpenBLAS or Atlas, OpenCV, OpenMP etc.).
Further, we recommend Ninja as a build system for faster builds (Ubuntu: sudo apt-get install ninja-build).

```bash
git clone --recursive https://github.com/hpi-xnor/BMXNet-v2.git # remember to include the --recursive
cd BMXNet-v2
mkdir build && cd build
cmake .. -G Ninja # if any error occurs, apply ccmake or cmake-gui to adjust the cmake config.
ccmake . # or GUI cmake
ninja
```

Build the MXNet Python binding

Step 1 Install prerequisites - python, setup-tools, python-pip and numpy.
```bash
sudo apt-get install -y python-dev python3-dev virtualenv
wget -nv https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
python2 get-pip.py
```

Step 1b (Optional) Create or activate a [virtualenv](https://virtualenv.pypa.io/).

Step 2 Install the MXNet Python binding.
```bash
cd <mxnet-root>/python
pip install -e .
```

If your mxnet python binding still not works, you can add the location of the libray to your ``LD_LIBRARY_PATH`` as well as the mxnet python folder to your ``PYTHONPATH``:
```bash
$ export LD_LIBRARY_PATH=<mxnet-root>/build/Release
$ export PYTHONPATH=<mxnet-root>/python
```

## Running the Thesis Code:

All code developed during the project is given in the 'ThesisCodeAlexanderKromer' directory. The Code is build on the repository bmxnet-examples\[2\].

A training can be started by executing main.py with the appropriate parameters. 

We will give example commands for a few trainings to allow an easy start with training the models. Important parameters are described here. Detailed information on all parameters can be found in util/arg\_parser.py.

#### Example 1: Training Ensemble on ImageNet without knowledge distillation
```bash
python main.py --model resnet18_e1 --data-dir /data/imagenet1k -j 8 --initial-layers imagenet --weight-quantization dorefa --activation-method round --clip-threshold 1.3 --fp-downsample-sc --pool-downsample-sc --dataset imagenet --epochs 60 --batch-size 128 --log /test_run_1_log --log-interval 500 --gpus 0 --lr 0.002 --lr-mode cosine --bit-widths 32 8 4 2 1 --target-bits 1 --mode hybrid --optimizer radam --result_path ../test_run1/ 
```

 --model parameter: The architecture of the models in the ensemble. Currently only resnet18_e1 is tested throughly.

 -- bit-widths: Specifies which bit-widths to include in the ensemble. 
 
 
#### Example 2: Training Ensemble on ImageNet using a pretrained ResNet18 teacher and simple knowledge distillation between teacher and ensemble

```bash
python main.py --model resnet18_e1 --data-dir /data/imagenet1k -j 8 --initial-layers imagenet --weight-quantization dorefa --activation-method round --clip-threshold 1.3 --fp-downsample-sc --pool-downsample-sc --dataset imagenet --epochs 60 --batch-size 128 --log test_run_2_log --log-interval 500 --gpus 0 --lr 0.002 --lr-mode cosine --bit-widths 32 8 4 2 1 --target-bits 1 --mode hybrid --optimizer radam --result_path ../test_run2/ --teacher ResNet18_v1 --kd-mode simple --kd-teacher-mode pretrained_external
```

--teacher: Specifies the teacher architecture

--kd-mode: Specifies the mode of knowledge distillation between ensemble and teacher. Can be 'none', 'simple' or 'progressive'

--kd-teacher-mode: Specifies the training level of the teacher. Currently impemented choices are 'none', 'pretrained_external' and 'untrained_external'.

--teacher-params: On ImageNet the pretrained teacher is loaded from gluon (and therefore not specified in the example above). If we want to use parameters of a pretrained teachers that are saved locally, we can load them instead using this parameter. 


#### Example 3: Training Ensemble on CIFAR-100 using an untrained ResNet110 teacher and progressive knowledge distillation within the ensemble

```bash
python main.py --model resnet18_e1 --weight-quantization dorefa --activation-method round --clip-threshold 1.3 --fp-downsample-sc --pool-downsample-sc --dataset cifar100 --epochs 150 --batch-size 128 --log test_run3 --log-interval 100 --gpus 0 --lr 0.01 --lr-mode cosine --bit-widths 32 8 4 2 --target-bits 4 --optimizer radam --result_path ../test_run3/ --wd 0.0001 --teacher cifar_resnet110_v2 --kd-mode progressive --kd-teacher-mode untrained_external --teacher-lr 0.1 --teacher-wd 0.0001 --teacher-lr-mode cosine --teacher-optimizer sgd
```

--kd-teacher-mode: Use an untrained teacher

The training hyperparameters for the teacher can be specified seperately from the ensembles hyperparameters. The corresponding arguments are prefixed with 'teacher', e.g.  --teacher-lr and --teacher-optimizer

#### Example 4: Training first stage of a Multi-Stage Training on CIFAR-100 without Knowledge distillaion

```bash
python main.py --model resnet18_e1 --weight-quantization dorefa --activation-method round --clip-threshold 1.3 --fp-downsample-sc --pool-downsample-sc --dataset cifar100 --epochs 150 --batch-size 128 --log test_run4 --log-interval 100 --gpus 0 --lr 0.01 --lr-mode cosine --bit-widths 32 8 4 2 1 --target-bits 1 --optimizer radam --result_path ../test_run4/ --fp-weights
```

--fp-weights: keeps the weights of the model in full precision instead of quantizing them


#### Example 5: Training second stage of Adopted Real-to-binary with ensemble teacher and continued training

```bash
python main.py --model resnet18_e1 --weight-quantization dorefa --activation-method round --clip-threshold 1.3 --fp-downsample-sc --pool-downsample-sc --dataset cifar100 --epochs 150 --batch-size 128 --log test_run5 --log-interval 100 --gpus 0 --lr 0.01 --lr-mode cosine --bit-widths 32 8 4 2 1 --target-bits 1 --optimizer radam --result_path ../test_run5/ --ensemble-teacher --resume first_stage.params --teacher-params first_stage.params
```

--ensemble-teacher: Specifies that an ensemble is used as a teacher

--resume: Load parameters from previous training to resume the model

--teacher-params: Load parameters of the teacher ensemble



## Citations:

\[1\] Haojin Yang, Martin Fritzsche, Christian Bartz, and Christoph Meinel. “BMXNet: An Open-Source Binary Neural Network Implementation Based on MXNet”. In: CoRR abs/1705.09864 (2017). arXiv: 1705.09864. url: http://arxiv.org/abs/1705.09864.

\[2\] Joseph Bethge, bmxnet-examples, (2021), GitLab repository, https://gitlab.hpi.de/hpi-xnor/bmxnet-projects/bmxnet-examples
