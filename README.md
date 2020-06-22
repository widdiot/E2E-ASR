# INSTALL
```
inside E2E-ASR
git clone https://github.com/HawkAaron/warp-transducer
cd warp-transducer
mkdir build; cd build
cmake -D WITH_GPU=ON ..
make -j <nproc>
```

## inside warp_transducer
## set:
```
export CUDA_HOME=/usr/local/cuda/ 
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CFLAGS="-I$CUDA_HOME/include $CFLAGS"
export WARP_RNNT_PATH=/home/E2E-ASR/warp-transducer/build  ##change to your path
```
## then run 
* cd pytorch_binding
* python setup.py install

## then execute
* warp-transducer/pytorch_binding/test/test.py

## you should get:
```
CPU Tests passed! 
GPU Tests passed!
```
# check if import works: 
```
from warprnnt_pytorch import RNNTLoss
```
# update
* model_wp1.py
* train_inter_check1.py
* asr1.py
these files contain code for multi-gpu training and batch resume.

# Graves 2013 experiments
## File description
* model.py: rnnt joint model
* model2012.py: graves2012 model
* train_rnnt.py: rnnt training script
* train_ctc.py: ctc acoustic model training script
* eval.py: rnnt & ctc decode
* DataLoader.py: kaldi feature loader

## Run
* Extract feature
link kaldi timit example dirs (`local` `steps` `utils` )
excute `run.sh` to extract 40 dim fbank feature
run `feature_transform.sh` to get 123 dim feature as described in Graves2013

* Train CTC acoustic model
```
python train_ctc.py --lr 1e-3 --bi --dropout 0.5 --out exp/ctc_bi_lr1e-3 --schedule
```

* Train RNNT joint model
```
python train_rnnt.py --lr 4e-4 --bi --dropout 0.5 --out exp/rnnt_bi_lr4e-4 --schedule
```

* Decode 
```
python eval.py <path to best model> [--ctc] --bi
```

## Results

| Model | PER |
| --- | --- |
| CTC | 21.38 |
| RNN-T | 20.59 |

## Requirements
* Python 3.6
* PyTorch >= 0.4
* numpy 1.14
* [warp-transducer](https://github.com/HawkAaron/warp-transducer)

## Reference
* RNN Transducer (Graves 2012): [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711)
* RNNT joint (Graves 2013): [Speech Recognition with Deep Recurrent Neural Networks](https://arxiv.org/abs/1303.5778 )
* (PyTorch End-to-End Models for ASR)[https://github.com/awni/speech]
* (A Fast Sequence Transducer GPU Implementation with PyTorch Bindings)[https://github.com/HawkAaron/warp-transducer/tree/add_network_accelerate]
