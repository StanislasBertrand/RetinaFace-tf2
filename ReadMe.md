#[WIP] RetinaFace-tf2
This repository contains a tensorflow2 reimplementation of the model presented in the [RetinaFace paper](https://arxiv.org/pdf/1905.00641.pdf)

### Credit
This work is largely based on the original implementation by the amazing [insightface](https://github.com/deepinsight/insightface) team

### Caution
This repository is a work in progress. Precision seems to be OK for the network with variable size input, however it runs only on cpu at the moment and is quite slow

### Test
Download pretrained weights on [Dropbox](https://www.dropbox.com/s/g4f2lap9cyrdfw5/retinafaceweights.npy?dl=0)  
Put weights in data/ folder  
Run test.py

### TODO
* ~~working, pretrained tf2 implementation of retinanet !~~
* manage to make variable input size network run on gpu
* manage to make nms run on gpu
* accelerate inference
* test on WIDERFACE
