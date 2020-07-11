# [WIP] RetinaFace-tf2
This repository contains a tensorflow2 reimplementation of the model presented in the [RetinaFace paper](https://arxiv.org/pdf/1905.00641.pdf)

![testing on a random internet selfie](retinaface_tf2_output.jpg)

### Installation
TODO
### Inference
Download pretrained weights on [Dropbox](https://www.dropbox.com/s/g4f2lap9cyrdfw5/retinafaceweights.npy?dl=0) and save them in the data folder  
Run  :
```angular2
python test.py
```

### Evaluate on WIDERFACE
In order to verify the models accuracy on the WiderFace dataset:
* Run the model on the dataset and generate text files as results
```angular2
python eval_widerface --weights_path="data/retinafaceweights.npy" --widerface_data_dir = "/data/WIDER_test/images" --save_folder="./WiderFace-Evaluation/results/"
```
* Evaluate the results
```angular2
cd ./WiderFace-Evaluation
python setup.py build_ext --inplace
python evaluation.py -p ./results_val/ -g ./ground_truth/
```
### Caution
This repository is a work in progress. Precision seems to be OK for the network with variable size input, however it runs only on cpu at the moment and is quite slow



### TODO
* ~~working, pretrained tf2 implementation of retinanet !~~
* ~~Make variable input size network run on gpu~~
* ~~Make nms run on gpu~~
* Calculate accuracy on WIDERFACE
* accelerate inference

### Aknowledgements
This work is largely based on the original implementation by the amazing [insightface](https://github.com/deepinsight/insightface) team  
Evaluation on widerface done with the [Widerface-Evaluation](https://github.com/wondervictor/WiderFace-Evaluation) repo