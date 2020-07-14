# RetinaFace-tf2
RetinaFace (RetinaFace: Single-stage Dense Face Localisation in the Wild, published in 2019) reimplemented in Tensorflow 2.0, with pretrained weights available

Original paper -> [arXiv](https://arxiv.org/pdf/1905.00641.pdf)  
Original Mxnet implementation -> [Insightface](https://github.com/deepinsight/insightface/tree/master/RetinaFace)

example output : 
![testing on a random internet selfie](retinaface_tf2_output.jpg)
*****
## Installation
To install dependencies, if you have a GPU, run : 
```
pip install -r requirements-gpu.txt
```
If not, run : 
```
pip install -r requirements.txt
```
Then build the rcnn module by running : 
```
make
```
*****
## Test on an image
Download pretrained weights on [Dropbox](https://www.dropbox.com/s/g4f2lap9cyrdfw5/retinafaceweights.npy?dl=0) and save them in the data folder  
Run  :
```angular2
python test.py --weights_path="./data/retinafaceweights.npy" --sample_img="./sample-images/WC_FR.jpeg"
```
*****
## Benchmark   
mAP result values on the WIDERFACE validation dataset:  

| Model  | Easy  | Medium  | Hard  |
|---|---|---|---|
|Original Mxnet implementation   | 96.5  | 95.6 | 90.4 |
| Ours | 95.6  | 94.6  | 88.5  |
*****
## Evaluate on WIDERFACE
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
*****
## Aknowledgements
This work is largely based on the original implementation by the amazing [insightface](https://github.com/deepinsight/insightface) team  
Evaluation on widerface done with the [Widerface-Evaluation](https://github.com/wondervictor/WiderFace-Evaluation) repo  
If you use this repo, please reference the original work :

```  
@inproceedings{Deng2020CVPR,
title = {RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild},
author = {Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irene and Zafeiriou, Stefanos},
booktitle = {CVPR},
year = {2020}
}
```