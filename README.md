# FaceMask-Detection_Damo-Yolo

This repo includes a demo for building a face mask detector using Damo-Yolo model. We use object detection method to detect whether people are wearing masks or not in image. 

![image](https://github.com/yichung-chen/FaceMask-Detection_Damo-Yolo/blob/main/results/maksssksksss12.png)

## Datasets

The model was trained on [Face-Mask](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) dataset which contains 853 images belonging to the 3 classes, as well as their bounding boxes in the PASCAL VOC format.

The classes are defined as follows:

* With mask
* Without mask
* Mask worn incorrectly

## DatasetsSetup
* Clone this repo and we have included Damo-YOLO repo
```
git clone https://github.com/yichung-chen/FaceMask-Detection_Damo-Yolo.git
cd customized
pip install -r requirements.txt
```
## Prepare Datasets
Download [Face-Mask](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) dataset from Kaggle and copy it into /customized/dataset/dataset/ folder.
Execute the following command to split it into train、valid and test sets and convert the data into the COCO format. The split ratio was set to 80/10%/10%.
```
copy dataset to /customized/dataset/dataset/
cd customized/model/

# Spilt dataset
python datasets_split.py \
        --datadir='/customized/dataset/dataset/' \
        --split=0.2 \
        --train_output='/customized/dataset/train/' \
        --val_output='/customized/dataset/val/
        --test_output='/customized/dataset/test/' \
        --image_ext= 'your image type(jpg or png)'

# Convert to COCO format
python voc2coco.py

# Choise which model you want, Default is damoyolo_tinynasL20_T
vim customized/hyperparameters/hyperparameters.yaml

```

## Training
```
cd customized/models/
python training.py

or use the original author code

cd customized/damo-yolo
python -m torch.distributed.launch --nproc_per_node=8 train.py -f configs/damoyolo_tinynasL20_T.py

