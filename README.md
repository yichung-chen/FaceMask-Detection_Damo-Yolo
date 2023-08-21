# FaceMask-Detection_Damo-Yolo

This repo includes a demo for building a face mask detector using Damo-Yolo model. We use object detection method to detect whether people are wearing masks or not in image. 

![image](https://github.com/yichung-chen/FaceMask-Detection_Damo-Yolo/blob/main/results/maksssksksss12.png)

## Datasets

The model was trained on [Face-Mask](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) dataset which contains 853 images belonging to the 3 classes, as well as their bounding boxes in the PASCAL VOC format.

The classes are defined as follows:

* With mask
* Without mask
* Mask worn incorrectly

```

git clone https://github.com/yichung-chen/FaceMask-Detection_Damo-Yolo.git
cd customized
pip install -r requirements.txt
```
