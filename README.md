## False Negative Reduction in Video Instance Segmentation using Uncertainty Estimates

Instance segmentation of images is an important tool for automated scene understanding. Neural networks are usually trained to optimize their overall performance in terms of accuracy. Meanwhile, in applications such as automated driving, an overlooked pedestrian seems more harmful than a falsely detected one. In this work, we present a false negative detection method for image sequences based on inconsistencies in time series of tracked instances given the availability of image sequences in online applications. As the number of instances can be greatly increased by this algorithm, we apply a false positive pruning using uncertainty estimates aggregated over instances. To this end, instance-wise metrics are constructed which characterize uncertainty and geometry of a given instance or are predicated on depth estimation. The proposed method serves as a post-processing step applicable to any neural network that can also be trained on single frames only. In our tests, we obtain an improved trade-off between false negative and false positive instances by our fused detection approach in comparison to the use of an ordinary score value provided by the instance segmentation network during inference.

For further reading, please refer to https://arxiv.org/abs/2106.14474.

## Preparation:
We assume that the user is already using a neural network for instance segmentation, a corresponding dataset and a multiple object tracking algorithm (see for example https://github.com/kmaag/Temporal-Uncertainty-Estimates). For each image from the instance segmentation dataset, Temporal-False-Negative-Reduction requires the following data where each video is in its own folder:

- the input image (height, width) as png
- the ground truth (height, width) as png
- a three-dimensional numpy array (num instances, height, width) that contains the predicted instance mask denoted by corresponding tracking ID for the current image 
- a one-dimensional numpy array (num instances) that contains the score values computed for the current image
- the depth estimation (height, width) as png


Before running Temporal-False-Negative-Reduction, please edit all necessary paths stored in "global_defs.py". The code is CPU based and parts of of the code trivially parallize over the number of input images, adjust "NUM_CORES" in "global_defs.py" to make use of this. Also, in the same file, select the tasks to be executed by setting the corresponding boolean variable (True/False).

## Run Code:
```sh
./run.sh
```

The results in https://arxiv.org/abs/2106.14474 have been obtained from two instance segmentation networks, the Mask R-CNN (https://github.com/matterport/Mask_RCNN) and the YOLACT network (https://github.com/dbolya/yolact), in combination with the KITTI and the MOT dataset (https://www.vision.rwth-aachen.de/page/mots).

## Author:
Kira Maag (University of Wuppertal)




 
