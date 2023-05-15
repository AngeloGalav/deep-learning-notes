In object detection, we are returning a _bounding box_ (the smallest possible box) containing the object. 

In some ways it is similar to segmentation, but while segmentation can be understood as an image to image process and thus the _loss function_ can be understood easily (it is the distance between the real segmentation and the segmentation guessed by the network), 
the same cannot be said about the object detection approach.

#### Pros & cons of this approach
- __pro__: no need to strive about borders 
- __cons__: 
	- multiple outputs of unknown number (the result is not of a fixed dimension) 
	- difficult to train end-to-end 
	- no evident loss function

How can we compute a loss function for this type of approach?

#### Intersection over Union - Quality indicator for Bounding Boxes
Typically, the quality of each individual bounding box is evaluated vs. the corresponding ground truth using _Intersection over Union_:
$$
IoU(A,B) = \dfrac{|A \cap B|}{|A \cup B|}
$$
![[intersection_over_union.png]]
Essentially, we measure how good is a prediction by checking how much the 2 boxes stack up/ how close they are. 

Then, these bounding boxes be summed up for all detections, and suitably combined with classification errors.

## Deep Object Detection Approaches
There are 2 main approaches: 
- __Region Proposals methods__ (R-CNN, Fast R-CNN, Faster R-CNN). Region Proposals are usually extracted via _Selective Search_ algorithms, aimed to identify possible locations of interest. These algorithms _typically exploit the texture and structure of the image_, and are object independent. 
	- [Detectron2](https://ai.facebook.com/tools/detectron2/) is a pytorch library developed by Facebook AI Research (FAIR) to support rapid implementation and evaluation of novel computer vision research. 
- __Single shots methods__ (Yolo, SSD, Retina-net, FPN). We shall mostly focus on these _really fast_ techniques.
	- In this kind of methods, we are trying to do everything on a single pass, meaning that we're trying to identify the boxes, and then to classify the content of the box. 
	- Especially suited for real-time applications. 

## YOLO’s architecture
![[yolo.png]]
Yolo is a _Fully Convolutional Network_. The input is _progressively downsampled_ by a factor $2^5 = 32$. This is done in order to _increase the receptive field of our neurons_ (each neuron at the end musts see a sufficiently large portion of the image).
For instance, an input image of dimension 416x416 would be reduced to a grid of neurons of dimension 13x13, which is _the feature map_.

#### How do we train this network?
Detection of an object may concern all neurons inside the bounding box. So, who’s in charge for detection (who's supposed to recognize this objects, and thus should be trained)? 

In YOLO, a _single neuron_ is responsible for _detection_: the ==one whose grid-cell contains the _center_ of the bounding box==. This neuron makes a _finite number of predictions_ (e.g. 3).
- We don't care what all the other neurons predict, in fact they get _masked_ in the loss function.  

#### Shape of each box
We have 13x13 neurons in the feature map. 
- Depth-wise, we have $(B \times (5 + C))$ _entries_, where $B$ represents the _number of bounding boxes_ each cell can predict (say, 3), and $C$ is the number of _different object categories_. 
- Each bounding box has $5 + C$ attributes, which describe the _center coordinates_ (2), the _dimensions_ (2), the _objectness score_ (1) and $C$ class confidences (1 for each prediction, usually is 3).
![[yolo_feature_map.png]]

#### Anchor Boxes
Trying to directly predict width and the height of the bounding box leads to _unstable gradients_ during _training_.
Most of the modern object detectors predict log-space affine _transforms_ for _pre-defined default bounding boxes_ called __anchors__. Then, these transforms are applied to the anchor boxes to obtain the prediction.
YOLO v3 has three anchors, which result in prediction of three bounding boxes per cell. The bounding box responsible for detecting the object is one whose anchor has the highest $IoU$ with the _ground truth box_.

[to add: slides that he skipped...]

## YOLO's Loss function
The loss consists of two parts, the __localization loss__ for _bounding box offset prediction_ and the __classification loss__ for _conditional class probabilities_. 
Since YOLO is a single pass method, the final loss function should compute both of this parts in a single computation. 
As usual, we shall use $v$ to denote a true value, and $\hat v$ to denote the corresponding predicted one. 

#### Localization loss
The localization loss is:
![[localization_loss.png]]
where $i$ ranges over cells, and $j$ over bounding boxes.
- $1^{obj}_{ij}$ is a _delta function_ indicating whether the _$j$-th bounding box of the cell $i$ is responsible for the object prediction_. 
	- Essentially, it is a mapping function containing 0 everywhere except for the neuron that is supposed to do the localization. This essentially masks the predictions of the other neurons. 

#### Classification loss
The classification loss is the sum of two components, relative to the _objectness confidence_ and the _actual classification_:
![[classification_loss.png]]
$λ_{noobj}$ is a configurable parameter meant to down-weigth the loss contributed by “background” cells containing no objects. This is important because they are a large majority.

#### Final result
The whole loss is:
![[final_loss.png]]
$λ_{coord}$ is an additional parameter, balancing the contribution between $L_{loc}$ and $L_{cls}$. 
In YOLO, $λ_{coord} = 5$ and $λ_{noobj} = 0.5$.

## Multi scale processing
Here's an overview of image processing techniques for object detection throughout history. 
![[image_pyramid1.png]]
- The older approach to object detection from the 2010s used a __featurized image pyramid__. With this approach, features are computed on each of the image scales independently, which is _slow_.
	- Essentially, images were scaled and rescaled multiple times in order to find the important features of the image.

- First systems for fast object detection (like YOLO v1) opted to use only higher level features at the _smallest scale_ (__single feature map__). This usually _compromises detection of small objects_.
![[image_pyramid2.png]]
- An alternative (Single Shot Detector - SSD) is to reuse the _pyramidal feature hierarchy_ computed by a ConvNet as if it were a _featurized image pyramid_.
- Modern Systems (FPN, RetinaNet, YOLOv3) recombine features along a __backward pathway__. This is as fast as (b) and (c), but more accurate. 

In the figures, feature maps are indicated by blue outlines and thicker outlines denote semantically stronger features.

#### Featurized Image Pyramid
![[image_pyramid3.png]]
- Bottom-up pathway is the normal feedforward computation. 
- _Top-down_ pathway goes in the inverse direction, adding coarse but _semantically stronger feature maps_ back into the previous pyramid levels of a larger size via lateral connections.
	- First, the higher-level features are _spatially upsampled_. 
	- The feature map coming from the Bottom-up pathway undergoes _channels reduction_ via a 1x1 conv layer. 
	- Finally, these two feature maps are _merged_ (by element-wise addition, or concatenation).

## Non Maximum Suppression
This is final phase of the YOLO algorithm. 
Essentially, we have to consider that we have a huge amount of predictions, since at each feature map, each neuron makes a prediction. 
YOLOv3 predicts feature maps at scales 13, 26 and 52.

For example, if we have a situation like this:
![[different_feature_maps.png]]
At the end, we have $((13×13)+(26×26)+(52×52)) \times 3 = 10647$ bounding boxes, each one of dimension $85$ (4 coordinates, 1 confidence, 80 class probabilities). 
How can we reduce this number to the few bounding boxes we expect?

These operations are done algorithmically, and they consist in 
- __Thresholding by Object Confidence__: first, we filter boxes based on their objectness score. Generally, boxes having scores _below a threshold are ignored_. 
- __Non Maximum Suppression__: NMS addresses the problem of _multiple detections of the same image_, corresponding to different anchors, adjacent cells in maps.

#### NMS outline
- Divide the bounding boxes $BB$ according to the predicted class $c$ (creating a list for each class). 
- Each list $BB_c$ is processed separately 
- Order $BB_c$ according to the object confidence. 
- Initialize `TruePredictions` to an empty list. 
- while $BB_c$ is not empy: 
	- pop the first element $p$ from $BB_c$ 
	- add $p$ to `TruePredictions` 
	- remove from $BB_c$ all elements with an $IoU$ with $p > th$ 
- return `TruePredictions`

Essentially, it ignores all overlapping BBs and keeps only the best ones. 