## Self supervised learning for fine grained classification in case of bird species identification

The project explores self supervised learning for fine grained bird species classification, with an aim to remove the dependence on ImageNet pre-trained model weights for learning the CNN pipeline weights. We used jigsaw puzzles solving as a pre-text training task on birds images collected from multiple online datasets, and explored if weights from jigsaw based training can be extended for fine grained classification.

### Why jigsaw puzzles
Solving jigsaw puzzles can be used to teach a system that an object is made up of parts and what these parts are. Given that conventional ML and deep learning solutions relied on part based annotations for learning a reliable classifier, we believe that jigsaw puzzles solving task could hence eliminate that dependence, as it would be able to learn during pretext training how different parts are different from one another and how they relate to each other.

### Datasets
We use [CUB200](http://www.vision.caltech.edu/visipedia/CUB-200.html), [INAT19](https://www.kaggle.com/c/inaturalist-2019-fgvc6/overview)
birds subset and [NABirds](https://dl.allaboutbirds.org/nabirds) dataset for pre-text training or the jigsaw puzzles
solviing task. While we use just the CUB200 for the downstream task of fine grained bird species classification.

### Training:

##### Background:
We explore two different dataset configurations for training jigsaw puzzles solver.
1. JS_D1: Jigsaw puzzles solving task dataset consisting of 20 permutations of each image in CUB200 training set in each epoch.
2. JS_D2: Jigsaw puzzles solving task dataset consisting of CUB200 training images, INAT 2019 birds subset, and NABirds images.

For both pre-text training and downstream task training we use ResNet18 architecture

##### Training on the jigsaw puzzles task
1. Run script `get_top_x_permutations.py` to build the permutations set of image patches that is used for training the 
jigsaw puzzle solving task. For this project we limited to 200 permutations of image patches or 200 different puzzle 
possibilities for an image.

2. Run `python train_test_jigsaw_solver.py --batch-size 128 --epochs 50 --lr 1e-2 --experiment-name <exp_name>
   --dataset-config <js_d1/js_d2>` for  training ResNet18 on jigsaw puzzles solving task on the dataset config of choice.

##### Training on the downstream classification task
1. For fine tuning block4 layers and softmax layer of ResNet18 on classification task, use command:
`python train_test_script.py --batch-size 128 --epochs 50 --lr 1e-2 --jigsaw-task-weights <jiigsaw_weights_file_path> --experiment-name e1_last_a --train-ssl-block-4-ft True`
2. For fine tuning block3  and block layers and softmax layer of ResNet18 on classification task, use command:
`python train_test_script.py --batch-size 128 --epochs 50 --lr 1e-2 --jigsaw-task-weights <jiigsaw_weights_file_path> --experiment-name e1_last_b --train-ssl-block-3-ft True`
3. For fine tuning complete model on classification task:
`python train_test_script.py --batch-size 128 --epochs 50 --lr 1e-2 --jigsaw-task-weights <jiigsaw_weights_file_path> --experiment-name e1_full --train-ssl-full-ft True`

##### Results

**Results on Jigsaw solving task**

Dataset config used for jigsaw solver | Train Accuracy | Validation Accuracy
--- | --- | ---
JS_D1 | 65.82% | 55.9%
JS_D2 | 77.71% | 71.45%


**Results on Downstream (classification) task**

Dataset config used for jigsaw solver | Layers fine tuned (in downstream task) | Train Accuracy | Validation Accuracy
---| --- | --- | ---
NA | Full model trained from scratch | 100% | 47%
NA | Full model trained from imagenet weights | 100% |  74.17%
JS_D1 | Block 4 and softmax | 63.57% | 19.17%
JS_D1 | Block 3, 4 and softmax | 93.99% | 37.17%
JS_D1 | Full model fine tuned | 99% | 37.16%
JS_D2 | Block 4 and softmax | 96.73% | 31%
JS_D2 | Block 3, 4 and softmax | 99.34% | 43%
JS_D2 | Full model fine tuned | 99.55% | 42.66%

### Files Index (If otherwise reqd)

1. **dataset_helpers.py**: Code for different data transforms used in model training. Plus helper methods to get image
   file paths.
2. **get_dataset.py**: Code for custom dataset objects, required for training jigsaw solver and downstream 
classification task.
3. **get_top_x_permutations.py**: Script to build the permutations set of image patches that is used for training the 
jigsaw  puzzle solving  task.
4. **resnet_file.py**: Script to define ResNet architecture, such that it is able to take up jigsaw puzzles task along 
with classification task.
5. **train_test_helper.py**: Contains helper classes for jigsaw puzzle solver training-testing and classification task
training testing requirements.
6. **train_test_jigsaw_solver.py**: Script for training and validation on jigsaw solving task.
7. **train_test_script.py** Script for training and validation on downstream classification task.
8. **evaluate_classification.py**: Code to evaluate the performance of final model on test set (for classification task).
