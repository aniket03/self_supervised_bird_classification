## Self supervised learning for fine grained classification in case of bird species identification

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
