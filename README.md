## Self supervised learning for fine grained classification in case of bird species identification

### Files Index

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


### Usage
1. For training ResNet18 on Jigsaw puzzles solving task use command:
`python train_test_jigsaw_solver.py --batch-size 128 --epochs 50 --lr 1e-2 --experiment-name e1_js --dataset-config <js_d1/js_d2>`
2. For fine tuning block4 layers and softmax layer of ResNet18 on classification task, use command:
`python train_test_script.py --batch-size 128 --epochs 50 --lr 1e-2 --jigsaw-task-weights <jiigsaw_weights_file_path> --experiment-name e1_last_a --train-ssl-block-4-ft True`
3. For fine tuning block3  and block layers and softmax layer of ResNet18 on classification task, use command:
`python train_test_script.py --batch-size 128 --epochs 50 --lr 1e-2 --jigsaw-task-weights <jiigsaw_weights_file_path> --experiment-name e1_last_b --train-ssl-block-3-ft True`
4. For fine tuning complete model on classification task:
`python train_test_script.py --batch-size 128 --epochs 50 --lr 1e-2 --jigsaw-task-weights <jiigsaw_weights_file_path> --experiment-name e1_full --train-ssl-full-ft True`