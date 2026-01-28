# About this project
This project searches to classify images of imune cells in leucograms through the use of deep learning. This work is the product of a colaboration between the Computer Grapics Research Group and the veterenary medicine department of Federal University of Juiz de Fora (UFJF).

Our objective is to build a neural network that can classify with considerable accuracy the most common types of imune cells found in leucograms to the point where it can be used in an aplication that facilitates and accelerates the work of veterinarians.

This is a work in progress, so you may find errors in the script or you may have to modify the code for the needs of your specific dataset. The project is currently testing out the waters, trying some pre-processing and identifying what the network is currently having a hard time with.

# How to run
This project is currently in a stage of trial and error and is not complete nor is it generalized by any means, meaning that running problems may ocurr when trying to run it on unexpected datasets. If you wish to run the experiments on a dataset divided in train and test sets, follow the instructions in the section "Training with train/val/test". If your dataset is divided in folds, follow the section "Training with cross validation".

## Training with train/val/test
### Structuring the dataset
The dataset should be structured following the folder structure, in which images of a certain class are contained in the folder of their class. You should divide your data in the 3 sets (train, test and validation) and then divide each set in the classes the neural network will have to identify.
### Running the script
As the name implies, you should focus your attention on the "train_no_folds" script.
- To train the network on a dataset, execute the following command on the root of this project:
```python3 train_no_folds -e <number-of-epochs>```
- If you wish for the script to display the dataset structure, use the tag `-c` or `--check-dataset`.

## Training with cross validation
### Structuring the dataset
The dataset should be structured following the folder structure, in which images of a certain class are contained in the folder of their class. You should divide your data in the folds (fold_1, fold_2, fold_3, etc) and then divide each fold in the classes the neural network will have to identify.
### Running the script
As the name implies, you should focus your attention on the "train_folds" script.
- To train the network on a dataset, execute the following command on the root of this project:
```python3 train_folds -e <number-of-epochs>```
- If you wish for the script to display statistics about the incorrect predictions, use the tag `-s` or `--statistics`.
- If you wish for the script to do a test on your data with certain weights instead of training, use the tag `-q` or `--quick-test`.
- If you wish to specify the prefix of the file of the weights you wish to use for the test, use the tag `-w <weights-file-prefix>` or `--weight-prefix <weights-file-prefix>`.
- If you wish to specify the version name the script should use as sufix for the results of the test, use the tag `-v <version-name>` or `--test-version <version-name>`.