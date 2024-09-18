# Commonality Augmented Disentanglement for Multimodal Crowdfunding Success Prediction

This is the official repository for our paper _Commonality Augmented Disentanglement for Multimodal Crowdfunding Success Prediction_.

## Usage

### Datasets
Datasets (GoFundMe and Indiegogo) can be downloaded from [here](https://drive.google.com/drive/folders/1r9qDzzINHkUvH3-urOJyujPJHWbq1IdM).
You can put the downloaded datasets into `./dataset` directory and split it into train, valid and test sets through `./dataset/dataset_split.py`.

### Experiment Running
- Training

First, the necessary parameters can be set in the `./config/config.json`. Then, you can select training dataset in `train.py`.
Training the model as below:
```
python train.py
```
By default, the trained model will be saved in `./pt` directory.

- Testing

Testing the trained model as below:
```
python test.py
```
Please set the path of trained model in `test.py`.