# Implementation of the CAMD Model

This repo contains the implementation of the CAMD model described in _Commonality Augmented Disentanglement for Multimodal Crowdfunding Success Prediction_.

> Commonality Augmented Disentanglement for Multimodal Crowdfunding Success Prediction  
> Jiayang Li, Xovee Xu, Yili Li, Ting Zhong, Kunpeng Zhang, and Fan Zhou  
> IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), Apr 6-11, 2025, Hyderabad, India  


## Usage

### Datasets
Datasets (GoFundMe and Indiegogo) can be downloaded from [here](https://drive.google.com/drive/folders/1r9qDzzINHkUvH3-urOJyujPJHWbq1IdM), then
put the downloaded datasets (`gofundme_data.pkl` and `indiegogo_data.pkl`) into `./dataset` directory and split it into train, valid and test sets through `python ./dataset/dataset_split.py`.

### Experiment Running

#### Environment

```shell
conda create --name camd python=3.9
conda activate camd
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install pandas scikit-learn easydict tqdm pynvml
```

#### Training

First, the necessary parameters can be set in the `./config/config.json`. Then, you can select training dataset in `train.py`.
Training the model as below:
```
python train.py
```
By default, the trained model will be saved in `./pt` directory.

#### Test

Testing the trained model as below:
```
python test.py
```
Please set the path of trained model in `test.py` before testing.

## Citation

    @inproceedings{li2025commonality,
      author = {Jiayang Li and Xovee Xu and Yili Li and Ting Zhong and Kunpeng Zhang and Fan Zhou},
      title = {Commonality Augmented Disentanglement for Multimodal Crowdfunding Success Prediction},
      booktitle = {IEEE ICASSP},
      year = {2025}
    }
