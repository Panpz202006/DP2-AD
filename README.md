# Pytorch implementation of AF-CLIP-DP

Pytorch implementation of Paper "AF-CLIP: Zero-Shot Anomaly Detection via Anomaly-Focused CLIP Adaptation"

## Installation

```bash
apt-get update && apt-get install -y libgl1 libglib2.0-0 && \
python -m pip install torchaudio tqdm einops scikit-learn ftfy regex pandas scikit-image opencv-python matplotlib seaborn \
  -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir && \
python -m pip install --force-reinstall "setuptools<81" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir
```

![](./pic/model.png)


please check you dataset dir is like below:
```
├── Br35H
├── Br35H
│   ├── no
│   └── yes
├── BrainMRI
│   ├── no
│   └── yes
├── btad
│   ├── 01
│   │   ├── ground_truth
│   │   │   └── ko
│   │   ├── test
│   │   │   ├── ko
│   │   │   └── ok
│   │   └── train
│   │       └── ok
│   ├── ...
├── CVC-ClinicDB
│   ├── images
│   └── masks
├── CVC-ColonDB
│   ├── images
│   └── masks
├── DAGM_KaggleUpload
│   ├── Class1
│   │   ├── Test
│   │   │   └── Label
│   │   └── Train
│   │       └── Label
│   ├── ...
├── DTD-Synthetic
│   ├── Blotchy_099
│   │   ├── ground_truth
│   │   │   └── bad
│   │   ├── test
│   │   │   ├── bad
│   │   │   └── good
│   │   └── train
│   │       └── good
│   ├── ...
├── ISIC2016
│   ├── ISBI2016_ISIC_Part1_Test_Data
│   └── ISBI2016_ISIC_Part1_Test_GroundTruth
├── Kvasir
│   ├── images
│   └── masks
├── mvtec
│   ├── bottle
│   │   ├── ground_truth
│   │   │   ├── broken_large
│   │   │   ├── broken_small
│   │   │   └── contamination
│   │   ├── test
│   │   │   ├── broken_large
│   │   │   ├── broken_small
│   │   │   ├── contamination
│   │   │   └── good
│   │   └── train
│   │       └── good
│   ├── ...
├── visa
│   ├── candle
│   │   └── Data
│   │       ├── Images
│   │       │   ├── Anomaly
│   │       │   └── Normal
│   │       └── Masks
│   │           └── Anomaly
│   ├── ...
│   ├── split_csv
```

Then change the data path `data_dir` in  `train.sh` or `test.sh` to train or test.

To train the zero-shot model, you can run
```
sh ./train.sh
```


To test model, you can run
```
sh ./test.sh
```
