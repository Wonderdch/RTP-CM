# RTP-CM

This is the pytorch implementation of paper "Learning Robust Travel Preferences via Check-in Masking for Next POI
Recommendation"

![model](figure/model.png)

## Installation

```
pip install -r requirements.txt
```

## Valid Requirements

```
torch==2.0.1
numpy==1.24.3
pandas==2.0.2
Pillow==9.4.0
python-dateutil==2.8.2
pytz==2023.3
six==1.16.0
torchvision==0.15.2
typing_extensions==4.5.0
```

## Train

- Unzip `raw_data/raw_data.zip` to `raw_data/`. The three files are PHO, NYC and SIN check-in data.

- Run `data_preprocessor.py` to construct input data.

- Train and evaluate the model using python `main.py`.

- The training and evaluation results will be stored in `result` folder.

```
python main.py --name NYC_auto0.4
                --run-times 1
                --device cuda:0
                --dataset NYC
                --mask-strategy Auto --mask-proportion 0.4
                --area-proportion 0.2
                --embed-size 60 
                --transformer-layers 1 --transformer-heads 1
                --dropout 0.2
                --epochs 40
                --lr 1e-5
```
