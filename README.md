## Cylinder3D Spconv-2.2 PyTorch Lightning

## Installation
1. Create conda environment with following command:
```
conda env create --name envname --file=environments.yml
```
2. Install PyTorch
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```
3. Install torch-scatter
```
pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
```
4. Install other dependencies
```
pip install -r requirements.txt
```
## Training
1. modify the `config/semantickitti_pl.yaml`, including data path, log path (tensorboard log and model checkpoint), batch size, ddp training GPU device number
2. train the network by running `CUDA_VISIBLE_DEVICES=0,1 python train_cylinder_asym_pl.py`

### Other Training Details
- 40 epochs
- 0.000707 base LR with sqrt_k scaling rule (equals to original 0.001 at batchsize = 2, equals 0.00489 at batchsize = 24)
- AdamW with Weight Decay 0.001
- CosineDecay Schedule

## Simple Benchmark


| GPU Type | Single GPU | 4 GPUs | 
| -------------- |:---------------------:|---------------------:|
| A100 | 60 mins/epoch    | 25 mins/epoch    | 
