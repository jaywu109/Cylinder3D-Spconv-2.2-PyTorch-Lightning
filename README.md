## Cylinder3D Spconv-2.2 PyTorch Lightning

### Installation
1. Create conda environment with following command:
```
cd Cylinder3D-Spconv-2.2-PyTorch-Lightning
conda env create --name envname --file=environment.yml
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
### Training
1. modify the `config/semantickitti_pl.yaml`, including data path, log path (tensorboard log and model checkpoint), batch size, ddp training GPU device number
2. train the network by running `CUDA_VISIBLE_DEVICES=0,1 python train_cylinder_asym_pl.py`
### Evaluation
1. evaluate the trained network by running `CUDA_VISIBLE_DEVICES=0,1 python train_cylinder_asym_pl_eval.py` (can eval on **single or multiple** GPUs)

### Other Training Details
- 40 epochs
- 0.000707 base LR with sqrt_k scaling rule (equals to original 0.001 at batchsize = 2, equals 0.00489 at batchsize = 24)
- AdamW with Weight Decay 0.001
- CosineDecay Schedule

---
### Simple Benchmark


| GPU Type | Single GPU | 4 GPUs | 
| -------------- |:---------------------:|---------------------:|
| A100 | 60 mins/epoch    | 25 mins/epoch    | 

## Reference

If you find our work useful in your research, please consider citing the original paper [paper](https://arxiv.org/pdf/2011.10033) and the repo:
```
@article{zhu2020cylindrical,
  title={Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation},
  author={Zhu, Xinge and Zhou, Hui and Wang, Tai and Hong, Fangzhou and Ma, Yuexin and Li, Wei and Li, Hongsheng and Lin, Dahua},
  journal={arXiv preprint arXiv:2011.10033},
  year={2020}
}
@software{githubGitHubJaywu109Cylinder3DSpconv22PyTorchLightning,
  author = {Wu, Dai-Jie},
  title = {{G}it{H}ub - jaywu109/{C}ylinder3{D}-{S}pconv-2.2-{P}y{T}orch-{L}ightning --- github.com},
  url = {https://github.com/jaywu109/Cylinder3D-Spconv-2.2-PyTorch-Lightning},
  year = {2023}
}
```

## Acknowledgments
We thanks for the opensource codebases, [Cylinder3D
](https://github.com/xinge008/Cylinder3D) and [Cylinder3D-updated-CUDA
](https://github.com/L-Reichardt/Cylinder3D-updated-CUDA)
