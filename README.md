# Position-aware Location Regression Network for Temporal Video Grounding


This repository contains an official PyTorch implementation of Position-aware Location Regression Network (PLRN) for temporal video grounding, which is presented in the paper [Position-aware Location Regression Network for Temporal Video Grounding](https://arxiv.org/abs/2204.05499).


### Position-aware Location Regression Network (PLRN)

![model_overview](./imgs/framework-PLRN.png)
The overall architecture of the proposed network (PLRN). To understand comprehensive contexts with only one semantic phrase, PLRN exploits position-aware features of a query and a video. Specifically, PLRN first encodes both the video and query using positional information of words and video segments. Then, a semantic phrase feature is extracted from an encoded query with attention. The semantic phrase feature and encoded video are merged and made into a context-aware feature by reflecting local and global contexts. Finally, PLRN predicts start, end, center, and width values of a grounding boundary.

## Requirement

 - Ubuntu 16.04
 - Anaconda 3
 - Python 3.6
 - Cuda 10.1
 - Cudnn 7.6.5
 - PyTorch 1.1.0
 
## Preparing Data

We downloaded all data including annotations, video features (I3D for Charades-STA, C3D for ActivityNet Captions), pre-processed annotation information from [here](https://github.com/JonghwanMun/LGI4temporalgrounding).

## Training

```
conda activate plrn
cd PLRN
bash scripts/train_model.sh PLRN plrn charades 0 4 0
```

## Evaluation

```
conda activate plrn
cd PLRN
bash scripts/eval_model.sh PLRN plrn charades 0
```


## Acknowledgement
[Local-Global Video-Text Interactions for Temporal Grounding](https://github.com/JonghwanMun/LGI4temporalgrounding) was very helpful for our implementation.

## Citation
If you have found our implementation useful, please cite our paper:

	@inproceedings{kim2021position,
			title={Position-aware Location Regression Network for Temporal Video Grounding},
			author={Kim, Sunoh and Yun, Kimin and Choi, Jin Young},
			booktitle={2021 17th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS)},
			year={2021}
	}


