# 3D Shape Reconstruction from Free-Hand Sketches
[[Project]](http://pwang.pw/3dsketch.html) [[Paper]](https://arxiv.org/abs/2006.09694)   

## Overview
This is authors' re-implementation of the paper described in:  
"[3D Shape Reconstruction from Free-Hand Sketches](https://arxiv.org/abs/2006.09694)"   
[Jiayun Wang](http://pwang.pw/),&nbsp; Jierui Lin,&nbsp; [Qian Yu](https://yuqian1023.github.io//)&nbsp; Runtao Liu,&nbsp; [Yubei Chen](https://redwood.berkeley.edu/people/yubei-chen/),&nbsp;   [Stella X. Yu](https://www1.icsi.berkeley.edu/~stellayu/)&nbsp; (UC Berkeley/ICSI/BUAA)&nbsp; in arXiv.

Note that this repo is still under construction and will be ready soon.

## Requirements
* [Tensorflow](https://www.tensorflow.org/) (version >= 1.12.0)

## Overall architecture

## Training the network

### Download the dataset

### Generate synthetic sketches from rendered images
We use a previous work [Unsupervised Sketch to Photo Synthesis](https://arxiv.org/abs/1909.08313) for generating synthetic sketches. Please refer to [their code](#) to generate data for training.

### Training the sketch standarization module

### Training the reconstruction network

## Note
The current code does not support multi-GPU settings.

## Q \& A
Please raise issues if you encounter any problem.

## License and Citation
The use of this software is released under [BSD-3](LICENSE).
```
@article{wang20203d,
  title={3D Shape Reconstruction from Free-Hand Sketches},
  author={Wang, Jiayun and Lin, Jierui and Yu, Qian and Liu, Runtao and Chen, Yubei and Yu, Stella X},
  journal={arXiv preprint arXiv:2006.09694},
  year={2020}
}
```
