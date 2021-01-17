# 3D Shape Reconstruction from Free-Hand Sketches
[[Project]](http://pwang.pw/3dsketch.html) [[Paper]](https://arxiv.org/abs/2006.09694)   

## Overview
This is authors' re-implementation of the paper described in:  
"[3D Shape Reconstruction from Free-Hand Sketches](https://arxiv.org/abs/2006.09694)"   
[Jiayun Wang](http://pwang.pw/),&nbsp; Jierui Lin,&nbsp; [Qian Yu](https://yuqian1023.github.io//)&nbsp; Runtao Liu,&nbsp; [Yubei Chen](https://redwood.berkeley.edu/people/yubei-chen/),&nbsp;   [Stella X. Yu](https://www1.icsi.berkeley.edu/~stellayu/)&nbsp; (UC Berkeley/ICSI/BUAA)&nbsp; in arXiv.

## Requirements
* [Tensorflow](https://www.tensorflow.org/) (version >= 1.12.0)

## Overall architecture
```
data
  |--Data Folder #please update corresponding path in depthestimate//BatchFetcher.py
  |--depthestimate # 3D reconstruction module
    |--BatchFetcher.py
    |--train_nn_trans.py
    |--visualizeptexample.v.py
    |--...
  |--sketchstd # sketch standarization module
```

## Training the network

### Download the dataset

We use the [ShapeNet](https://www.shapenet.org/) dataset in our experiments, which are available below:

- ShapeNet rendering images: http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
- ShapeNet voxelized models: http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz (optional, for comparison with voxelized output only)

Make sure you update the data path in the BatchFetcher.py.

### Generate synthetic sketches from rendered images
We use a previous work [Unsupervised Sketch to Photo Synthesis](https://arxiv.org/abs/1909.08313) for generating synthetic sketches. Please refer to [their code](https://github.com/rt219/Unpaired-Sketch-to-Photo-Translation) to generate data for training.

You can use [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) as an alternative to generate synthetic sketches from rendered images. This may lead to a worse result.

### Compiling CUDA code

You will need to first edit makefile to match your nvcc, cudalib, and tensorflow path.

Then compile CUDA code
```
$ make
```


### Training the sketch standarization module

We use [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to train the sketch standarization module. Our code will be released soon.

### Training the reconstruction network
Update path to STANDARIZED sketch and point cloud dataset in depthestimate//BatchFetcher.py first.

```	
$ python depthestimate/train_nn_trans.py data=. dump=. train
```


### Predict with trained network
```
$ python depthestimate/train_nn_trans.py data=. dump=. predict
```

### Visualualize dumpped prediction (press space to view the next one)
```
$ python depthestimate/visualizeptexample.v.py <path>/train_nn.v.pkl
example: $ python depthestimate/visualizeptexample.v.py dump/train_nn.v.pkl
```
	

## Note
- The repo is adapted from [PointSetGeneration](https://github.com/fanhqme/PointSetGeneration).
- The current code does not support multi-GPU settings.

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
