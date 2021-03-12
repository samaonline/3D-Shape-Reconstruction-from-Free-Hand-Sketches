import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
from pdb import set_trace as st
import numpy as np
from img_utils import (mls_affine_deformation, mls_affine_deformation_inv,
                       mls_similarity_deformation, mls_similarity_deformation_inv,
                       mls_rigid_deformation, mls_rigid_deformation_inv)
import imageio
from cv2 import warpAffine
import skimage
from scipy.ndimage import morphology
import itertools

def binary(im):
    return (im>125)*255

def affine_im(face, transpix=[10, 5], angle=10):
    im1 = skimage.transform.rotate(255-binary(face), angle)
    translation_matrix = np.float32([ [1, 0, transpix[0]], [0, 1, transpix[1]] ])
    im1 = warpAffine(im1, translation_matrix, (face.shape[0], face.shape[1]))
    
    im1 = im1/np.max(im1)
    return (1-im1)*255

def transform_im(image, NUM_MVP = 20):
    rowid, colid = np.where( image <100)
    id2cs = np.random.choice(range(len(rowid)), NUM_MVP, replace=False)
    
    p = np.stack((rowid, colid)).transpose()
    p = p[id2cs]
    q = p + (20*(np.random.rand(p.shape[0], p.shape[1]) -0.5 ) ).astype(int)
    
    transformed_image = 255*mls_rigid_deformation_inv(image, p, q, alpha=1, density=1)

    if np.random.uniform()>0.5:
        transformed_image = affine_im(transformed_image, transpix=0.05*image.shape[0]*(np.random.uniform(size=[2])-0.5), angle=5*(np.random.uniform()-0.5))
    return binary(transformed_image)

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        st()
        
        idx = np.random.randint(0,24)
        if(idx<10):
            idx = '0'+str(idx)
        else:
            idx = str(idx)
        
        AB_path = os.path.join(AB_path, "rendering", idx+".png")
        AB = Image.open(AB_path).convert('L')
        transform_params = get_params(self.opt, AB.size)
        
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        
        A = A_transform(AB)
        AB_ = np.squeeze(np.asarray(AB))
        if np.random.uniform() > 0.5:
            try:
                AB = transform_im(AB_)
            except:
                AB = AB_
        else:
            AB = AB_
        
        AB = 255*(1- morphology.binary_dilation(1-AB/255, iterations=5)) + 255*(1- morphology.binary_dilation(1 - binary(AB_)/255, iterations=5))
        AB = 255*(1-(AB < 500).astype(int))
        
        AB = Image.fromarray(np.uint8(AB))
        
        B = B_transform(AB)
        #A = transform_im

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
