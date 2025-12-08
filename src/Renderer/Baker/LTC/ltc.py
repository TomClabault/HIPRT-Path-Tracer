import os, sys, math, random, time
import numpy as np
import torch

import utils

'''
Anisotropic LTC implementation. This stores the LUT.
This is the batched version.

transform_* --> Functions that use the LUT to transform cosine/target samples to target/cosine, respectively.
forward --> Function that evaluates LTC amplitude. Transforms from target to cosine.

Refer to LTC paper: https://eheitzresearch.wordpress.com/415-2/
'''


class LTCAnisotropic(torch.nn.Module):

    def __init__(self, lut_size=(8, 8, 8, 8)):
        super().__init__()

        self.mat_flag = False
        self.lut_size = lut_size

        # Generates 8^4 LUT filled with identity matrices, and marks as a parameter for optimization
        self.LUT = torch.tile( torch.eye(3), (lut_size[0], lut_size[1], lut_size[2], lut_size[3], 1, 1) ).cuda()
        self.LUT = torch.nn.Parameter(self.LUT)

    def optimize_mat(self):
        self.mat_flag = True
    
    def freeze(self):
        self.mat_flag = False
    
    def get_mat(self, alphax_idx=None, alphay_idx=None, theta_idx=None, phi_idx=None):

        if alphax_idx is None and alphay_idx is None and theta_idx is None and phi_idx is None:
            # Reshape to (*, 3, 3), a batch of matrices, and return
            ltc_mat = self.LUT[:, :, :, :, :, :].reshape(-1, 3, 3)

            return ltc_mat
        
        else:
            # If dimensions are given, then retrive one single mat from LUT
            ltc_mat = self.LUT[alphax_idx, alphay_idx, theta_idx, phi_idx, :, :]

            return ltc_mat
    
    def transform_to_target(self, x):
        # x: shape expeced ( lut_size[0]*lut_size[1]*lut_size[2]*lut_size[3], N, 3 )

        ltc_mat = self.get_mat().reshape(-1, 3, 3)

        # Apply all matrices in the LUT in a single matmul operation
        x = x.permute(0, 2, 1)
        x0 = torch.matmul(ltc_mat, x).permute(0, 2, 1)
        x0 = x0 / torch.linalg.norm(x0, axis=2, keepdim=True)

        return x0
    
    def transform_to_cosine(self, x):
        # x: shape expeced ( lut_size[0]*lut_size[1]*lut_size[2]*lut_size[3], N, 3 )

        # Inverse is appliedon individual 3x3 matrices
        ltc_mat = torch.linalg.inv( self.LUT[:, :, :, :, :, :].reshape(-1, 3, 3) )

        # Apply all matrices in the LUT in a single matmul operation
        x = x.permute(0, 2, 1)
        x0 = torch.matmul(ltc_mat, x).permute(0, 2, 1)
        x0 = x0 / torch.linalg.norm(x0, axis=2, keepdim=True)

        return x0

class LTCIsotropic(torch.nn.Module):

    def __init__(self, lut_size=(64, 64)):
        super().__init__()

        self.mat_flag = False
        self.lut_size = lut_size

        self.LUT = torch.tile( torch.eye(3), (lut_size[0], lut_size[1], 1, 1) ).cuda()
        self.LUT = torch.nn.Parameter(self.LUT)

    def optimize_mat(self):
        self.mat_flag = True
    
    def freeze(self):
        self.mat_flag = False
    
    def get_mat(self, alpha_idx=None, theta_idx=None):

        if alpha_idx is None and theta_idx is None:
            ltc_mat = self.LUT[:, :, :, :].reshape(-1, 3, 3)

            return ltc_mat
        
        else:
            ltc_mat = self.LUT[alpha_idx, theta_idx, :, :]

            return ltc_mat
    
    def transform_to_target(self, x):
        # x: shape expeced ( lut_size[0]*lut_size[1], N, 3 )

        ltc_mat = self.get_mat().reshape(-1, 3, 3)

        x = x.permute(0, 2, 1)
        x0 = torch.matmul(ltc_mat, x).permute(0, 2, 1)
        x0 = x0 / torch.linalg.norm(x0, axis=2, keepdim=True)

        return x0

if __name__ == '__main__':
    ltc_aniso = LTCAnisotropic()

    ltc_aniso.freeze()
    
    x = torch.rand((8*8*8*8, 1000, 3), dtype=torch.float, device='cuda:0')
    print(ltc_aniso.transform_to_target(x).shape)