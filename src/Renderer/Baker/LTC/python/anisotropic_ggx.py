import os, sys, math, random, time

import torch
import numpy as np

import utils

'''
Anisotropic model of GGX based on Eric Heitz et al.: https://jcgt.org/published/0007/04/01/

GGXAniso: This is a serian implementation i.e. processes one view and light direction pair.
GGXAnisoBatch: This is a batched implementation i.e. processes batches of view and light directions at once.

eval() returns the BRDF value given a pair of view and light directions.
Call eval() with a pair of view (V) and light (L) diretions.

Expects V, L to be batched. 
V --> (B, 3) B 3d-vectors
L --> (B, N, 3) B batches of vectors, each batch contains N vectors
        For example, each cell of LUT contains 20K samples of that distribution.

Each V has many L for which BRDF value is returned.

Returns:
BRDF --> (B, N, 1) response of the BRDF function.

Fresnel term not included. 
cos(\theta_l) factor cancelled from the cosine foreshortening term in rendering eq.

f(V, L) = D(H) * G2(V, L)
          ---------------
          4 * cos(\theta_v)

'''

class GGXAnisoBatch:

    def __init__(self):
        pass
    
    def D(self, alpha_x, alpha_y, N):
        value = math.pi * alpha_x * alpha_y * ((N[:, :, 0:1]/alpha_x)**2 + (N[:, :, 1:2]/alpha_y)**2 + N[:, :, 2:3]**2)**2

        return 1.0/value
    
    def Lambda(self, alpha_x, alpha_y, V):
        value = 0.5 * (-1 + torch.sqrt(1.0 + ((V[:, :, 0:1]*alpha_x)**2 + (V[:, :, 1:2]*alpha_y)**2)/(V[:, :, 2:3]**2)))

        return value
    
    def G1(self, alpha_x, alpha_y, V):
        value = 1.0 / (1.0 + self.Lambda(alpha_x, alpha_y, V))

        zeros = torch.zeros(value.shape, dtype=torch.float, device='cuda:0')
        idx = torch.where(V[:, :, 2:3] < 0, True, False)
        value = torch.where(idx, zeros, value)

        return value
    
    def G2(self, alpha_x, alpha_y, V, L):
        value = 1.0 / (1.0 + self.Lambda(alpha_x, alpha_y, V) + self.Lambda(alpha_x, alpha_y, L))

        zeros = torch.zeros(value.shape, dtype=torch.float, device='cuda:0')
        idx = torch.where(V[:, :, 2:3] < 0, True, False)
        value = torch.where(idx, zeros, value)

        zeros = torch.zeros(value.shape, dtype=torch.float, device='cuda:0')
        idx = torch.where(L[:, :, 2:3] < 0, True, False)
        value = torch.where(idx, zeros, value)

        return value
    
    def eval(self, V, L, alphax, alphay):
        # V: shape expected ( b, 3 )
        # L: shape expected ( b, N, 3 )
        # alphax, alphay: shape expected ( b )

        alphax = torch.unsqueeze( torch.unsqueeze(alphax, dim=1), dim=1)
        alphax = alphax.repeat(1, L.shape[1], 1)

        alphay = torch.unsqueeze( torch.unsqueeze(alphay, dim=1), dim=1)
        alphay = alphay.repeat(1, L.shape[1], 1)
        
        V = torch.unsqueeze(V, dim=1)
        V = V.repeat(1, L.shape[1], 1)

        H = (V+L)
        H = H / torch.linalg.norm(H, dim=2, keepdim=True)

        pdf = self.D(alphax, alphay, H)
        value = pdf * self.G2(alphax, alphay, V, L) / 4.0 / V[:, :, 2:3]

        return value, pdf
    
    def calc_nD_fD(self, V, alphax, alphay):
        # Calculate amplitude and fresnel coefficients
        # Refer to: https://advances.realtimerendering.com/s2016/s2016_ltc_fresnel.pdf

        # V: shape expected ( b, 3 )
        # alphax, alphay: shape expected ( b )

        L, _, V, alphax, alphay = utils.sample_anisotropic_batch(V.shape[0], 10000, V, alphax, alphay)

        H = (V+L)
        H = H / torch.linalg.norm(H, dim=2, keepdim=True)

        w = self.G2(alphax, alphay, V, L) / self.G1(alphax, alphay, V)
        fr = ( 1 - torch.sum(V*H, dim=2, keepdim=True) ) ** 5

        nD = torch.mean( torch.squeeze(w, dim=2), dim=1 )
        fD = torch.mean( torch.squeeze(fr*w, dim=2), dim=1 )

        return nD, fD

class GGXAniso:

    def __init__(self):
        pass
    
    def D(self, alpha_x, alpha_y, N):
        value = math.pi * alpha_x * alpha_y * ((N[:, 0:1]/alpha_x)**2 + (N[:, 1:2]/alpha_y)**2 + N[:, 2:3]**2)**2
        return 1.0/value
    
    def Lambda(self, alpha_x, alpha_y, V):
        value = 0.5 * (-1 + torch.sqrt(1.0 + ((V[:, 0:1]*alpha_x)**2 + (V[:, 1:2]*alpha_y)**2)/(V[:, 2:3]**2)))
        return value
    
    def G1(self, alpha_x, alpha_y, V):
        value = 1.0 / (1.0 + self.Lambda(alpha_x, alpha_y, V))

        idx = V[:, 2:3] < 0
        value[idx] = 0

        return value
    
    def G2(self, alpha_x, alpha_y, V, L):
        value = 1.0 / (1.0 + self.Lambda(alpha_x, alpha_y, V) + self.Lambda(alpha_x, alpha_y, L))

        idx = V[:, 2:3] < 0
        value[idx] = 0

        idx = L[:, 2:3] < 0
        value[idx] = 0

        return value
    
    def eval(self, V, L, alphax, alphay):
        V = V.repeat(L.shape[0], 1)

        H = (V+L)
        H = H / torch.linalg.norm(H, dim=1, keepdim=True)

        value = self.D(alphax, alphay, H) * self.G2(alphax, alphay, V, L) / 4.0 / V[:, 2:3]

        return value, value

    def norm(self, v, alphax, alphay):
        s = utils.sample_hemisphere_cosine(2500)
        
        val, pdf = self.eval(v, s, alphax, alphay)
        res = val / pdf
        res = torch.mean(res)

        return res
    
    def avg_dir(self, v, alphax, alphay):
        s = utils.sample_hemisphere_cosine(2500)
        val, pdf = self.eval(v, s, alphax, alphay)
        d = s / pdf
        d = torch.mean(d, axis=0)
        d[2] = max(0.0, d[2])
        d = d / (torch.linalg.norm(d) + 1e-8)

        return d

if __name__ == '__main__':
    ggx = GGXAnisoBatch()

    theta = 0
    theta = theta * np.pi / 180.0
    phi = 0
    phi = phi * np.pi / 180.0

    wo = torch.tensor([ np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta) ]).type(torch.float)
    wo = wo / torch.linalg.norm(wo)
    wo = torch.unsqueeze(wo, dim=0).cuda()

    alphax = torch.tensor([1.0], dtype=torch.float, device='cuda:0')
    alphay = torch.tensor([1.0], dtype=torch.float, device='cuda:0')

    nD, fD = ggx.calc_nD_fD(wo, alphax, alphay)

    print(nD)
    print(fD)