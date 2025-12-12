import os, sys, time, random, math
import numpy as np 
import torch

def sample_concentric_disc_batch(b, n):
    r1 = torch.rand((b, n), dtype=torch.float, device='cuda:0') * 2.0 - 1.0
    r2 = torch.rand((b, n), dtype=torch.float, device='cuda:0') * 2.0 - 1.0

    zero_x_y = torch.where(r1 == 0, True, False)
    zero_x_y = torch.logical_and(zero_x_y, torch.where(r2 == 0, True, False))
    zero_x_y = torch.stack((zero_x_y, zero_x_y), dim=2)

    zeros = torch.zeros((b, n, 2), dtype=torch.float, device='cuda:0')

    c1, c2 = 4 * r1, 4 * r2
    x = torch.where(torch.abs(r1) > torch.abs(r2), torch.cos(np.pi * r2 / c1), torch.cos(np.pi / 2 - np.pi * r1 / c2))
    y = torch.where(torch.abs(r1) > torch.abs(r2), torch.sin(np.pi * r2 / c1), torch.sin(np.pi / 2 - np.pi * r1 / c2))

    r = torch.where(torch.abs(r1) > torch.abs(r2), r1, r2)
    r = torch.stack((r, r), dim=2)

    points = r * torch.stack((x, y), dim=2)

    return torch.where(zero_x_y, zeros, points)

def sample_cosine_batch(b, n):
    disk_point = sample_concentric_disc_batch(b, n)

    xx = disk_point[:, :, 0] ** 2
    yy = disk_point[:, :, 1] ** 2

    z = torch.tensor(1, dtype=torch.float, device='cuda:0') - xx - yy
    z = torch.sqrt(torch.where(z < 0., torch.tensor(0., dtype=torch.float, device='cuda:0'), z))
    z = torch.unsqueeze(z, dim=2)

    wi_d = torch.cat((disk_point, z), dim=2)
    wi_d = wi_d / (torch.linalg.norm(wi_d, dim=2, keepdim=True) + 1e-8)

    return wi_d

def sample_anisotropic_batch(b, n, v, alphax, alphay):
    # Based on Eric's paper: https://jcgt.org/published/0007/04/01/
    # Batched implementation

    # b: batch
    # n: number of smaples
    # v: (b, 3)
    # alphax, alphay: (b)

    zero = torch.tensor(0.0, dtype=torch.float, device='cuda:0')
    
    v = torch.unsqueeze(v, dim=1)
    v = v.repeat(1, n, 1)
    v[:, :, 2] = torch.maximum(zero, v[:, :, 2])

    # alphax = torch.tensor(alphax, dtype=torch.float, device='cuda:0')
    alphax = torch.unsqueeze( torch.unsqueeze(alphax, dim=1), dim=1)
    alphax = alphax.repeat(1, n, 1)
    # alphay = torch.tensor(alphay, dtype=torch.float, device='cuda:0')
    alphay = torch.unsqueeze( torch.unsqueeze(alphay, dim=1), dim=1)
    alphay = alphay.repeat(1, n, 1)
    
    u1 = torch.rand((b, n, 1), dtype=torch.float, device='cuda:0')
    u2 = torch.rand((b, n, 1), dtype=torch.float, device='cuda:0')

    vh = torch.cat( (alphax * v[:, :, 0:1], alphay * v[:, :, 1:2], v[:, :, 2:3]), dim=2 )
    vh = vh / (torch.linalg.norm(vh, dim=2, keepdim=True)+1e-8)

    lensq = vh[:, :, 0:1] * vh[:, :, 0:1] + vh[:, :, 1:2] * vh[:, :, 1:2]
    # idx = torch.where(lensq[:, :, 0] == 0)
    idx = torch.where(torch.where(lensq[:, :, 0] == 0, True, False))
    lensq[idx] = 1e-8

    zero_tensor = torch.zeros((vh.shape[0], vh.shape[1], 1), dtype=torch.float, device='cuda:0')

    T1 = torch.cat( (-vh[:, :, 1:2], vh[:, :, 0:1], zero_tensor), dim=2 ) / torch.sqrt(lensq)
    T1[idx] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float, device='cuda:0')
    
    T2 = torch.cross(vh, T1)

    r = torch.sqrt(u1)
    phi = 2.0 * math.pi * u2
    t1 = r * torch.cos(phi)
    t2 = r * torch.sin(phi)
    s = 0.5 * (1.0 + vh[:, :, 2:3])
    t2 = (1.0 - s) * torch.sqrt(1.0 - t1*t1) + s*t2

    maxi = torch.maximum(zero, 1.0 - t1*t1 - t2*t2)
    nh = t1*T1 + t2*T2 + torch.sqrt(maxi)*vh

    ne = torch.cat( (alphax * nh[:, :, 0:1], alphay * nh[:, :, 1:2], torch.maximum(zero, nh[:, :, 2:3])), dim=2 )
    ne = ne / (torch.linalg.norm(ne, dim=2, keepdim=True)+1e-8)

    l = -v + 2.0 * ne * torch.sum(ne*v, dim=2, keepdim=True)
    l = l / (torch.linalg.norm(l, axis=2, keepdim=True)+1e-8)

    return l, ne, v, alphax, alphay

def buffer_rejection(b, n, buffer, target, v, alphax, alphay):
    # Probabilistically replace samples in buffer using rejection sampling
    # based on the ratio G2/G1

    wi_target, _, wo_, alphax_, alphay_ = sample_anisotropic_batch(b, n, v, alphax, alphay)

    weight = target.G2(alphax_, alphay_, wo_, wi_target) / target.G1(alphax_, alphay_, wo_)
    eta = torch.rand((wi_target.shape[0], wi_target.shape[1], 1), dtype=torch.float, device='cuda:0')

    cond = torch.where(weight >  eta, True, False)
    buffer = torch.where(cond, wi_target, buffer)

    return buffer

def sample_anisotropic(n, v, alphax, alphay):
    # Based on Eric's paper: https://jcgt.org/published/0007/04/01/

    # n: number of smaples
    # v: view vector

    zero_tensor = torch.tensor(0.0, dtype=torch.float, device='cuda:0')
    
    v = v.repeat(n, 1)
    v[:, 2] = torch.maximum(zero_tensor, v[:, 2])

    # alphax = torch.tensor(alphax, dtype=torch.float, device='cuda:0')
    # alphay = torch.tensor(alphay, dtype=torch.float, device='cuda:0')
    
    u1 = torch.rand((n, 1), dtype=torch.float, device='cuda:0')
    u2 = torch.rand((n, 1), dtype=torch.float, device='cuda:0')

    vh = torch.hstack( (alphax * v[:, 0:1], alphay * v[:, 1:2], v[:, 2:3]) )
    vh = vh / (torch.linalg.norm(vh, dim=1, keepdim=True)+1e-8)

    lensq = vh[:, 0:1] * vh[:, 0:1] + vh[:, 1:2] * vh[:, 1:2]
    idx = torch.where(lensq[:, 0] == 0)
    lensq[idx[0]] = 1e-8

    T1 = torch.hstack( (-vh[:, 1:2], vh[:, 0:1], torch.zeros((vh.shape[0], 1), dtype=torch.float, device='cuda:0')) ) / torch.sqrt(lensq)
    T1[idx] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float, device='cuda:0')
    
    T2 = torch.cross(vh, T1)

    r = torch.sqrt(u1)
    phi = 2.0 * math.pi * u2
    t1 = r * torch.cos(phi)
    t2 = r * torch.sin(phi)
    s = 0.5 * (1.0 + vh[:, 2:3])
    t2 = (1.0 - s) * torch.sqrt(1.0 - t1*t1) + s*t2

    maxi = torch.maximum(zero_tensor, 1.0 - t1*t1 - t2*t2)
    nh = t1*T1 + t2*T2 + torch.sqrt(maxi)*vh

    ne = torch.hstack( (alphax * nh[:, 0:1], alphay * nh[:, 1:2], torch.maximum(zero_tensor, nh[:, 2:3])) )
    ne = ne / (torch.linalg.norm(ne, dim=1, keepdim=True)+1e-8)

    l = -v + 2.0 * ne * torch.sum(ne*v, axis=1, keepdims=True)
    l = l / (torch.linalg.norm(l, axis=1, keepdims=True)+1e-8)

    return l, ne, v

def rejection_sampling(n, target, v, alphax, alphay):
    # Sample 'n' directions from 'target', given view vector 'v', 'alphax' & 'alphay'

    wi_target, _, wo_ = sample_anisotropic(n, v, alphax, alphay)

    weight = target.G2(alphax, alphay, wo_, wi_target) / target.G1(alphax, alphay, wo_)
    eta = torch.rand((wi_target.shape[0], 1), dtype=torch.float, device='cuda:0')

    cond = torch.where(weight >  eta)
    wi_target = wi_target[cond[0]] # Probabilistically select directions based on weight (ratio of G2 and G1)

    return wi_target

def sample_concentric_disc(n):
    r1, r2 = torch.rand(n, dtype=torch.float, device='cuda:0') * 2.0 - 1.0, torch.rand(n, dtype=torch.float, device='cuda:0') * 2.0 - 1.0

    zero_x_y = torch.where(r1 == 0, True, False)
    zero_x_y = torch.logical_and(zero_x_y, torch.where(r2 == 0, True, False))
    zero_x_y = torch.stack((zero_x_y, zero_x_y), dim=1)
    zeros = torch.zeros((n, 2), dtype=torch.float, device='cuda:0')

    c1, c2 = 4 * r1, 4 * r2
    x = torch.where(torch.abs(r1) > torch.abs(r2), torch.cos(np.pi * r2 / c1), torch.cos(np.pi / 2 - np.pi * r1 / c2))
    y = torch.where(torch.abs(r1) > torch.abs(r2), torch.sin(np.pi * r2 / c1), torch.sin(np.pi / 2 - np.pi * r1 / c2))

    r = torch.where(torch.abs(r1) > torch.abs(r2), r1, r2)
    r = torch.stack((r, r), dim=1)

    points = r * torch.stack((x, y), dim=1)

    return torch.where(zero_x_y, zeros, points)

def sample_cosine(n):
    disk_point = sample_concentric_disc(n)
    xx = disk_point[:, 0] ** 2
    yy = disk_point[:, 1] ** 2
    z = torch.tensor(1, dtype=torch.float, device='cuda:0') - xx - yy
    z = torch.sqrt(torch.where(z < 0., torch.tensor(0., dtype=torch.float, device='cuda:0'), z))

    wi_d = torch.cat((disk_point, torch.unsqueeze(z, dim=-1)), dim=-1)
    wi_d = wi_d / (torch.linalg.norm(wi_d, dim=1, keepdims=True) + 1e-8)

    return wi_d