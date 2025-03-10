import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.utils import bilinear_sampler, coords_grid
from libs.GANet.modules.GANet import NLFMax, NLFIter

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class NLF(nn.Module):

    def __init__(self, in_channel=32):
        super(NLF, self).__init__()
        self.nlf = NLFIter()
    
    def forward(self, x, g):
        """ 4 directional aggregation of 4d correlation volume
            TODO: what does nlf stand for?

        Args:
            x (torch.Tensor): input tensor (4d correlation volume)
            g (torch.Tensor): guidance for 4d correlation volume
        """
        N, D1, D2, H, W = x.shape

        # merge pixel height/width
        x = x.reshape(N, D1*D2, H, W).contiguous()
        rem = x
        
        # split guidance for each direction: down, up, right, left
        k1, k2, k3, k4 = torch.split(g, (5, 5, 5, 5), 1)

#        k1, k2, k3, k4 = self.getweights(x)
        
        # L1 normalize the guidance for each direction
        k1 = F.normalize(k1, p=1, dim=1)
        k2 = F.normalize(k2, p=1, dim=1)
        k3 = F.normalize(k3, p=1, dim=1)
        k4 = F.normalize(k4, p=1, dim=1)

        # apply nlf to 4d cost volume
        x = self.nlf(x, k1, k2, k3, k4)
#        x = x + rem

        # reshape to separate pixel channels
        x = x.reshape(N, D1, D2, H, W)
        
        return x
        


class CorrBlock:
    def __init__(self, fmap1, fmap2, guid, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        self.nlf = NLF()
        corr = self.corr_compute(fmap1, fmap2, guid, reverse=True)
        #corr = self.nlf(corr, g)
        #corr = corr.permute(0, 3,4, 1,2)

        batch, h1, w1, h2, w2 = corr.shape
        self.shape = corr.shape
        corr = corr.reshape(batch*h1*w1, 1, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)
    
    def separate(self):
        """ create two separate correlation volumes 
            in this version, they only consist of the max and avg
            -> no use of self-adaptive compression, because only attention in vector
            -> no additional fields aggregated through attention

            they are interpolated on every level to have the original ht/wd dimension

            shape: 
                sep_u: (batch, 2*levels, wd, ht, wd)
                sep_v: (batch, 2*levels, ht, ht, wd)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
        """
        sep_u = []
        sep_v = []

        # for each level of 4d correlation volume pyramid
        for i in range(self.num_levels):
            # correlation volume at level i
            # shape: (batch*ht*wd, 1, ht/2**i, wd/2**i)
            corr = self.corr_pyramid[i]
            
            # max values of correlation volume
            # shape: (batch*ht*wd, 1, 1, wd/2**i)
            m1, _ = corr.max(dim=2, keepdim=True)
            
            # avg values of correlation volume
            # shape: (batch*ht*wd, 1, 1, wd/2**i)
            m2 = corr.mean(dim=2, keepdim=True)
            
            # this is the attention vector from the paper
            # sep only contains max and avg, no additional attention-based fields like in paper
            # (batch*ht*wd, 1, 1, wd/2**i) (batch*ht*wd, 1, 1, wd/2**i) -> (batch*ht*wd, 1, 2, wd/2**i)
            sep = torch.cat((m1, m2), dim=2)
            
            # reshape: (batch*ht*wd, 1, 2, wd/2**i)     -> (batch, ht, wd, 2, wd/2**i)
            # permute: (batch, ht, wd, 2, wd/2**i)   -> (batch, 2, wd/2**i, ht, wd)
            sep = sep.reshape(self.shape[0], self.shape[1], self.shape[2], sep.shape[2], sep.shape[3]).permute(0, 3, 4, 1, 2)
            
            # TODO: why upsample from reduced level resolution?
            # maybe they used the largest-level correlation volume to pair the attention with
            # also, upsampling is the only way the concatenation at the end is possible (otherwise, shape[2] would not match)
            # (batch, 2, wd/2**i, ht, wd) -> (batch, 2, wd, ht, wd)
            sep = F.interpolate(sep, [self.shape[4], self.shape[1], self.shape[2]], mode='trilinear', align_corners=True)
            
            sep_u.append(sep)
            
            # exactly the same for the width dimension
            # (batch*ht*wd, 1, ht/2**i, 1)
            m1, _ = corr.max(dim=3, keepdim=True)
            # (batch*ht*wd, 1, ht/2**i, 1)
            m2 = corr.mean(dim=3, keepdim=True)
            # (batch*ht*wd, 1, ht/2**i, 2)
            sep = torch.cat((m1, m2), dim=3)
            # (batch, 2, ht/2**i, ht, wd)
            sep = sep.reshape(self.shape[0], self.shape[1], self.shape[2], sep.shape[2], sep.shape[3]).permute(0, 4, 3, 1, 2)
            # (batch, 2, ht, ht, wd)
            sep = F.interpolate(sep, [self.shape[3], self.shape[1], self.shape[2]], mode='trilinear', align_corners=True)
            
            sep_v.append(sep)
        
        # concatenate over all levels
        # liste -> (batch, 2*levels, wd, ht, wd)
        sep_u = torch.cat(sep_u, dim=1)
        # liste -> (batch, 2*levels, ht, ht, wd)
        sep_v = torch.cat(sep_v, dim=1)
        
        return sep_u, sep_v


    def __call__(self, coords, sep=False):
        
        # returns two 3d cost volumes
        # sep_u: (batch, ht, wd, ht), sep_v: (batch, ht, wd, wd)
        # very strange behaviour:
        #   if sep==False:
        #       return indexed 4d correlation volume
        #   if sep==True:
        #       return two NOT indexed 3d correlation volumes
        #       they need to be indexed in CorrBlock1D still
        if sep:
            return self.separate()
        
        # this part is the same as RAFT
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dy = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)

        # permutation:
        # (batch, h1, w1, num_levels*dim*(2*r+1)*(2*r+1)) -> (batch, num_levels*dim*(2*r+1)*(2*r+1), h1, w1)
        # also contiguous is used to copy the tensor with new memory layout according to shape
        return out.permute(0, 3, 1, 2).contiguous().float()

    #@staticmethod
    def corr_compute(self, fmap1, fmap2, guid, reverse=True):
        batch, dim, ht, wd = fmap1.shape
        
        # flatten height/width dimension
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        # reverse has matmul(fmap2, fmap1) instead of matmul(fmap1, fmap2)
        if reverse:
            
            # see non-reverse branch
            corr = torch.matmul(fmap2.transpose(1,2), fmap1) / torch.sqrt(torch.tensor(dim).float())
            corr = corr.view(batch, ht, wd, ht, wd)
            corr = self.nlf(corr, guid)
            
            # swapping the fmap1 height/width dimensions back to the front:
            # (batch, ht_fmap2, wd_fmap2, ht_fmap1, wd_fmap1) 
            # -> (batch, ht_fmap1, wd_fmap1, ht_fmap2, wd_fmap2)
            corr = corr.permute(0, 3, 4, 1, 2)

        # non-reverse has matmul(fmap1, fmap2) as in raft
        else:
            # standart 4d correlation volume computation from raft
            # matmul: (batch, ht*wd, feature_dim) (batch, feature_dim, ht*wd) -> (batch, ht*wd, ht*wd)
            corr = torch.matmul(fmap1.transpose(1,2), fmap2) / torch.sqrt(torch.tensor(dim).float())
            
            # reshape to separate height/width dimensions
            corr = corr.view(batch, ht, wd, ht, wd)
            
            # nlf seems to somehow combine the 4d correlation volume with guidance 
            # guidance is a combination of fmap1 and img1
            # nlf has an up, down, left and right component
            # shape: (batch, ht, wd, ht, wd)
            corr = self.nlf(corr, guid)

        return corr


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())

class CorrBlock1D:

    def __init__(self, corr1, corr2, num_levels=4, radius=4):
        """ responsible for indexing the 3d correlation volumes for u and v


        Args:
            corr1 (torch.Tensor): _description_
            corr2 (_type_): _description_
            num_levels (int, optional): _description_. Defaults to 4.
            radius (int, optional): _description_. Defaults to 4.
        """
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid1 = []
        self.corr_pyramid2 = []

        corr1 = corr1.permute(0,3,4,1,2)
        corr2 = corr2.permute(0,3,4,1,2)
        batch, h1, w1, dim, w2 = corr1.shape
        batch, h1, w1, dim, h2 = corr2.shape
        assert(corr1.shape[:-1] == corr2.shape[:-1])
        assert(h1 == h2 and w1 == w2)

        #self.coords = coords_grid(batch, h2, w2).to(corr1.device)

        corr1 = corr1.reshape(batch*h1*w1, dim, 1, w2)
        corr2 = corr2.reshape(batch*h1*w1, dim, 1, h2)

        self.corr_pyramid1.append(corr1)
        self.corr_pyramid2.append(corr2)
        for i in range(self.num_levels):
            corr1 = F.avg_pool2d(corr1, [1,2], stride=[1,2])
            self.corr_pyramid1.append(corr1)
            corr2 = F.avg_pool2d(corr2, [1,2], stride=[1,2])
            self.corr_pyramid2.append(corr2)
            #print(corr1.shape, corr1.mean().item(), corr2.shape, corr2.mean().item())
    def bilinear_sampler(self, img, coords, mode='bilinear', mask=False):
        """ Wrapper for grid_sample, uses pixel coordinates """
        H, W = img.shape[-2:]
        xgrid, ygrid = coords.split([1,1], dim=-1)
        xgrid = 2*xgrid/(W-1) - 1
        assert torch.unique(ygrid).numel() == 1 and H == 1 # This is a stereo problem

        grid = torch.cat([xgrid, ygrid], dim=-1)
        img = F.grid_sample(img, grid, align_corners=True)

        if mask:
            mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
            return img, mask.float()

        return img

    def __call__(self, coords):
        coords_org = coords.clone()
        coords = coords_org[:, :1, :, :]
        coords = coords.permute(0, 2, 3, 1)
        r = self.radius
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid1[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(1, 1, 2*r+1, 1).to(coords.device)
            x0 = dx + coords.reshape(batch*h1*w1, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            coords_lvl = torch.cat([x0,y0], dim=-1)
            coords_lvl = torch.clamp(coords_lvl, -1, 1)
            #print("corri:", corr.shape, corr.mean().item(), coords_lvl.shape, coords_lvl.mean().item())
            corr = self.bilinear_sampler(corr, coords_lvl)
            #print("corri:", corr.shape, corr.mean().item())
            corr = corr.view(batch, h1, w1, -1)
            #print("corri:", corr.shape, corr.mean().item())
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        out1 = out.permute(0, 3, 1, 2).contiguous().float()

        coords = coords_org[:, 1:, :, :]
        coords = coords.permute(0, 2, 3, 1)
        r = self.radius
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid2[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(1, 1, 2*r+1, 1).to(coords.device)
            x0 = dx + coords.reshape(batch*h1*w1, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            coords_lvl = torch.cat([x0, y0], dim=-1)
            corr = self.bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        out2 = out.permute(0, 3, 1, 2).contiguous().float()
        return out1, out2
