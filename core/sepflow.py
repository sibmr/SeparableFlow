import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, CorrBlock1D
from cost_agg import CostAggregation
from utils.utils import bilinear_sampler, coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass
class Guidance(nn.Module):
    def __init__(self, channels=32, refine=False):
        super(Guidance, self).__init__()
        self.bn_relu = nn.Sequential(nn.InstanceNorm2d(channels),
                                     nn.ReLU(inplace=True))
        self.conv0 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1),
                                   nn.InstanceNorm2d(16),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(16, int(channels/4), kernel_size=3, stride=2, padding=1),
                                   nn.InstanceNorm2d(int(channels/4)),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(int(channels/4), int(channels/2), kernel_size=3, stride=2, padding=1),
                                   nn.InstanceNorm2d(int(channels/2)),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(int(channels/2), channels, kernel_size=3, stride=2, padding=1),
                                   nn.InstanceNorm2d(channels),
                                   nn.ReLU(inplace=True))
        inner_channels = channels // 4
        self.wsize = 20
        self.conv1 = nn.Sequential(nn.Conv2d(channels*2, inner_channels, kernel_size=3, padding=1),
                                   nn.InstanceNorm2d(inner_channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
                                   nn.InstanceNorm2d(inner_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=1, padding=1),
                                   nn.InstanceNorm2d(inner_channels),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
                                   nn.InstanceNorm2d(inner_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=1, padding=1),
                                   nn.InstanceNorm2d(inner_channels),
                                   nn.ReLU(inplace=True))
        self.conv11 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels*2, kernel_size=3, stride=2, padding=1),
                                   nn.InstanceNorm2d(inner_channels*2),
                                   nn.ReLU(inplace=True))
        self.conv12 = nn.Sequential(nn.Conv2d(inner_channels*2, inner_channels*2, kernel_size=3, stride=1, padding=1),
                                   nn.InstanceNorm2d(inner_channels*2),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(inner_channels*2, inner_channels*2, kernel_size=3, stride=1, padding=1),
                                   nn.InstanceNorm2d(inner_channels*2),
                                   nn.ReLU(inplace=True))
        self.weights = nn.Sequential(nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
                                      nn.InstanceNorm2d(inner_channels),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inner_channels, self.wsize, kernel_size=3, stride=1, padding=1))
        self.weight_sg1 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
                                        nn.InstanceNorm2d(inner_channels),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(inner_channels, self.wsize*2, kernel_size=3, stride=1, padding=1))
        self.weight_sg2 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
                                        nn.InstanceNorm2d(inner_channels),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(inner_channels, self.wsize*2, kernel_size=3, stride=1, padding=1))
        self.weight_sg11 = nn.Sequential(nn.Conv2d(inner_channels*2, inner_channels*2, kernel_size=3, padding=1),
                                        nn.InstanceNorm2d(inner_channels*2),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(inner_channels*2, self.wsize*2, kernel_size=3, stride=1, padding=1))
        self.weight_sg12 = nn.Sequential(nn.Conv2d(inner_channels*2, inner_channels*2, kernel_size=3, padding=1),
                                        nn.InstanceNorm2d(inner_channels*2),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(inner_channels*2, self.wsize*2, kernel_size=3, stride=1, padding=1))
        self.weight_sg3 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
                                        nn.InstanceNorm2d(inner_channels),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(inner_channels, self.wsize*2, kernel_size=3, stride=1, padding=1))
        #self.getweights = nn.Sequential(GetFilters(radius=1),
        #                                nn.Conv2d(9, 20, kernel_size=1, stride=1, padding=0, bias=False))



    def forward(self, fea, img):
        x = self.conv0(img)
        x = torch.cat((self.bn_relu(fea), x), 1)
        x = self.conv1(x)
        rem = x
        x = self.conv2(x) + rem
        rem = x
        guid = self.weights(x)
        x = self.conv3(x) + rem
        sg1 = self.weight_sg1(x)
        sg1_u, sg1_v = torch.split(sg1, (self.wsize, self.wsize), dim=1)
        sg2 = self.weight_sg2(x)
        sg2_u, sg2_v = torch.split(sg2, (self.wsize, self.wsize), dim=1)
        sg3 = self.weight_sg3(x)
        sg3_u, sg3_v = torch.split(sg3, (self.wsize, self.wsize), dim=1)
        x = self.conv11(x)
        rem = x 
        x = self.conv12(x) + rem
        sg11 = self.weight_sg11(x)
        sg11_u, sg11_v = torch.split(sg11, (self.wsize, self.wsize), dim=1)
        sg12 = self.weight_sg12(x)
        sg12_u, sg12_v = torch.split(sg12, (self.wsize, self.wsize), dim=1)
        guid_u = dict([('sg1', sg1_u),
                       ('sg2', sg2_u),
                       ('sg3', sg3_u),
                       ('sg11', sg11_u),
                       ('sg12', sg12_u)])
        guid_v = dict([('sg1', sg1_v),
                       ('sg2', sg2_v),
                       ('sg3', sg3_v),
                       ('sg11', sg11_v),
                       ('sg12', sg12_v)])
        return guid, guid_u, guid_v 


class SepFlow(nn.Module):
    def __init__(self, args):
        super(SepFlow, self).__init__()
        self.args = args
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block

        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
        self.guidance = Guidance(channels=256)
        self.cost_agg1 = CostAggregation(in_channel=8)
        self.cost_agg2 = CostAggregation(in_channel=8)

    def freeze_bn(self):
        count1, count2, count3 = 0, 0, 0
        for m in self.modules():
            if isinstance(m, nn.SyncBatchNorm):
                count1 += 1
                m.eval()
            if isinstance(m, nn.BatchNorm2d):
                count2 += 1
                m.eval()
            if isinstance(m, nn.BatchNorm3d):
                count3 += 1
                #print(m)
                m.eval()
        #print(count1, count2, count3)
                #print(m)

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, upsample=True):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim
        
        # calculate per-eighth-pixel features
        fmap1 = self.fnet(image1)
        fmap2 = self.fnet(image2)
        
        # cast to float
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        # TODO: what is guidance?
        guid, guid_u, guid_v = self.guidance(fmap1.detach(), image1)
        
        # correlation now seems to use guidance
        corr_fn = CorrBlock(fmap1, fmap2, guid, radius=self.args.corr_radius)

        # context features calculation:
        # hidden state initialization + context features
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        
        # calculated C_u and C_v with shape (batch, |U|, ht, wd) / (batch, |V|, ht, wd)
        corr1, corr2 = corr_fn(None, sep=True)
        
        # initializing the flow (like raft, zero-flow)
        # coords1 - coords0 = flow = zeros
        coords0, coords1 = self.initialize_flow(image1)

        # cost aggregation reduces corr1 and corr2 from (batch, K, ht, wd) to (batch, 1, ht, wd)
        if self.training:
            u0, u1, flow_u, corr1 = self.cost_agg1(corr1, guid_u, max_shift=384, is_ux=True)
            v0, v1, flow_v, corr2 = self.cost_agg2(corr2, guid_v, max_shift=384, is_ux=False)
            flow_init = torch.cat((flow_u, flow_v), dim=1)
            
            flow_predictions = []
            flow_predictions.append(torch.cat((u0, v0), dim=1))
            flow_predictions.append(torch.cat((u1, v1), dim=1))
            flow_predictions.append(flow_init)
            
        else:
            flow_u, corr1 = self.cost_agg1(corr1, guid_u, max_shift=384, is_ux=True)
            flow_v, corr2 = self.cost_agg2(corr2, guid_v, max_shift=384, is_ux=False)
            flow_init = torch.cat((flow_u, flow_v), dim=1)
        
        # downsample inital flow
        flow_init = F.interpolate(flow_init.detach()/8.0, [cnet.shape[2], cnet.shape[3]], mode='bilinear', align_corners=True)
        
        # create 1d correlation block from C^A_u and C^A_v
        corr1d_fn = CorrBlock1D(corr1, corr2, radius=self.args.corr_radius)
        
        # update coords1 with inital flow estimate
        coords1 = coords1 + flow_init
        
        # iterative optimization
        for itr in range(iters):
            coords1 = coords1.detach()
            
            # index 4d correlation volume
            corr = corr_fn(coords1) # index correlation volume
            
            # index the two 3d correlation volumes
            corr1, corr2 = corr1d_fn(coords1) # index correlation volume
            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, corr1, corr2, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            if self.training:
                flow_predictions.append(flow_up)

        if self.training:
            return flow_predictions
        else:
            return coords1 - coords0, flow_up
            
