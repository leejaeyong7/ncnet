from os import path
import torch
import torch.nn as nn
import torch.nn.functional as NF
import random
import numpy as np
import time

from argparse import ArgumentParser, Namespace
from .matching import filter_match, symmetric_match, compute_match_metrics
from .diffusion_filter import create_complex_kernel, create_simple_filter
import matplotlib.pyplot as plt
from lib.neigh_consensus import NeighConsensus



class PatchMatchConsensus(NeighConsensus):
    default_options = {
        'filter_type': {
            'argtype': str,
            'choices': ['simple', 'complex'],
            'default': 'complex'
        },
        'num_iterations': {
            'type': int,
            'default': 8
        },
        'num_neighbors': {
            'type': int,
            'default': 2
        },
    }
    def __init__(self, *args, **kwargs):
        super(PatchMatchConsensus, self).__init__(*args, **kwargs)
        self.hparams = self.setup_hparams(None)
        if(self.hparams.filter_type == 'simple'):
            self.prop_kernel = create_simple_filter()
            self.prop_pad = nn.ReflectionPad2d(5)
        else:
            self.prop_kernel = create_complex_kernel()
            self.prop_pad = nn.ReflectionPad2d(5)

    def setup_hparams(self, hparams):
        h_dict = vars(hparams) if hparams != None else {}
        defaults = {
            k: v['default'] 
            for k, v 
            in PatchMatchConsensus.default_options.items()
        }

        merged_dict = {**defaults, **h_dict}
        merged_namespace = Namespace()
        merged_namespace.__dict__.update(merged_dict)
        return merged_namespace

    def permute_neighbors(self, kp_map):
        '''
        Arguments:
            kp_map: Dx2xFHxFW tensor representing key points
        '''
        permuter = torch.LongTensor([
            -1, 0, 1,
            -1, 0, 1,
            -1, 0, 1,
        ]).view(3, 3).to(kp_map.device)

        # 2x3x3
        n_permuter = torch.stack((permuter.t(), permuter), dim=1).view(1, 9, 2, 1, 1)
        return kp_map.unsqueeze(1) + n_permuter 

    # -- actual model definitions -- #
    def forward(self, dcorrs, src_f, tar_f, num_iterations=8):
        """
        dcorrs: _, _, SH, SW, TH, TW shaped tensor
        src_f: CxSHxSW shaped tensor
        tar_f: CxSHxSW shaped tensor
        """

        corrs = dcorrs[0, 0].permute(2, 3, 0, 1)
        SFH, SFW, RFH, RFW = corrs.shape
        dev = corrs.device

        # initialize depth map.
        # Depth map is initialized with D intervals from min / max ranges
        init_ks = corrs.view(-1, RFH, RFW).argmax(0)
        init_h = init_ks // SFW
        init_w = init_ks % SFW

        nd_map = torch.stack((init_w, init_h), dim=0).unsqueeze(0)


        for iteration in range(num_iterations):
            # fig, axes = plt.subplots(1, 2)
            # [a.axis('off') for a in axes]
            # axes[0].imshow(nd_map[0, 0].cpu())
            # axes[1].imshow(nd_map[0, 1].cpu())
            # fig.savefig('{:04d}.png'.format(iteration))
            # plt.close(fig)

            # aggre depth map contains DxSx2xHxW values
            aggr_kp_maps = self.propagate(nd_map)
            neighbor_maps = self.permute_neighbors(nd_map)
            aggr_kp_maps = torch.cat((aggr_kp_maps, neighbor_maps), dim=1)
            aggr_kp_maps[:, :, 0].clamp_(0, SFW - 1)
            aggr_kp_maps[:, :, 1].clamp_(0, SFH - 1)

            D, S, _,  H, W = aggr_kp_maps.shape

            scores = []

            # evaluate each depth and propagate

            s_scores = self.compute_scores(corrs, aggr_kp_maps, True)

            # DxSxHxW
            # s_scores = torch.cat(scores, dim=1)

            # DxSx2xHxW => Dx2xHxW
            nd_map = self.update(aggr_kp_maps, s_scores, dim=1).squeeze(1)


        # among best D scores, choose the best one
        d_scores = s_scores.max(1)[0]
        kp_map = self.update(nd_map.unsqueeze(1), d_scores.unsqueeze(1), dim=0)[0, 0]
        scores = d_scores.max(0)[0]
        kp_map = kp_map.permute(1, 2, 0).float()
        return kp_map, scores

    def compute_scores(self, corrs, kp_maps, out):
        '''
        Arguments:
            kp_map: DxRHxRWx2
            corrs: RHxRWxSHxSW

        Returns:
            Dx1xRHxRW
        '''
        D, P, _, RH, RW = kp_maps.shape
        SH, SW, RH, RW = corrs.shape
        dev = corrs.device
        n = self.hparams.num_neighbors
        S = 2 * n + 1
        V = S ** 2

        xs = kp_maps[:, :, 0]
        ys = kp_maps[:, :, 1]
        
        # DxSxRHxRW
        inds = (ys * SW + xs).view(-1, RH, RW)


        # n_inds = VxDxRHxRW
        corrs_f = corrs.view(1, -1, RH, RW)
        d_scores = torch.zeros((V, V, D, P, RH, RW), device=dev)

        # r padded = SHxSWxRH+2S+RW+2S
        r_padded = NF.pad(corrs_f, [n, n, n, n])[0].view(SH*SW, -1)
        r_padded_t = r_padded.t().view(1, -1, SH, SW)

        # RH2S, RW2S, SH2S, SW2S
        padded = NF.pad(r_padded_t, [n, n, n, n])[0].permute(1, 2, 0)
        padded = padded.view(RH + 2 * n, RW + 2 * n, SH + 2 * n, SW + 2 * n)

        for rx in range(-n, n + 1):
            pt = n - rx
            pb = n + rx
            for ry in range(-n, n + 1):
                pr = n - ry
                pl = n + ry
                rvid = rx * S + ry

                start = time.time()

                for sx in range(-n, n + 1):
                    st = n - sx
                    sb = n + sx
                    for sy in range(-n, n + 1):
                        sr = n - sy
                        sl = n + sy
                        svid = sx * S + sy
                        # 

                        # ipad = SHSW, RH, RW
                        ipad = padded[sb:-st, sl:-sr, pb:-pt, pl:-pr].reshape(-1, RH, RW)

                        # ipad = SHSW x RH x RW
                        # inds = DxRHxRW
                        s = ipad.gather(0, inds)

                        d_scores[rvid, svid] = s.view(D, P, RH, RW)
                end= time.time()
                # print('looping took {}'.format(end - start))

        # VxVxDxPxRHxRW
        f_scores = d_scores.permute(2, 3, 4, 5, 0, 1).view(-1, 1, S, S, S, S)
        computed = self.conv(f_scores)

        return computed


    def propagate(self, kp_map):
        '''
        Given depth map, obtain propagatable depth values by using convolution
        Arguments:
            depth_map(torch.Tensor): Dx2xHxW kp maps
        Returns:
            aggr_depth_map(torch.Tensor): 2xSxHxW depth values, where S 
                                          represents number of valid pixels in 
                                          the filter
        '''
        # padded kp => Dx2xWxH => Dx1x2x(H+m)x(W+m)
        padded_kp= self.prop_pad(kp_map.float()).unsqueeze(1)

        # filter = Sx1x1xKHxKW
        kernel = self.prop_kernel.to(kp_map.device).unsqueeze(1)

        # returns DxSx2xHxW
        return NF.conv3d(padded_kp, kernel).long()


    def update(self, aggr_map, scores, dim=1):
        '''
        Aggregates values based on scores on dimension.

        given NxMxLxO... maps
              NxMxO... scores
        aggregates map into MxLxO or NxLxO depending on dimension

        reduction is done using argmax of score

        Arguments:
            aggr_map: DxSx2xHxW depth maps
            scores: DxSxHxW scores
            dim: dimension to perform aggregation

        Returns:
            aggregated : tensor of reduced shape, where reduction is done on dim
        '''
        # we only want to use depth values between min / max ranges
        score_index = scores.argmax(dim).unsqueeze(dim)
        score_index = score_index.unsqueeze(2).repeat(1, 1, 2, 1, 1)
        return aggr_map.gather(dim, score_index)

    def symmetric_match(self, ref_corr, src_corr, thresh):
        return symmetric_match(ref_corr, src_corr, thresh)

    def filter_match(self, ref_corr, ref_score, thresh):
        return filter_match(ref_corr, ref_score, thresh)
