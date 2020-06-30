import torch

def generate_pixel_grids(shape):
    '''returns W x H grid pixels

    Given width and height, creates a mesh grid, and returns homogeneous 
    coordinates
    of image in a 3 x W*H Tensor

    Arguments:
        width {Number} -- Number representing width of pixel grid image
        height {Number} -- Number representing height of pixel grid image

    Returns:
        torch.Tensor -- (HW)x2 tensor, oriented in H, W order
                        individual coords are oriented [x, y]
    '''
    H, W = shape
    # from 0.5 to w-0.5
    x_coords = torch.linspace(0.5, W - 0.5, W)
    # from 0.5 to h-0.5
    y_coords = torch.linspace(0.5, H - 0.5, H)
    y_grid_coords, x_grid_coords = torch.meshgrid([y_coords, x_coords])
    x_grid_coords = x_grid_coords.contiguous().view(-1)
    y_grid_coords = y_grid_coords.contiguous().view(-1)
    return torch.stack([x_grid_coords, y_grid_coords], dim=1)

def filter_match(corrs, scores, threshold):
    H, W, _ = corrs.shape
    c = corrs.clone().view(-1, 2)
    s = scores.view(-1)
    c[s < threshold] = float('nan')
    return c.view(H, W, 2)

def symmetric_match(ref_corrs, src_corrs, threshold=10):
    RH, RW, _ = ref_corrs.shape
    SH, SW, _ = src_corrs.shape
    dev = ref_corrs.device

    ref_p = generate_pixel_grids((RH, RW)).to(dev)

    ref_x = ref_corrs[:, :, 0].view(-1).long()
    ref_y = ref_corrs[:, :, 1].view(-1).long()
    c_ref_x = ref_x.clamp(0, SW-1)
    c_ref_x[c_ref_x != c_ref_x] = 0
    c_ref_y = ref_y.clamp(0, SH-1)
    c_ref_y[c_ref_y != c_ref_y] = 0

    # RHRWx2
    ref__src__ref_p = src_corrs[c_ref_y, c_ref_x]
    dists = ((ref__src__ref_p - ref_p) ** 2).sum(1)
    inliers = (dists < (threshold ** 2)) & \
              (ref_x == ref_x) & \
              (ref_y == ref_y) & \
              (0 <= ref_x) & (ref_x < SW) & \
              (0 <= ref_y) & (ref_y < SH)
            

    r = ref_corrs.clone().view(-1, 2)
    r[~inliers] = float('nan')

    return r.view(RH, RW, 2)

def compute_match_metrics(corrs, gt_corrs, thresholds=[3]):
    '''
    Arguments:
        corrs: HxWx2 correspondence
        gt_corrs: HxWx2 grounds truth correspondence
        thresholds: list of inlier values
    '''
    dev = corrs.device
    # flatten corrs
    corrs = corrs.view(-1, 2)
    gt_corrs = gt_corrs.view(-1, 2)

    ts = torch.FloatTensor(thresholds).to(dev).view(1, -1)

    # compute squared dists
    # HWx1
    sq_dists = ((corrs - gt_corrs) ** 2).sum(1).unsqueeze(1)

    # 
    tp = (sq_dists <= (ts ** 2)).sum(0).float()
    nm = (corrs == corrs).all(1).sum().float()
    tm = (gt_corrs == gt_corrs).all(1).sum().float()

    return {
        'precision': tp / nm,
        'recall': tp / tm,
        'num correct': tp,
        'num matches': nm,
        'num total': tm,
    }
