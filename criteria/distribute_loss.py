import torch.nn as nn
import numpy
import torch

sfm = nn.Softmax(dim=1)
kl_loss = nn.KLDivLoss()
sim = nn.CosineSimilarity()

def js_div(p_output, q_output):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    log_mean_output = ((p_output + q_output )/2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2

def js_divloss(pred, target):
    pred = sfm(pred)
    target = sfm(target)
    return js_div(pred,target)

def distribution_loss(feat_source, feat_target, batch=2, feat_const_batch=3, kl_wt=1):
    # feat_const_batch=batch=中间向量个数=len(feat_source)
    n_latent = feat_const_batch
    feat_ind = numpy.random.randint(0, n_latent, size=feat_const_batch)
    batch_ind = numpy.random.randint(0, batch, size=feat_const_batch)
    dist_source = torch.zeros([feat_const_batch, feat_const_batch - 1]).cuda()
    dist_target = torch.zeros([feat_const_batch, feat_const_batch - 1]).cuda()
    with torch.set_grad_enabled(False):
        for pair1 in range(feat_const_batch):
            tmpc = 0
            # comparing the possible pairs
            for pair2 in range(feat_const_batch):
                if batch_ind[pair1] != batch_ind[pair2]:
                    #  feat_source[feat_ind[pair1]]选择某一中间层，feat_source[feat_ind[pair1]][pair1]选择某一中间层batch中的某一个
                    anchor_feat = torch.unsqueeze(
                        feat_source[feat_ind[pair1]][batch_ind[pair1]].reshape(-1), 0)
                    compare_feat = torch.unsqueeze(
                        feat_source[feat_ind[pair1]][batch_ind[pair2]].reshape(-1), 0)
                    dist_source[pair1, tmpc] = sim(
                        anchor_feat, compare_feat)
                    tmpc += 1
        dist_source = sfm(dist_source)  # [3, 2]
    for pair1 in range(feat_const_batch):
        tmpc = 0
        for pair2 in range(feat_const_batch):  # comparing the possible pairs
            if batch_ind[pair1] != batch_ind[pair2]:
                anchor_feat = torch.unsqueeze(
                    feat_target[feat_ind[pair1]][batch_ind[pair1]].reshape(-1), 0)
                compare_feat = torch.unsqueeze(
                    feat_target[feat_ind[pair1]][batch_ind[pair2]].reshape(-1), 0)
                dist_target[pair1, tmpc] = sim(anchor_feat, compare_feat)
                tmpc += 1
    dist_target = sfm(dist_target)  # [3, 2]
    rel_loss = kl_wt * kl_loss(torch.log(dist_target), dist_source)
    return rel_loss

# noise1 = [torch.rand(size=[8, 512, 64, 64]),torch.rand(size=[8, 512, 32, 32]),torch.rand(size=[8, 512, 16, 16])]
# noise2 = [torch.rand(size=[8, 512, 64, 64]),torch.rand(size=[8, 512, 32, 32]),torch.rand(size=[8, 512, 16, 16])]
#
# loss = distribution_loss(noise1,noise2)
# print(loss)