import torch
from torch import nn
import torch.nn.functional as F
from functools import partial


def CORAL(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss

    return loss

def compute_pairwise_distances(x, y):

    if not x.dim() == y.dim() == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.size(1) != y.size(1):
        raise ValueError('The number of features should be the same.')

    def norm(x): return torch.sum(torch.pow(x, 2), 1)

    return torch.transpose(norm(torch.unsqueeze(x, 2) - torch.transpose(y, 0, 1)), 0, 1)


def gaussian_kernel_matrix(x, y, sigmas):

    beta = 1. / (2. * (torch.unsqueeze(sigmas, 1)))
    dist = compute_pairwise_distances(x, y)
    # print('dist shape={}'.format(dist.size()))
    s = torch.matmul(beta, dist.contiguous().view(1, -1))

    return torch.sum(torch.exp(-s), 0).view(*dist.size())


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):

    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))
    # We do not allow the loss to become negative.
    cost = torch.clamp(cost, min=0.0)

    return cost




def HOMM(xs, xt, order=3, num=300000):
    # print("HOMM called...")
    xs = xs - torch.mean(xs, 0)
    xt = xt - torch.mean(xt, 0)
    dim = xs.shape[1]
    index = torch.randint(0, dim - 1, (num, dim))
    index = index[:, :order]
    xs = xs.T
    xs = xs[index]  # dim=[num,order,batchsize]
    xt = xt.T
    xt = xt[index]
    HO_Xs = torch.prod(xs, 1)
    HO_Xs = torch.mean(HO_Xs, 1)
    HO_Xt = torch.prod(xt, 1)
    HO_Xt = torch.mean(HO_Xt, 1)
    return torch.mean((HO_Xs - HO_Xt)**2)

def MMD(xs, xt, kernel=gaussian_kernel_matrix):
    '''maximum mean discrepancy, a combination of multiple kernels'''
    sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5,
            10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
    gaussian_kernel = partial(gaussian_kernel_matrix,
                            sigmas=torch.Tensor(sigmas).float().cuda())
    loss_value = maximum_mean_discrepancy(xs, xt, kernel=gaussian_kernel)

    return torch.clamp(loss_value, min=1e-4)


def mask_correlated_samples(batch_size):
    mask = torch.ones((batch_size * 2, batch_size * 2), dtype=bool)
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i,batch_size + i] = 0
        mask [batch_size + i, i] = 0
    return mask

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x
class CdistContrastiveLoss(nn.Module):
    def __init__(self, temperature,normalize=True):
        super(CdistContrastiveLoss,self).__init__()
        self.tau = temperature
        self.normalize = normalize

    def forward(self, xi, xj):
        x = torch.cat((xi, xj), dim=0)

        is_cuda = x.is_cuda
        sim_mat = torch.cdist(x, x,p=2)
        if self.normalize:
            sim_mat_denom = torch.cdist(torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1),p=2)
            sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)

        sim_mat = torch.exp(sim_mat / self.tau)

        # no diag because it's not diffrentiable -> sum - exp(1 / tau)
        # diag_ind = torch.eye(xi.size(0) * 2).bool()
        # diag_ind = diag_ind.cuda() if use_cuda else diag_ind

        # sim_mat = sim_mat.masked_fill_(diag_ind, 0)

        # top
        if self.normalize:
            sim_mat_denom = torch.cdist(torch.norm(xi, dim=1), torch.norm(xj, dim=1),p=2)
            sim_match = torch.exp(torch.cdist(xi ,xj, p=2) / sim_mat_denom / self.tau)
        else:
            sim_match = torch.exp(torch.cidst(xi ,xj, p=2) / self.tau)

        sim_match = torch.cat((sim_match, sim_match), dim=0)

        norm_sum = torch.exp(torch.zeros(x.size(0)) / self.tau)
        norm_sum = norm_sum.cuda() if is_cuda else norm_sum
        loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))

        return loss

def tripletprotoLoss(xi,xj,centre,mask):
    loss = torch.sum(torch.abs(xi-centre)**2 - torch.abs(xj-centre)**2,dim=-1)
    if mask is not None:
        loss = (loss*mask).mean()
    else:
        loss = loss.mean()
    return loss
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature,normalize=True):
        super(ContrastiveLoss,self).__init__()
        self.tau = temperature
        self.normalize = normalize

    def forward(self, xi, xj):
        x = torch.cat((xi, xj), dim=0)

        is_cuda = x.is_cuda
        sim_mat = torch.mm(x, x.t())
        if self.normalize:
            sim_mat_denom = torch.mm(torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).t())
            sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)

        sim_mat = torch.exp(sim_mat / self.tau)

        # no diag because it's not diffrentiable -> sum - exp(1 / tau)
        # diag_ind = torch.eye(xi.size(0) * 2).bool()
        # diag_ind = diag_ind.cuda() if use_cuda else diag_ind

        # sim_mat = sim_mat.masked_fill_(diag_ind, 0)

        # top
        if self.normalize:
            sim_mat_denom = torch.norm(xi, dim=1) * torch.norm(xj, dim=1)
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / sim_mat_denom / self.tau)
        else:
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / self.tau)

        sim_match = torch.cat((sim_match, sim_match), dim=0)

        norm_sum = torch.exp(torch.ones(x.size(0)) / self.tau)
        norm_sum = norm_sum.cuda() if is_cuda else norm_sum
        loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))

        return loss
class NormalizeContrastiveLoss(nn.Module):
    def __init__(self, temperature,normalize=True):
        super(NormalizeContrastiveLoss,self).__init__()
        self.tau = temperature
        self.normalize = normalize

    def forward(self, xi, xj):
        xi = normalize(xi)
        xj = normalize(xj)
        x = torch.cat((xi, xj), dim=0)

        is_cuda = x.is_cuda
        sim_mat = torch.mm(x, x.t())
        if self.normalize:
            sim_mat_denom = torch.mm(torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).t())
            sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)

        sim_mat = torch.exp(sim_mat / self.tau)

        # no diag because it's not diffrentiable -> sum - exp(1 / tau)
        # diag_ind = torch.eye(xi.size(0) * 2).bool()
        # diag_ind = diag_ind.cuda() if use_cuda else diag_ind

        # sim_mat = sim_mat.masked_fill_(diag_ind, 0)

        # top
        if self.normalize:
            sim_mat_denom = torch.norm(xi, dim=1) * torch.norm(xj, dim=1)
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / sim_mat_denom / self.tau)
        else:
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / self.tau)

        sim_match = torch.cat((sim_match, sim_match), dim=0)

        norm_sum = torch.exp(torch.ones(x.size(0)) / self.tau)
        norm_sum = norm_sum.cuda() if is_cuda else norm_sum
        loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))

        return loss

def newsimclr(xi,xj):
        x = torch.cat((xi, xj), dim=0)

        is_cuda = x.is_cuda
        sim_mat = torch.mm(x, x.t())
        sim_mat = torch.exp(sim_mat / 0.05)
        loss = torch.mean(-torch.log(sim_mat/(1+sim_mat)))
        return loss

class FocalContrastiveLoss(nn.Module):
    def __init__(self, temperature,normalize=True,gamma=2):
        super(FocalContrastiveLoss,self).__init__()
        self.tau = temperature
        self.normalize = normalize
        self.gamma=gamma

    def forward(self, xi, xj):

        x = torch.cat((xi, xj), dim=0)

        is_cuda = x.is_cuda
        sim_mat = torch.mm(x, x.t())
        if self.normalize:
            sim_mat_denom = torch.mm(torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).t())
            sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)

        sim_mat = torch.exp(( ((1 - sim_mat) ** self.gamma) *sim_mat) / self.tau)

        # no diag because it's not diffrentiable -> sum - exp(1 / tau)
        # diag_ind = torch.eye(xi.size(0) * 2).bool()
        # diag_ind = diag_ind.cuda() if use_cuda else diag_ind

        # sim_mat = sim_mat.masked_fill_(diag_ind, 0)

        # top
        if self.normalize:
            sim_mat_denom = torch.norm(xi, dim=1) * torch.norm(xj, dim=1)
            temp= torch.sum(xi * xj, dim=-1) / sim_mat_denom 
            sim_match = torch.exp(( ((1 - temp) ** self.gamma) *temp)/ self.tau)
        else:
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / self.tau)

        sim_match = torch.cat((sim_match, sim_match), dim=0)

        norm_sum = torch.exp((((1 - torch.ones(x.size(0))) ** self.gamma) *torch.ones(x.size(0))) / self.tau)
        norm_sum = norm_sum.cuda() if is_cuda else norm_sum
        loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))

        return loss

class SupConLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super(SupConLoss,self).__init__()
        self.register_buffer("temperature", torch.tensor(temperature).cuda())
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2,dtype=torch.uint8)).float().cuda())
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        batch_size = emb_i.shape[0]
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss

class GroupSupConLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super(SupConLoss,self).__init__()
        self.register_buffer("temperature", torch.tensor(temperature).cuda())
        #self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2,dtype=torch.uint8)).float().cuda())
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        batch_size = emb_i.shape[0]
        negatives_mask= (~torch.eye(batch_size * 2, batch_size * 2,dtype=torch.uint8)).float().cuda()
          
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss


class SupervisedConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupervisedConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
