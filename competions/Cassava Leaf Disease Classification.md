# Cassava Leaf Disease Classification 参赛总结

这个比赛是最基本的分类任务，对叶片疾病进行分类，主要涉及 fine-grained ,Learning with noisy labels领域。

这个比赛主要是实验不同的loss和数据增强方法，通过Ensemble 来对抗shake-up，ViT 在这个比赛中的表现很好。



### Mixup(ICLR 2018)//Cutmix(ICCV 2019)/SnapMix(AAAI 2021)

[Code](#aug)

[Mixup](https://arxiv.org/pdf/1710.09412.pdf) [Cutout](https://arxiv.org/pdf/1708.04552.pdf) [Cutmix](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.pdf)  [SnapMix](https://arxiv.org/pdf/2012.04846.pdf)

Mixup:将随机的两张样本按比例混合，分类的结果按比例分配；

Cutout:随机的将样本中的部分区域cut掉，并且填充0像素值，分类的结果不变；

CutMix:就是将一部分区域cut掉但不填充0像素而是随机填充训练集中的其他数据的区域像素值，分类结果按一定的比例分配

但是在细粒度分类任务中，Mixup和Cutmix单纯的从像素级别进行混合存在不合理的地方，下左图所示，SnapMix 从语义图中进行混合

![image-20210617211156448](C:\Users\86138\Desktop\参赛总结\images\snapMix.png)




### label smoothing

标签软化更利于学习,在YOLO,Inception等论文中都用到了这项技术

label smoothing公式:

​            $$ q=(k|x) = (i - \epsilon)\delta_{k,y} + \epsilon\mu(k)  $$

代码：

```
class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # 此处的self.smoothing即我们的epsilon平滑参数。

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
```



### Attention Learning in CV

[Attention Discussion](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/203083)



### ViT model(ICLR 2021)

[paper](https://arxiv.org/pdf/2010.11929.pdf)



### cleanlab 置信学习(ICML 2020)

[paper](https://arxiv.org/pdf/1911.00068.pdf)

[notebook](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/218269)

采用预测标签和真实标签的联合分布来过滤噪声样本。

```python
if args.get_noise_indices:
    # Prediction
    # preds_df, avg_preds = get_scores_train(folds=[], base_model=''):
    preds_df, psx = get_scores_train(['./models/seresnet50/seresnet50_extra_2.pth'])

    labels = preds_df[TARGET_COL].values

    ordered_label_errors = get_noise_indices(
        s=labels, # label_true
        psx=psx, # label_preds
        sorted_index_method='normalized_margin',  # Orders label errors
    )
```

![image-20210617172037872](C:\Users\86138\Desktop\参赛总结\images\cassava_1.png)



### losses 

 [Bi-Tempered Logistic Loss(NIPS 2019)](#bitloss)    [paper](https://papers.nips.cc/paper/2019/file/8cd7775f9129da8b5bf787a063d8426e-Paper.pdf)

 [Taylor Cross Entropy loss(IJCAI 2020)](#taylorloss)     [paper](https://www.ijcai.org/Proceedings/2020/0305.pdf)

 [Focal Loss (ICCV 2017)](#floss)     [paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf) 
 Focal Loss原本提出来用于目标检测生成的区域框中正负样本极度不平衡的情况,也可以拓展到分类任务中，来解决样本分布不均衡的情况。



###  <span id="bitloss"> Bi-Tempered Logistic Loss(NIPS 2019)</span>

公式：

![image-20210617203051232](C:\Users\86138\Desktop\参赛总结\images\Bi-Tempered_loss.png)

代码：

```python
def log_t(u, t):
    """Compute log_t for `u'."""
    if t==1.0:
        return u.log()
    else:
        return (u.pow(1.0 - t) - 1.0) / (1.0 - t)

def exp_t(u, t):
    """Compute exp_t for `u'."""
    if t==1:
        return u.exp()
    else:
        return (1.0 + (1.0-t)*u).relu().pow(1.0 / (1.0 - t))

def compute_normalization_fixed_point(activations, t, num_iters):

    """Returns the normalization value for each example (t > 1.0).
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same shape as activation with the last dimension being 1.
    """
    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations_step_0 = activations - mu

    normalized_activations = normalized_activations_step_0

    for _ in range(num_iters):
        logt_partition = torch.sum(
                exp_t(normalized_activations, t), -1, keepdim=True)
        normalized_activations = normalized_activations_step_0 * \
                logt_partition.pow(1.0-t)

    logt_partition = torch.sum(
            exp_t(normalized_activations, t), -1, keepdim=True)
    normalization_constants = - log_t(1.0 / logt_partition, t) + mu

    return normalization_constants

def compute_normalization_binary_search(activations, t, num_iters):

    """Returns the normalization value for each example (t < 1.0).
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (< 1.0 for finite support).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """

    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations = activations - mu

    effective_dim = \
        torch.sum(
                (normalized_activations > -1.0 / (1.0-t)).to(torch.int32),
            dim=-1, keepdim=True).to(activations.dtype)

    shape_partition = activations.shape[:-1] + (1,)
    lower = torch.zeros(shape_partition, dtype=activations.dtype, device=activations.device)
    upper = -log_t(1.0/effective_dim, t) * torch.ones_like(lower)

    for _ in range(num_iters):
        logt_partition = (upper + lower)/2.0
        sum_probs = torch.sum(
                exp_t(normalized_activations - logt_partition, t),
                dim=-1, keepdim=True)
        update = (sum_probs < 1.0).to(activations.dtype)
        lower = torch.reshape(
                lower * update + (1.0-update) * logt_partition,
                shape_partition)
        upper = torch.reshape(
                upper * (1.0 - update) + update * logt_partition,
                shape_partition)

    logt_partition = (upper + lower)/2.0
    return logt_partition + mu

class ComputeNormalization(torch.autograd.Function):
    """
    Class implementing custom backward pass for compute_normalization. See compute_normalization.
    """
    @staticmethod
    def forward(ctx, activations, t, num_iters):
        if t < 1.0:
            normalization_constants = compute_normalization_binary_search(activations, t, num_iters)
        else:
            normalization_constants = compute_normalization_fixed_point(activations, t, num_iters)

        ctx.save_for_backward(activations, normalization_constants)
        ctx.t=t
        return normalization_constants

    @staticmethod
    def backward(ctx, grad_output):
        activations, normalization_constants = ctx.saved_tensors
        t = ctx.t
        normalized_activations = activations - normalization_constants 
        probabilities = exp_t(normalized_activations, t)
        escorts = probabilities.pow(t)
        escorts = escorts / escorts.sum(dim=-1, keepdim=True)
        grad_input = escorts * grad_output
        
        return grad_input, None, None

def compute_normalization(activations, t, num_iters=5):
    """Returns the normalization value for each example. 
    Backward pass is implemented.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """
    return ComputeNormalization.apply(activations, t, num_iters)

def tempered_sigmoid(activations, t, num_iters = 5):
    """Tempered sigmoid function.
    Args:
      activations: Activations for the positive class for binary classification.
      t: Temperature tensor > 0.0.
      num_iters: Number of iterations to run the method.
    Returns:
      A probabilities tensor.
    """
    internal_activations = torch.stack([activations,
        torch.zeros_like(activations)],
        dim=-1)
    internal_probabilities = tempered_softmax(internal_activations, t, num_iters)
    return internal_probabilities[..., 0]


def tempered_softmax(activations, t, num_iters=5):
    """Tempered softmax function.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature > 1.0.
      num_iters: Number of iterations to run the method.
    Returns:
      A probabilities tensor.
    """
    if t == 1.0:
        return activations.softmax(dim=-1)

    normalization_constants = compute_normalization(activations, t, num_iters)
    return exp_t(activations - normalization_constants, t)

def bi_tempered_binary_logistic_loss(activations,
        labels,
        t1,
        t2,
        label_smoothing = 0.0,
        num_iters=5,
        reduction='mean'):

    """Bi-Tempered binary logistic loss.
    Args:
      activations: A tensor containing activations for class 1.
      labels: A tensor with shape as activations, containing probabilities for class 1
      t1: Temperature 1 (< 1.0 for boundedness).
      t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      label_smoothing: Label smoothing
      num_iters: Number of iterations to run the method.
    Returns:
      A loss tensor.
    """
    internal_activations = torch.stack([activations,
        torch.zeros_like(activations)],
        dim=-1)
    internal_labels = torch.stack([labels.to(activations.dtype),
        1.0 - labels.to(activations.dtype)],
        dim=-1)
    return bi_tempered_logistic_loss(internal_activations, 
            internal_labels,
            t1,
            t2,
            label_smoothing = label_smoothing,
            num_iters = num_iters,
            reduction = reduction)

def bi_tempered_logistic_loss(activations,
        labels,
        t1,
        t2,
        label_smoothing=0.0,
        num_iters=5,
        reduction = 'mean'):

    """Bi-Tempered Logistic Loss.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      labels: A tensor with shape and dtype as activations (onehot), 
        or a long tensor of one dimension less than activations (pytorch standard)
      t1: Temperature 1 (< 1.0 for boundedness).
      t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      label_smoothing: Label smoothing parameter between [0, 1). Default 0.0.
      num_iters: Number of iterations to run the method. Default 5.
      reduction: ``'none'`` | ``'mean'`` | ``'sum'``. Default ``'mean'``.
        ``'none'``: No reduction is applied, return shape is shape of
        activations without the last dimension.
        ``'mean'``: Loss is averaged over minibatch. Return shape (1,)
        ``'sum'``: Loss is summed over minibatch. Return shape (1,)
    Returns:
      A loss tensor.
    """

    if len(labels.shape)<len(activations.shape): #not one-hot
        labels_onehot = torch.zeros_like(activations)
        labels_onehot.scatter_(1, labels[..., None], 1)
    else:
        labels_onehot = labels

    if label_smoothing > 0:
        num_classes = labels_onehot.shape[-1]
        labels_onehot = ( 1 - label_smoothing * num_classes / (num_classes - 1) ) \
                * labels_onehot + \
                label_smoothing / (num_classes - 1)

    probabilities = tempered_softmax(activations, t2, num_iters)

    loss_values = labels_onehot * log_t(labels_onehot + 1e-10, t1) \
            - labels_onehot * log_t(probabilities, t1) \
            - labels_onehot.pow(2.0 - t1) / (2.0 - t1) \
            + probabilities.pow(2.0 - t1) / (2.0 - t1)
    loss_values = loss_values.sum(dim = -1) #sum over classes

    if reduction == 'none':
        return loss_values
    if reduction == 'sum':
        return loss_values.sum()
    if reduction == 'mean':
        return loss_values.mean()
    
class BiTemperedLogisticLoss(nn.Module): 
    def __init__(self, t1, t2, smoothing=0.0): 
        super(BiTemperedLogisticLoss, self).__init__() 
        self.t1 = t1
        self.t2 = t2
        self.smoothing = smoothing
    def forward(self, logit_label, truth_label):
        loss_label = bi_tempered_logistic_loss(
            logit_label, truth_label,
            t1=self.t1, t2=self.t2,
            label_smoothing=self.smoothing,
            reduction='none'
        )
        
        loss_label = loss_label.mean()
        return loss_label
```



###  <span id="floss">Focal Loss (ICCV 2017)</span>

Focal Loss原本提出来用于目标检测生成的区域框中正负样本极度不平衡的情况,也可以拓展到分类任务中，来解决样本分布不均衡的情况。

公式：

​                     $$ FL(p_{t}) = - (1 - p_{t})^{\gamma} log(p_{t}) $$

代码：

```
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss()(inputs, targets)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
```


###  <span id="taylorloss">Taylor Cross Entropy loss(IJCAI 2020)</span>



代码：

```
class TaylorSoftmax(nn.Module):
    '''
    This is the autograd version
    '''
    def __init__(self, dim=1, n=2):
        super(TaylorSoftmax, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        '''
        usage similar to nn.Softmax:
            >>> mod = TaylorSoftmax(dim=1, n=4)
            >>> inten = torch.randn(1, 32, 64, 64)
            >>> out = mod(inten)
        '''
        fn = torch.ones_like(x)
        denor = 1.
        for i in range(1, self.n+1):
            denor *= i
            fn = fn + x.pow(i) / denor
        out = fn / fn.sum(dim=self.dim, keepdims=True)
        return out

    
class TaylorCrossEntropyLoss(nn.Module):
    def __init__(self, n=2, ignore_index=-1, reduction='mean', smoothing=0.05):
        super(TaylorCrossEntropyLoss, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = TaylorSoftmax(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lab_smooth = LabelSmoothingLoss(CFG.target_size, smoothing=smoothing)

    def forward(self, logits, labels):
        log_probs = self.taylor_softmax(logits).log()
        #loss = F.nll_loss(log_probs, labels, reduction=self.reduction,
        #        ignore_index=self.ignore_index)
        loss = self.lab_smooth(log_probs, labels)
        return loss

```



###  <span id="aug">Augmentation Code</span>

```
import torch
import torch.nn as nn
import imp
import numpy as np
import utils
import os
import torch.nn.functional as F
import random
import copy

def get_spm(input,target,conf,model):

    imgsize = (conf.cropsize,conf.cropsize)
    bs = input.size(0)
    with torch.no_grad():
        output,fms,_ = model(input)
        if 'inception' in conf.netname:
            clsw = model.module.fc
        else:
            clsw = model.module.classifier
        weight = clsw.weight.data
        bias = clsw.bias.data
        weight = weight.view(weight.size(0),weight.size(1),1,1)
        fms = F.relu(fms)
        poolfea = F.adaptive_avg_pool2d(fms,(1,1)).squeeze()
        clslogit = F.softmax(clsw.forward(poolfea))
        logitlist = []
        for i in range(bs):
            logitlist.append(clslogit[i,target[i]])
        clslogit = torch.stack(logitlist)

        out = F.conv2d(fms, weight, bias=bias)

        outmaps = []
        for i in range(bs):
            evimap = out[i,target[i]]
            outmaps.append(evimap)

        outmaps = torch.stack(outmaps)
        if imgsize is not None:
            outmaps = outmaps.view(outmaps.size(0),1,outmaps.size(1),outmaps.size(2))
            outmaps = F.interpolate(outmaps,imgsize,mode='bilinear',align_corners=False)

        outmaps = outmaps.squeeze()

        for i in range(bs):
            outmaps[i] -= outmaps[i].min()
            outmaps[i] /= outmaps[i].sum()


    return outmaps,clslogit



def snapmix(input,target,conf,model=None):

    r = np.random.rand(1)
    lam_a = torch.ones(input.size(0))
    lam_b = 1 - lam_a
    target_b = target.clone()

    if r < conf.prob:
        wfmaps,_ = get_spm(input,target,conf,model)
        bs = input.size(0)
        lam = np.random.beta(conf.beta, conf.beta)
        lam1 = np.random.beta(conf.beta, conf.beta)
        rand_index = torch.randperm(bs).cuda()
        wfmaps_b = wfmaps[rand_index,:,:]
        target_b = target[rand_index]

        same_label = target == target_b
        bbx1, bby1, bbx2, bby2 = utils.rand_bbox(input.size(), lam)
        bbx1_1, bby1_1, bbx2_1, bby2_1 = utils.rand_bbox(input.size(), lam1)

        area = (bby2-bby1)*(bbx2-bbx1)
        area1 = (bby2_1-bby1_1)*(bbx2_1-bbx1_1)

        if  area1 > 0 and  area>0:
            ncont = input[rand_index, :, bbx1_1:bbx2_1, bby1_1:bby2_1].clone()
            ncont = F.interpolate(ncont, size=(bbx2-bbx1,bby2-bby1), mode='bilinear', align_corners=True)
            input[:, :, bbx1:bbx2, bby1:bby2] = ncont
            lam_a = 1 - wfmaps[:,bbx1:bbx2,bby1:bby2].sum(2).sum(1)/(wfmaps.sum(2).sum(1)+1e-8)
            lam_b = wfmaps_b[:,bbx1_1:bbx2_1,bby1_1:bby2_1].sum(2).sum(1)/(wfmaps_b.sum(2).sum(1)+1e-8)
            tmp = lam_a.clone()
            lam_a[same_label] += lam_b[same_label]
            lam_b[same_label] += tmp[same_label]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            lam_a[torch.isnan(lam_a)] = lam
            lam_b[torch.isnan(lam_b)] = 1-lam

    return input,target,target_b,lam_a.cuda(),lam_b.cuda()


def as_cutmix(input,target,conf,model=None):

    r = np.random.rand(1)
    lam_a = torch.ones(input.size(0))
    lam_b = 1 - lam_a
    target_b = target.clone()

    if r < conf.prob:
        bs = input.size(0)
        lam = np.random.beta(conf.beta, conf.beta)
        rand_index = torch.randperm(bs).cuda()
        target_b = target[rand_index]

        bbx1, bby1, bbx2, bby2 = utils.rand_bbox(input.size(), lam)
        bbx1_1, bby1_1, bbx2_1, bby2_1 = utils.rand_bbox(input.size(), lam)

        if (bby2_1-bby1_1)*(bbx2_1-bbx1_1) > 4 and  (bby2-bby1)*(bbx2-bbx1)>4:
            ncont = input[rand_index, :, bbx1_1:bbx2_1, bby1_1:bby2_1].clone()
            ncont = F.interpolate(ncont, size=(bbx2-bbx1,bby2-bby1), mode='bilinear', align_corners=True)
            input[:, :, bbx1:bbx2, bby1:bby2] = ncont
            # adjust lambda to exactly match pixel ratio
            lam_a = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            lam_a *= torch.ones(input.size(0))
    lam_b = 1 - lam_a

    return input,target,target_b,lam_a.cuda(),lam_b.cuda()

def cutmix(input,target,conf,model=None):

    r = np.random.rand(1)
    lam_a = torch.ones(input.size(0)).cuda()
    target_b = target.clone()

    if r < conf.prob:
        bs = input.size(0)
        lam = np.random.beta(conf.beta, conf.beta)
        rand_index = torch.randperm(bs).cuda()
        target_b = target[rand_index]
        input_b = input[rand_index].clone()
        bbx1, bby1, bbx2, bby2 = utils.rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = input_b[:, :, bbx1:bbx2, bby1:bby2]

        # adjust lambda to exactly match pixel ratio
        lam_a = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
        lam_a *= torch.ones(input.size(0))

    lam_b = 1 - lam_a


    return input,target,target_b,lam_a.cuda(),lam_b.cuda()



def cutout(input,target,conf=None,model=None):

    r = np.random.rand(1)
    lam = torch.ones(input.size(0)).cuda()
    target_b = target.clone()
    lam_a = lam
    lam_b = 1-lam

    if r < conf.prob:
        bs = input.size(0)
        lam = 0.75
        bbx1, bby1, bbx2, bby2 = utils.rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = 0

    return input,target,target_b,lam_a.cuda(),lam_b.cuda()


def mixup(input,target,conf,model=None):
    r = np.random.rand(1)
    lam_a = torch.ones(input.size(0)).cuda()
    bs = input.size(0)
    target_a = target
    target_b = target

    if r < conf.prob:
        rand_index = torch.randperm(bs).cuda()
        target_b = target[rand_index]
        lam = np.random.beta(conf.beta, conf.beta)
        lam_a = lam_a*lam
        input = input * lam + input[rand_index] * (1-lam)

    lam_b = 1 - lam_a

    return input,target,target_b,lam_a.cuda(),lam_b.cuda()

```

