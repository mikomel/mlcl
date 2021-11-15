import torch
import torch.nn as nn
from torch.nn.functional import normalize


class MultiheadSupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07, num_heads=1):
        super(MultiheadSupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.num_heads = num_heads

    def forward(self, features: torch.Tensor, additional_negatives: torch.Tensor, labels: torch.Tensor):
        """
        Forward pass of the multi-head multi label supervised contrastive loss.
        Both features and additional_negatives should be tensors with logits (unnormalized).
        :param features: a Tensor with shape (batch_size, num_views, embedding_size)
        :param additional_negatives: additional negative samples, a Tensor with shape (batch_size_2, num_views, embedding_size)
        :param labels: a Tensor with multi-label class assignment with shape (batch_size, num_classes)
        :returns: loss as a single element Tensor
        """
        device = features.device
        batch_size, num_views, embedding_size = features.size()
        mask = (torch.matmul(labels, labels.T) > 0)

        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        additional_negatives = torch.cat(torch.unbind(additional_negatives, dim=1), dim=0)

        # split heads
        num_features, feature_size = features.size()
        num_additional_negatives = additional_negatives.size(0)
        feature_size = feature_size // self.num_heads
        features = features.view(num_features, self.num_heads, feature_size)
        additional_negatives = additional_negatives.view(num_additional_negatives, self.num_heads, feature_size)

        # normalize each head
        features = normalize(features, dim=-1)
        additional_negatives = normalize(additional_negatives, dim=-1)

        # compute logits
        dot_products = torch.matmul(
            features.view(-1, feature_size),
            features.view(-1, feature_size).T
        ).view(num_features, self.num_heads, num_features, self.num_heads).permute(0, 2, 1, 3)
        dot_products = dot_products.diagonal(dim1=2, dim2=3)
        heads = dot_products.argmax(dim=-1)

        dot_products_negatives = torch.matmul(
            features.view(-1, feature_size),
            additional_negatives.view(-1, feature_size).T
        ).view(num_features, self.num_heads, num_additional_negatives, self.num_heads).permute(0, 2, 1, 3)
        dot_products_negatives = dot_products_negatives.diagonal(dim1=2, dim2=3)

        # scale logits with temperature
        dot_products = dot_products / self.temperature
        dot_products_negatives = dot_products_negatives / self.temperature

        # choose dot products of appropriate heads
        h1 = heads.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, num_features)
        dot_products = dot_products.unsqueeze(dim=2).repeat(1, 1, num_features, 1)
        dot_products = dot_products.permute(0, 2, 3, 1)
        dot_products = dot_products.gather(dim=2, index=h1).squeeze()

        h2 = heads.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, num_additional_negatives)
        dot_products_negatives = dot_products_negatives.unsqueeze(dim=2).repeat(1, 1, num_features, 1)
        dot_products_negatives = dot_products_negatives.permute(0, 2, 3, 1)
        dot_products_negatives = dot_products_negatives.gather(dim=2, index=h2).squeeze()

        # for numerical stability
        logits_max, _ = dot_products.max(dim=2, keepdim=True)
        logits = dot_products - logits_max.detach()
        logits_max_negatives, _ = dot_products_negatives.max(dim=2, keepdim=True)
        logits_negatives = dot_products_negatives - logits_max_negatives.detach()

        # tile mask
        mask = mask.repeat(num_views, num_views)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask, device=device),
            1,
            torch.arange(batch_size * num_views, device=device).view(-1, 1),
            0
        )
        mask = mask * logits_mask

        # compute log_prob: log(exp(x1)/sum(exp(x2))) = log(exp(x1)) - log(sum(exp(x2))) = x1 - log(sum(exp(x2)))
        # logits of positive pairs lie on the diagonal
        logits_mask = logits_mask.unsqueeze(dim=1).repeat(1, num_features, 1)
        exp_logits = logits.exp() * logits_mask  # don't include anchor * anchor in the denominator
        log_prob = logits.diagonal(dim1=1, dim2=2) - (
                exp_logits.sum(2) + logits_negatives.exp().sum(2)
        ).log()

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        # When augmentation is not used (num_views == 1) and given batch contains an anchor without positive pairs, then
        # mask.sum(1)[anchor] will be 0 and therefore mean_log_prob_pos[anchor] will be nan
        # Such anchors are excluded from loss calculation
        loss = loss[loss.isfinite()].mean()
        if loss.isnan():
            loss = torch.zeros(1, device=loss.device)

        return loss


class MultiheadSupervisedContrastiveLossNoNegatives(nn.Module):
    """ Multi-Label Contrastive Loss without the term $\Sigma_{i,k}$ (incorrectly completed RPMs). """

    def __init__(self, temperature=0.07, base_temperature=0.07, num_heads=1):
        super(MultiheadSupervisedContrastiveLossNoNegatives, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.num_heads = num_heads

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Forward pass of the multi-head multi label supervised contrastive loss.
        Both features and incorrect_features should be tensors with logits (unnormalized).
        :param features: a Tensor with shape (batch_size, num_views, embedding_size)
        :param labels: a Tensor with multi-label class assignment with shape (batch_size, num_classes)
        :returns: loss as a single element Tensor
        """
        device = features.device
        batch_size, num_views, embedding_size = features.size()
        mask = (torch.matmul(labels, labels.T) > 0)

        features = torch.cat(torch.unbind(features, dim=1), dim=0)

        # split heads
        num_features, feature_size = features.size()
        feature_size = feature_size // self.num_heads
        features = features.view(num_features, self.num_heads, feature_size)

        # normalize each head
        features = normalize(features, dim=-1)

        # compute logits
        dot_products = torch.matmul(
            features.view(-1, feature_size),
            features.view(-1, feature_size).T
        ).view(num_features, self.num_heads, num_features, self.num_heads).permute(0, 2, 1, 3)
        dot_products = dot_products.diagonal(dim1=2, dim2=3)
        heads = dot_products.argmax(dim=-1)

        # scale logits with temperature
        dot_products = dot_products / self.temperature

        # choose dot products of appropriate heads
        h1 = heads.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, num_features)
        dot_products = dot_products.unsqueeze(dim=2).repeat(1, 1, num_features, 1)
        dot_products = dot_products.permute(0, 2, 3, 1)
        dot_products = dot_products.gather(dim=2, index=h1).squeeze()

        # for numerical stability
        logits_max, _ = dot_products.max(dim=2, keepdim=True)
        logits = dot_products - logits_max.detach()

        # tile mask
        mask = mask.repeat(num_views, num_views)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask, device=device),
            1,
            torch.arange(batch_size * num_views, device=device).view(-1, 1),
            0
        )
        mask = mask * logits_mask

        # compute log_prob: log(exp(x1)/sum(exp(x2))) = log(exp(x1)) - log(sum(exp(x2))) = x1 - log(sum(exp(x2)))
        # logits of positive pairs lie on the diagonal
        logits_mask = logits_mask.unsqueeze(dim=1).repeat(1, num_features, 1)
        exp_logits = logits.exp() * logits_mask  # don't include anchor * anchor in the denominator
        log_prob = logits.diagonal(dim1=1, dim2=2) - exp_logits.sum(2).log()

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        # When augmentation is not used (num_views == 1) and given batch contains an anchor without positive pairs, then
        # mask.sum(1)[anchor] will be 0 and therefore mean_log_prob_pos[anchor] will be nan
        # Such anchors are excluded from loss calculation
        return loss[loss.isfinite()].mean()
