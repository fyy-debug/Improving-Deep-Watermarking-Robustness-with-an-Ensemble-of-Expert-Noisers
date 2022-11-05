import torch
import torch.nn.functional as F

def ns_loss(pred_fake, pred_real=None):
    bce = F.binary_cross_entropy_with_logits
    if pred_real is not None:
        loss_real = bce(pred_real, torch.ones_like(pred_real))
        loss_fake = bce(pred_fake, torch.zeros_like(pred_fake))
        return loss_fake, loss_real
    else:
        loss_fake = bce(pred_fake, torch.ones_like(pred_fake))
        return loss_fake


def wgan_loss(pred_fake, pred_real=None):
    if pred_real is not None:
        loss_real = -pred_real.mean()
        loss_fake = pred_fake.mean()
        return loss_fake, loss_real
    else:
        loss_fake = -pred_fake.mean()
        return loss_fake


def hinge_loss(pred_fake, pred_real=None):
    if pred_real is not None:
        loss_real = F.relu(1 - pred_real).mean()
        loss_fake = F.relu(1 + pred_fake).mean()
        return loss_fake, loss_real
    else:
        loss_fake = -pred_fake.mean()
        return loss_fake


loss_map = {
    'ns': ns_loss,
    'wgan': wgan_loss,
    'hinge': hinge_loss,
}