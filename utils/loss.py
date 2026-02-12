import torch

class WeightedGDLMSE(torch.nn.Module):
    def __init__(
            self, 
            weights=None, 
            alpha=2,
            lambda_gdl=1,
            lambda_mse=1,
            calc_z = True):
        super(WeightedGDLMSE, self).__init__()
        self.weights = weights
        self.alpha = alpha
        self.lambda_gdl = lambda_gdl
        self.lambda_mse = lambda_mse
        self.calc_z = calc_z

    def forward(self, pred, gt, weights=None):
        if weights is None:
            weights = self.weights
        if self.lambda_mse > 0:
            MSE = weights * (pred - gt) ** 2
            if len(torch.nonzero(MSE)) != 0:
                mask = torch.masked_select(MSE, torch.gt(weights, 0))
                MSE_loss = torch.mean(mask)
            else:
                MSE_loss = torch.mean(MSE)
        else:
            MSE_loss = 0.0
        if self.lambda_gdl > 0:
            grad_pred_y = torch.abs(pred[:, :, :,  1:, :] - pred[:, :, :, :-1, :])
            grad_gt_y   = torch.abs(gt[:, :, :, 1:, :] - gt[:, :, :, :-1, :])
            grad_diff_y = torch.abs(grad_pred_y - grad_gt_y) ** self.alpha

            grad_pred_x = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])
            grad_gt_x = torch.abs(gt[:, :, :, :, 1:] - gt[:, :, :, :, :-1])
            grad_diff_x = torch.abs(grad_pred_x - grad_gt_x) ** self.alpha
            if self.calc_z:
                grad_pred_z = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :])
                grad_gt_z = torch.abs(gt[:, :, 1:, :, :] - gt[:, :, :-1, :, :])
                grad_diff_z = torch.abs(grad_pred_z - grad_gt_z) ** self.alpha

            if weights is not None:
                weight_y = weights[:,:,:,1:,:]
                weight_x = weights[:,:,:,:,1:]
                weight_z = weights[:,:,1:,:,:]
                grad_diff_y = grad_diff_y * weight_y
                grad_diff_x = grad_diff_x * weight_x
                if self.calc_z:
                    grad_diff_z = grad_diff_z * weight_z
            if self.calc_z:
                loss_gdl = torch.mean(grad_diff_x) + torch.mean(grad_diff_y) + torch.mean(grad_diff_z)
            else:
                loss_gdl = torch.mean(grad_diff_x) + torch.mean(grad_diff_y)
        else:
            loss_gdl = 0.0

        loss = self.lambda_mse*MSE_loss + self.lambda_gdl*loss_gdl

        return loss

class GradientDifferenceLoss(torch.nn.Module):
    def __init__(self, weights=None, alpha=2):
        super(GradientDifferenceLoss, self).__init__()
        self.weights = weights
        self.alpha = alpha  # Controls the exponent as in the paper

    def forward(self, pred, gt, weights=None): 
        """
        Computes the Gradient Difference Loss (GDL) as defined in https://arxiv.org/abs/1511.05440

        Args:
            pred (torch.Tensor): Predicted image tensor (B, C, Z, Y, X)
            gt   (torch.Tensor): Ground truth image tensor (B, C, Z, Y, X)

        Returns:
            torch.Tensor: GDL loss
        """
        if weights is None:
            weights = self.weights

        grad_pred_y = torch.abs(pred[:, :, :,  1:, :] - pred[:, :, :, :-1, :])
        grad_gt_y   = torch.abs(gt[:, :, :, 1:, :] - gt[:, :, :, :-1, :])
        grad_diff_y = torch.abs(grad_pred_y - grad_gt_y) ** self.alpha

        grad_pred_x = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])
        grad_gt_x = torch.abs(gt[:, :, :, :, 1:] - gt[:, :, :, :, :-1])
        grad_diff_x = torch.abs(grad_pred_x - grad_gt_x) ** self.alpha
        
        if weights is not None:
            weight_y = weights[:,:,:,1:,:]
            weight_x = weights[:,:,:,:,1:]
            grad_diff_y = grad_diff_y * weight_y
            grad_diff_x = grad_diff_x * weight_x

        loss_gdl = torch.mean(grad_diff_x) + torch.mean(grad_diff_y)
        return loss_gdl