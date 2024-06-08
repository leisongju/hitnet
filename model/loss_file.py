import torch
import torch.nn as nn
from torch.nn import L1Loss
from kornia.filters import  gaussian_blur2d
import torch.nn.functional as F
from kornia.losses import TotalVariation





"""
Sampling strategies: RS (Random Sampling), EGS (Edge-Guided Sampling), and IGS (Instance-Guided Sampling)
"""
###########
# RANDOM SAMPLING
# input:
# inputs[i,:], targets[i, :], masks[i, :], self.mask_value, self.point_pairs
# return:
# inputs_A, inputs_B, targets_A, targets_B, consistent_masks_A, consistent_masks_B
###########
def randomSampling(inputs, targets, masks, threshold, sample_num):

    # find A-B point pairs from predictions
    inputs_index = torch.masked_select(inputs, targets.gt(threshold))
    num_effect_pixels = len(inputs_index)
    shuffle_effect_pixels = torch.randperm(num_effect_pixels).cuda()
    inputs_A = inputs_index[shuffle_effect_pixels[0:sample_num*2:2]]
    inputs_B = inputs_index[shuffle_effect_pixels[1:sample_num*2:2]]
    # find corresponding pairs from GT
    target_index = torch.masked_select(targets, targets.gt(threshold))
    targets_A = target_index[shuffle_effect_pixels[0:sample_num*2:2]]
    targets_B = target_index[shuffle_effect_pixels[1:sample_num*2:2]]
    # only compute the losses of point pairs with valid GT
    consistent_masks_index = torch.masked_select(masks, targets.gt(threshold))
    consistent_masks_A = consistent_masks_index[shuffle_effect_pixels[0:sample_num*2:2]]
    consistent_masks_B = consistent_masks_index[shuffle_effect_pixels[1:sample_num*2:2]]

    # The amount of A and B should be the same!!
    if len(targets_A) > len(targets_B):
        targets_A = targets_A[:-1]
        inputs_A = inputs_A[:-1]
        consistent_masks_A = consistent_masks_A[:-1]

    return inputs_A, inputs_B, targets_A, targets_B, consistent_masks_A, consistent_masks_B

###########
# EDGE-GUIDED SAMPLING
# input:
# inputs[i,:], targets[i, :], masks[i, :], edges_img[i], thetas_img[i], masks[i, :], h, w
# return:
# inputs_A, inputs_B, targets_A, targets_B, masks_A, masks_B
###########
def ind2sub(idx, cols):
    r = idx / cols
    c = idx - r * cols
    return r, c

def sub2ind(r, c, cols):
    idx = r * cols + c
    return idx

def edgeGuidedSampling(inputs, targets, edges_img, thetas_img, masks, h, w):

    # find edges
    edges_max = edges_img.max()
    edges_mask = edges_img.ge(edges_max*0.1)
    edges_loc = edges_mask.nonzero()

    inputs_edge = torch.masked_select(inputs, edges_mask)
    targets_edge = torch.masked_select(targets, edges_mask)
    thetas_edge = torch.masked_select(thetas_img, edges_mask)
    minlen = inputs_edge.size()[0]

    # find anchor points (i.e, edge points)
    sample_num = minlen
    index_anchors = torch.randint(0, minlen, (sample_num,), dtype=torch.long).cuda()
    anchors = torch.gather(inputs_edge, 0, index_anchors)
    theta_anchors = torch.gather(thetas_edge, 0, index_anchors)
    row_anchors, col_anchors = ind2sub(edges_loc[index_anchors].squeeze(1), w)
    ## compute the coordinates of 4-points,  distances are from [2, 30]
    distance_matrix = torch.randint(2, 31, (4,sample_num)).cuda()
    pos_or_neg = torch.ones(4, sample_num).cuda()
    pos_or_neg[:2,:] = -pos_or_neg[:2,:]
    distance_matrix = distance_matrix.float() * pos_or_neg
    col = col_anchors.unsqueeze(0).expand(4, sample_num).long() + torch.round(distance_matrix.double() * torch.cos(theta_anchors).unsqueeze(0)).long()
    row = row_anchors.unsqueeze(0).expand(4, sample_num).long() + torch.round(distance_matrix.double() * torch.sin(theta_anchors).unsqueeze(0)).long()

    # constrain 0=<c<=w, 0<=r<=h
    # Note: index should minus 1
    col[col<0] = 0
    col[col>w-1] = w-1
    row[row<0] = 0
    row[row>h-1] = h-1

    # a-b, b-c, c-d
    a = sub2ind(row[0,:], col[0,:], w)
    b = sub2ind(row[1,:], col[1,:], w)
    c = sub2ind(row[2,:], col[2,:], w)
    d = sub2ind(row[3,:], col[3,:], w)
    A = torch.cat((a,b,c), 0)
    B = torch.cat((b,c,d), 0)

    inputs_A = torch.gather(inputs, 0, A.long())
    inputs_B = torch.gather(inputs, 0, B.long())
    targets_A = torch.gather(targets, 0, A.long())
    targets_B = torch.gather(targets, 0, B.long())
    masks_A = torch.gather(masks, 0, A.long())
    masks_B = torch.gather(masks, 0, B.long())

    return inputs_A, inputs_B, targets_A, targets_B, masks_A, masks_B, sample_num

######################################################
# EdgeguidedRankingLoss (with regularization term)
# Please comment regularization_loss if you don't want to use multi-scale gradient matching term
#####################################################
class EdgeguidedRankingLoss(nn.Module):
    def __init__(self, point_pairs=10000, sigma=0.03, alpha=1.0, mask_value=-1e-8):
        super(EdgeguidedRankingLoss, self).__init__()
        self.point_pairs = point_pairs # number of point pairs
        self.sigma = sigma # used for determining the ordinal relationship between a selected pair
        self.alpha = alpha # used for balancing the effect of = and (<,>)
        self.mask_value = mask_value
        #self.regularization_loss = GradientLoss(scales=4)

    def getEdge(self, images):
        n,c,h,w = images.size()
        a = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).cuda().view((1,1,3,3)).repeat(1, 1, 1, 1)
        b = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).cuda().view((1,1,3,3)).repeat(1, 1, 1, 1)
        if c == 3:
            gradient_x = F.conv2d(images[:,0,:,:].unsqueeze(1), a)
            gradient_y = F.conv2d(images[:,0,:,:].unsqueeze(1), b)
        else:
            gradient_x = F.conv2d(images, a)
            gradient_y = F.conv2d(images, b)
        edges = torch.sqrt(torch.pow(gradient_x,2)+ torch.pow(gradient_y,2))
        edges = F.pad(edges, (1,1,1,1), "constant", 0)
        thetas = torch.atan2(gradient_y, gradient_x)
        thetas = F.pad(thetas, (1,1,1,1), "constant", 0)

        return edges, thetas

    def forward(self, inputs, targets, images, masks=None):
        # 假设 disp_tensor 是包含视差值的张量
        # min_disp = inputs.min()
        # if min_disp < 0:
        #     inputs += (-min_disp + 1)  # 平移所有视差值使其为正

        
        # 假设 inputs 是包含视差值的张量，形状为 [B, 1, H, W]
        # 首先去除单一通道维度，使其更容易处理
        # inputs_squeezed = inputs.squeeze(1)  # 结果形状为 [B, H, W]
        # # 计算每个样本的最小视差
        # min_disp_per_sample = inputs_squeezed.view(inputs_squeezed.size(0), -1).min(1)[0]  # 计算每个样本的最小视差
        # min_disp_per_sample = min_disp_per_sample.view(-1, 1, 1)  # 调整形状以便于广播
        # # 对每个样本的视差进行平移，使所有值为正
        # adjusted_inputs = inputs_squeezed + (-min_disp_per_sample + 1)
        # # 如果需要保持原始四维形状，可以再次添加单通道维度
        # inputs = adjusted_inputs.unsqueeze(1)  # 结果形状为 [B, 1, H, W]




        if masks == None:
            masks = targets > self.mask_value
        # Comment this line if you don't want to use the multi-scale gradient matching term !!!
        # regularization_loss = self.regularization_loss(inputs.squeeze(1), targets.squeeze(1), masks.squeeze(1))
        # find edges from RGB
        edges_img, thetas_img = self.getEdge(images)

        #=============================
        n,c,h,w = targets.size()
        if n != 1:
            inputs = inputs.view(n, -1).double()
            targets = targets.view(n, -1).double()
            masks = masks.view(n, -1).double()
            edges_img = edges_img.view(n, -1).double()
            thetas_img = thetas_img.view(n, -1).double()

        else:
            inputs = inputs.contiguous().view(1, -1).double()
            targets = targets.contiguous().view(1, -1).double()
            masks = masks.contiguous().view(1, -1).double()
            edges_img = edges_img.contiguous().view(1, -1).double()
            thetas_img = thetas_img.contiguous().view(1, -1).double()

        # initialization
        loss = torch.DoubleTensor([0.0]).cuda()


        for i in range(n):
            # Edge-Guided sampling
            inputs_A, inputs_B, targets_A, targets_B, masks_A, masks_B, sample_num = edgeGuidedSampling(inputs[i,:], targets[i, :], edges_img[i], thetas_img[i], masks[i, :], h, w)
            # Random Sampling
            random_sample_num = sample_num
            random_inputs_A, random_inputs_B, random_targets_A, random_targets_B, random_masks_A, random_masks_B = randomSampling(inputs[i,:], targets[i, :], masks[i, :], self.mask_value, random_sample_num)

            # Combine EGS + RS
            inputs_A = torch.cat((inputs_A, random_inputs_A), 0)
            inputs_B = torch.cat((inputs_B, random_inputs_B), 0)
            targets_A = torch.cat((targets_A, random_targets_A), 0)
            targets_B = torch.cat((targets_B, random_targets_B), 0)
            masks_A = torch.cat((masks_A, random_masks_A), 0)
            masks_B = torch.cat((masks_B, random_masks_B), 0)

            #GT ordinal relationship
            target_ratio = torch.div(targets_A+1e-6, targets_B+1e-6)
            mask_eq = target_ratio.lt(1.0 + self.sigma) * target_ratio.gt(1.0/(1.0+self.sigma))
            labels = torch.zeros_like(target_ratio)
            labels[target_ratio.ge(1.0 + self.sigma)] = 1
            labels[target_ratio.le(1.0/(1.0+self.sigma))] = -1

            # consider forward-backward consistency checking, i.e, only compute losses of point pairs with valid GT
            consistency_mask = masks_A * masks_B

            equal_loss = (inputs_A - inputs_B).pow(2) * mask_eq.double() * consistency_mask
            unequal_loss = torch.log(1 + torch.exp((-inputs_A + inputs_B) * labels)) * (~mask_eq).double() * consistency_mask

            # Please comment the regularization term if you don't want to use the multi-scale gradient matching loss !!!
            loss = loss + self.alpha * equal_loss.mean() + 1.0 * unequal_loss.mean() #+ 0.2 * regularization_loss.double()

        return loss[0].float()/n




def generate_vectorc(radius):
    abs_radius = radius.abs()
    basic_disk = torch.ones(abs_radius * 2 + 1, abs_radius * 2 + 1)
    for i in range(abs_radius * 2 + 1):
        for j in range(abs_radius * 2 + 1):
            if ((i-abs_radius) ** 2 + (j-abs_radius) ** 2) > abs_radius ** 2:
                basic_disk[i][j] = 0.0
    sign_radius = radius.sign()
    blur_kernel = torch.zeros_like(basic_disk)
    for i in range(2 * abs_radius):
        center_point = i * sign_radius + abs_radius
        start_point = max(center_point - abs_radius, 0)
        end_point = min(center_point + abs_radius + 1, abs_radius * 2+1)
        blur_kernel[:,start_point:end_point] += basic_disk[:, start_point:end_point] ** 2

    sum_kernel = blur_kernel.sum()
    blur_kernel = blur_kernel/(sum_kernel)
    return blur_kernel


class GemoLoss(nn.Module):
    def __init__(self):
        super(GemoLoss, self).__init__()
        self.mse_loss = L1Loss()
        # MSELoss()
        self.alpha = {
            "unsurpervise": 10,
            "tv_loss": 0.1,
            "rank_loss": 0.1,
        }
        self.beta = 0.1
        self.gamma = 1
        self.radius = 25
        self.epoch = 0
        self.tv_loss = TotalVariation()
        self.radius_set, self.weight_pos_set, self.weight_neg_set = self.radius_dict(self.radius)
        self.rank_loss = EdgeguidedRankingLoss()

    def radius_dict(self, c_radius):
        radius_set = [torch.tensor([[1.]])]
        for i in range(1,c_radius):
            radius_set.append(generate_vectorc(torch.tensor(i)))

        weight_pos_set = []
        weight_neg_set = []
        for i in range(1, c_radius+1):
            current_conv = nn.Conv2d(1, 1, kernel_size=i * 2 + 1, bias=False)
            current_conv.weight.data = radius_set[i-1].unsqueeze(0).unsqueeze(0)
            current_conv.requires_grad_(False)
            weight_pos_set.append(current_conv.cuda())
        for i in range(1, c_radius + 1):
            current_conv = nn.Conv2d(1, 1, kernel_size=i * 2 + 1, bias=False)
            current_conv.weight.data = radius_set[i-1].flip(1).unsqueeze(0).unsqueeze(0)
            current_conv.requires_grad_(False)
            weight_neg_set.append(current_conv.cuda())

        return radius_set, weight_pos_set, weight_neg_set

    def compute_gemo(self, blur_map, left, right):
        left_gemo = torch.zeros_like(blur_map)
        right_gemo = torch.zeros_like(blur_map)

        for i in range(self.radius):
            # current_left = F.pad(x[:, :1, :, :], pad=(i, i, i, i))
            # current_right = F.pad(x[:, 1:, :, :], pad=(i, i, i, i))

            current_left = F.pad(left, pad=(i, i, i, i))
            current_right = F.pad(right, pad=(i, i, i, i))


            lpos_mask = ((i - 1 < blur_map) & (blur_map <= i)).float()
            rpos_mask = ((i < blur_map) & (blur_map < i + 1)).float()
            lneg_mask = ((-(i + 1) < blur_map) & (blur_map <= -i)).float()
            rneg_mask = ((-i < blur_map) & (blur_map < -(i - 1))).float()

            pos_mask = (blur_map - i + 1) * lpos_mask + (i + 1 - blur_map) * rpos_mask
            neg_mask = (blur_map + i + 1) * lneg_mask + (-blur_map - i + 1) * rneg_mask

            if i == 0:
                pos_mask = pos_mask * 0.5
                neg_mask = neg_mask * 0.5

            if (pos_mask.sum() + neg_mask.sum()) > 0:
                for j in range(1):
                    left_gemo[:, j:j+1, :, :] += \
                        self.weight_pos_set[i](current_left[:, j:j+1, :, :]) * pos_mask[:, j:j+1, :, :]
                    left_gemo[:, j:j+1, :, :] += \
                        self.weight_neg_set[i](current_left[:, j:j+1, :, :]) * neg_mask[:, j:j+1, :, :]

                    right_gemo[:, j:j+1, :, :] += \
                        self.weight_neg_set[i](current_right[:, j:j+1, :, :]) * pos_mask[:, j:j+1, :, :]
                    right_gemo[:, j:j+1, :, :] += \
                        self.weight_pos_set[i](current_right[:, j:j+1, :, :]) * neg_mask[:, j:j+1, :, :]
        return left_gemo, right_gemo

    def blur_mean_loss(self, img1, img2):
        diff = (img1 - img2)
        diff = gaussian_blur2d(diff, (3, 3), (1.5, 1.5)).abs()
        return diff.mean()


    def x_grad(self, img):

        img = F.pad(img, (0, 1, 0, 0), mode='replicate')
        grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]

        return grad_x

    def y_grad(self, img):

        img = F.pad(img, (0, 0, 0, 1), mode='replicate')
        grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]

        return grad_y


    def disp_smoothness_single_image(self, disp, image):
        # 计算视差图的横向和纵向梯度
        disp_x_grad = self.x_grad(disp)
        disp_y_grad = self.y_grad(disp)

        # 计算图像的横向和纵向梯度
        image_x_grad = self.x_grad(image)
        image_y_grad = self.y_grad(image)

        # 根据图像梯度计算权重
        weight_x = torch.exp(-50 * torch.mean(torch.abs(image_x_grad), 1, keepdim=True))
        weight_y = torch.exp(-50 * torch.mean(torch.abs(image_y_grad), 1, keepdim=True))


        # 创建一个简单的均值滤波器核，大小为3x3，用于膨胀权重
        kernel_size = 3
        dilation = 1  # 可以调整膨胀的大小
        padding = dilation * (kernel_size - 1) // 2  # 计算需要的填充以保持尺寸不变
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=weight_x.device) / (kernel_size * kernel_size)

        # 对权重进行膨胀
        weight_x = F.conv2d(weight_x, kernel, padding=padding, groups=weight_x.shape[1])
        weight_y = F.conv2d(weight_y, kernel, padding=padding, groups=weight_y.shape[1])

        # 应用权重到视差梯度
        smoothness_x = disp_x_grad * weight_x
        smoothness_y = disp_y_grad * weight_y

        # 计算最终的平滑度损失
        smoothness = (torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))) / 2

        return smoothness


    def local_ranking_loss(self, disp, depth):
        # 假设 disp 和 depth 形状为 (B, C, H, W)，C 应该为 1

        # 获取水平方向相邻像素的视差和深度
        right_disp = disp[:, :, :, 1:]
        left_disp = disp[:, :, :, :-1]
        right_depth = depth[:, :, :, 1:]
        left_depth = depth[:, :, :, :-1]

        # 计算水平方向的排序损失
        horizontal_loss = torch.mean(F.relu((left_disp - right_disp) * (left_depth - right_depth)))

        # 获取垂直方向相邻像素的视差和深度
        down_disp = disp[:, :, 1:, :]
        up_disp = disp[:, :, :-1, :]
        down_depth = depth[:, :, 1:, :]
        up_depth = depth[:, :, :-1, :]

        # 计算垂直方向的排序损失
        vertical_loss = torch.mean(F.relu((up_disp - down_disp) * (up_depth - down_depth)))

        # 总损失为水平和垂直损失的平均
        loss = (horizontal_loss + vertical_loss) / 2.0
        return loss

    def whole_ranking_loss(self, disp, depth):
        # 展平图像
        flat_disp = disp.view(disp.size(0), -1)
        flat_depth = depth.view(depth.size(0), -1)

        # 计算全局排序索引
        sorted_indices_disp = torch.argsort(flat_disp, dim=1)
        sorted_indices_depth = torch.argsort(flat_depth, dim=1)

        # 计算排序不一致的像素对数量
        inconsistent_pairs = torch.sum(sorted_indices_disp != sorted_indices_depth, dim=1).float()
        loss = torch.mean(inconsistent_pairs)  # 计算平均不一致的数量

        return loss

    def forward(self, blur_map, left, right, depth=None):
        losses = {}
        left_gemo, right_gemo = self.compute_gemo(blur_map, left, right)
        unsurpervise_loss = self.blur_mean_loss(left_gemo, right_gemo) * self.alpha["unsurpervise"]
        

        losses["unsp loss"] = unsurpervise_loss
        # tv_loss = self.tv_loss(blur_map).sum() * self.alpha["tv_loss"]

        rank_loss = 0
        if depth is not None:
            tv_loss = self.disp_smoothness_single_image(blur_map, depth) * self.alpha["tv_loss"]
            rank_loss = self.rank_loss(blur_map, depth, left) * self.alpha["rank_loss"]
            losses["rank loss"] = rank_loss
        else:
            tv_loss = self.disp_smoothness_single_image(blur_map, torch.cat([left, right],dim=1)) * self.alpha["tv_loss"]
            
        losses["tv loss"] = tv_loss
        
        # if self.epoch < 1000:
        #     losses["total_loss"] = unsurpervise_loss
        # else:
        losses["loss"] = unsurpervise_loss + tv_loss + rank_loss
        self.epoch += 1
        return losses




def normalize_coords(grid):
    """Normalize coordinates of image scale to [-1, 1]
    Args:
        grid: [B, 2, H, W]
    """
    assert grid.size(1) == 2
    h, w = grid.size()[2:]
    grid[:, 0, :, :] = 2 * (grid[:, 0, :, :].clone() / (w - 1)) - 1  # x: [-1, 1]
    grid[:, 1, :, :] = 2 * (grid[:, 1, :, :].clone() / (h - 1)) - 1  # y: [-1, 1]
    grid = grid.permute((0, 2, 3, 1))  # [B, H, W, 2]
    return grid


def meshgrid(img, homogeneous=False):
    """Generate meshgrid in image scale
    Args:
        img: [B, _, H, W]
        homogeneous: whether to return homogeneous coordinates
    Return:
        grid: [B, 2, H, W]
    """
    b, _, h, w = img.size()

    x_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(img)  # [1, H, W]
    y_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(img)

    grid = torch.cat((x_range, y_range), dim=0)  # [2, H, W], grid[:, i, j] = [j, i]
    grid = grid.unsqueeze(0).expand(b, 2, h, w)  # [B, 2, H, W]

    if homogeneous:
        ones = torch.ones_like(x_range).unsqueeze(0).expand(b, 1, h, w)  # [B, 1, H, W]
        grid = torch.cat((grid, ones), dim=1)  # [B, 3, H, W]
        assert grid.size(1) == 3
    return grid


def disp_warp(img, disp, padding_mode='border'):
    """Warping by disparity
    Args:
        img: [B, 1, H, W] or [B, 3, H, W]
        disp: [B, 1, H, W], positive or negative
        padding_mode: 'zeros' or 'border'
    Returns:
        warped_img: [B, 1, H, W] or [B, 3, H, W]
        valid_mask: [B, 1, H, W] or [B, 3, H, W]
    """

    grid = meshgrid(img)  # [B, 2, H, W] in image scale
    # Note that disp can be positive or negative
    offset = torch.cat((-disp, torch.zeros_like(disp)), dim=1)  # [B, 2, H, W]
    sample_grid = grid + offset
    sample_grid = normalize_coords(sample_grid)  # [B, H, W, 2] in [-1, 1]
    warped_img = F.grid_sample(img, sample_grid, mode='bilinear', padding_mode=padding_mode, align_corners=True)

    mask = torch.ones_like(img)
    valid_mask = F.grid_sample(mask, sample_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    valid_mask[valid_mask < 0.9999] = 0
    valid_mask[valid_mask > 0] = 1
    return warped_img, valid_mask


class TernaryLoss(nn.Module):
    def __init__(self, max_distance=1):
        super(TernaryLoss, self).__init__()
        self.max_distance = max_distance
        self.patch_size = 2 * max_distance + 1

    def ternary_transform(self, image):
        intensities = image.mean(dim=1, keepdim=True) * 255  # 将图像转换为灰度图像
        out_channels = self.patch_size * self.patch_size

        # 使用 unfold 创建图像的补丁
        patches = F.unfold(intensities, kernel_size=self.patch_size, padding=self.patch_size // 2)
        patches = patches.view(image.size(0), out_channels, image.size(2), image.size(3))

        transf = patches - intensities
        transf_norm = transf / torch.sqrt(0.81 + transf ** 2)
        return transf_norm

    def hamming_distance(self, t1, t2):
        dist = (t1 - t2) ** 2
        dist_norm = dist / (0.1 + dist)
        dist_sum = dist_norm.sum(dim=1, keepdim=True)
        return dist_sum

    def create_mask(self, tensor, paddings):
        shape = tensor.shape
        inner_width = shape[2] - (paddings[0][0] + paddings[0][1])
        inner_height = shape[3] - (paddings[1][0] + paddings[1][1])
        inner = torch.ones((inner_width, inner_height), device=tensor.device)

        mask2d = F.pad(inner, [paddings[1][0], paddings[1][1], paddings[0][0], paddings[0][1]])
        mask3d = mask2d.unsqueeze(0).repeat(shape[0], 1, 1)
        mask4d = mask3d.unsqueeze(1)
        return mask4d.detach()  # 停止梯度计算

    def charbonnier_loss(self, diff, mask, eps=1e-3):
        loss = torch.sqrt(diff ** 2 + eps ** 2)
        return (loss * mask).mean()

    def forward(self, im1, im2_warped, mask):
        t1 = self.ternary_transform(im1)
        t2 = self.ternary_transform(im2_warped)
        dist = self.hamming_distance(t1, t2)

        transform_mask = self.create_mask(mask, [[self.max_distance, self.max_distance], [self.max_distance, self.max_distance]])
        return self.charbonnier_loss(dist, mask * transform_mask)

def ternary_loss_func(im1, im2_warped, mask, max_distance=1):
    patch_size = 2 * max_distance + 1
    out_channels = patch_size * patch_size

    def _ternary_transform(image):
        intensities = normalize_image(image)* 255
        w = torch.eye(out_channels).view(out_channels, 1, patch_size, patch_size)
        weights = w.to(image.device).type(image.dtype)
        # 计算 'SAME' 填充
        height, width = intensities.shape[2:]
        pad_height = max((patch_size - 1), 0)
        pad_width = max((patch_size - 1), 0)
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        intensities_padded = F.pad(intensities, [pad_left, pad_right, pad_top, pad_bottom])
        try:
            patches = F.conv2d(intensities_padded, weights, stride=1)
        except:
            weights = weights.to(intensities_padded.dtype)
            patches = F.conv2d(intensities_padded, weights, stride=1)
        
        transf = patches - intensities
        transf_norm = transf / (torch.sqrt(0.81 + torch.square(transf)) + 1e-6)
        return transf_norm

    def normalize_image(image):
        # 计算每个批次中的最小和最大值
        batch_min = image.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        batch_max = image.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

        # 归一化到 [0, 1]
        normalized_image = (image - batch_min) / (batch_max - batch_min + 1e-6)
        return normalized_image

    def _hamming_distance(t1, t2):
        dist = torch.square(t1 - t2)
        dist_norm = dist / (0.1 + dist)
        dist_sum = torch.sum(dist_norm, dim=1, keepdim=True)
        return dist_sum

    def create_mask(mask, kernel_size):
        # You need to define or adjust this function according to your needs
        # This is a placeholder implementation
        return F.max_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=max_distance)

    def charbonnier_loss(input, target, alpha=0.45, epsilon=1e-3):
        diff = input - target
        epsilon_tensor = torch.tensor(epsilon, device=diff.device, dtype=diff.dtype)
        loss = torch.sum(torch.pow(torch.square(diff) + torch.square(epsilon_tensor), alpha))
        return loss

    t1 = _ternary_transform(im1)
    t2 = _ternary_transform(im2_warped)
    dist = _hamming_distance(t1, t2)

    transform_mask = create_mask(mask, (patch_size, patch_size))
    loss = charbonnier_loss(dist, mask * transform_mask)
    effective_elements = mask.sum()
    loss = loss / effective_elements if effective_elements > 0 else torch.tensor(0.0)
    return loss


class DisparitySmoothnessLoss(torch.nn.Module):
    def __init__(self):
        super(DisparitySmoothnessLoss, self).__init__()

    def gradient_x(self, img):
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]
        return gx

    def gradient_y(self, img):
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gy

    def forward(self, disp, img):
        # Calculate gradients of disparity
        disp_grad_x = self.gradient_x(disp)
        disp_grad_y = self.gradient_y(disp)

        # Calculate image gradients
        img_grad_x = self.gradient_x(img)
        img_grad_y = self.gradient_y(img)

        # Weight disparity gradients with image gradients
        disp_grad_x = disp_grad_x * torch.exp(-5*torch.mean(torch.abs(img_grad_x), 1, keepdim=True))
        disp_grad_y = disp_grad_y * torch.exp(-5*torch.mean(torch.abs(img_grad_y), 1, keepdim=True))

        # Compute the smoothness loss
        loss_x = disp_grad_x.abs().mean()
        loss_y = disp_grad_y.abs().mean()

        loss = loss_x + loss_y
        return loss


class Whole_loss(torch.nn.Module):
    def __init__(self):
        super(Whole_loss, self).__init__()
        self.ternary = TernaryLoss()
        self.smooth = DisparitySmoothnessLoss()
        self.weights = [0.4, 0.5, 0.6, 0.7, 1]  # 权重列表，长度为 5

    def forward(self, disps, left, right, train=True):
        total_ternary_loss = 0
        total_smooth_loss = 0
        if train:
            # 循环计算多尺度损失
            for i, disp in enumerate(disps):
                right_warped, valid_mask = disp_warp(left, disp)
                ternary_loss = ternary_loss_func(left, right_warped, valid_mask)
                smooth_loss = self.smooth(disp, left)

                # 按权重累加损失
                total_ternary_loss += self.weights[i] * ternary_loss
                total_smooth_loss += self.weights[i] * smooth_loss * 0.01

        else:
            right_warped, valid_mask = disp_warp(left, disps)
            ternary_loss = ternary_loss_func(left, right_warped, valid_mask)
            smooth_loss = self.smooth(disps, left)

            # 按权重累加损失
            total_ternary_loss += ternary_loss
            total_smooth_loss += smooth_loss * 0.01

        # 所有损失的总和
        total_loss = total_ternary_loss + total_smooth_loss 

        loss_dict = {
            'ternary_loss': total_ternary_loss,
            'smooth_loss': total_smooth_loss,
            'loss': total_loss  # 总损失
        }

        return loss_dict

        