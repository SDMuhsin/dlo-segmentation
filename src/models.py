"""
DGCNN and PointNet++ models for point cloud segmentation.
Based on the CDLO dataset with 5 classes: Wire, Endpoint, Bifurcation, Connector, Noise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    """Compute k-nearest neighbors for each point."""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    """Get edge features for each point based on k-nearest neighbors."""
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn(x, k=k)

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class EdgeConv(nn.Module):
    """EdgeConv layer for DGCNN."""

    def __init__(self, in_channels, out_channels, k=20):
        super(EdgeConv, self).__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]
        return x


class DGCNNSegmentation(nn.Module):
    """
    DGCNN for point cloud segmentation.
    H43 Teacher model: 5 EdgeConv layers (~1,246,597 params)
    Configuration: [64, 64, 48, 128, 192] -> emb=1024 -> [384, 192] -> 5
    """

    def __init__(self, num_classes=5, k=20, dropout=0.5):
        super(DGCNNSegmentation, self).__init__()
        self.k = k

        # EdgeConv layers
        self.conv1 = EdgeConv(3, 64, k=k)
        self.conv2 = EdgeConv(64, 64, k=k)
        self.conv3 = EdgeConv(64, 48, k=k)
        self.conv4 = EdgeConv(48, 128, k=k)
        self.conv5 = EdgeConv(128, 192, k=k)

        # Local features: 64+64+48+128+192 = 496
        local_feat_dim = 64 + 64 + 48 + 128 + 192  # = 496
        emb_dims = 1024

        # MLP for global feature
        self.conv6 = nn.Sequential(
            nn.Conv1d(local_feat_dim, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Segmentation head: local_feat + emb_dims = 1520
        concat_dim = local_feat_dim + emb_dims

        self.conv7 = nn.Sequential(
            nn.Conv1d(concat_dim, 384, kernel_size=1, bias=False),
            nn.BatchNorm1d(384),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv8 = nn.Sequential(
            nn.Conv1d(384, 192, kernel_size=1, bias=False),
            nn.BatchNorm1d(192),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.dp1 = nn.Dropout(p=dropout)
        self.conv9 = nn.Conv1d(192, num_classes, kernel_size=1)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        # EdgeConv layers
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        # Concatenate features
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        # Global feature
        x_global = self.conv6(x)
        x_global = x_global.max(dim=-1, keepdim=True)[0]
        x_global = x_global.repeat(1, 1, num_points)

        # Concatenate local and global features
        x = torch.cat((x, x_global), dim=1)

        # Segmentation head
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.dp1(x)
        x = self.conv9(x)

        return x


class DGCNNStudent(nn.Module):
    """
    Smaller DGCNN for knowledge distillation (4 EdgeConv layers).
    Target: 702,245 params, 2.69 MB
    Configuration: [72, 64, 144, 192] -> emb=576 -> [288, 144] -> 5
    """

    def __init__(self, num_classes=5, k=20, dropout=0.5):
        super(DGCNNStudent, self).__init__()
        self.k = k

        # 4 EdgeConv layers
        self.conv1 = EdgeConv(3, 72, k=k)
        self.conv2 = EdgeConv(72, 64, k=k)
        self.conv3 = EdgeConv(64, 144, k=k)
        self.conv4 = EdgeConv(144, 192, k=k)

        # Local features: 72+64+144+192 = 472
        local_feat_dim = 72 + 64 + 144 + 192  # = 472
        emb_dims = 576

        # MLP for global feature
        self.conv5 = nn.Sequential(
            nn.Conv1d(local_feat_dim, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Segmentation head: local_feat + emb_dims = 1048
        concat_dim = local_feat_dim + emb_dims

        self.conv6 = nn.Sequential(
            nn.Conv1d(concat_dim, 288, kernel_size=1, bias=False),
            nn.BatchNorm1d(288),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(288, 144, kernel_size=1, bias=False),
            nn.BatchNorm1d(144),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.dp1 = nn.Dropout(p=dropout)
        self.conv8 = nn.Conv1d(144, num_classes, kernel_size=1)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        # EdgeConv layers
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        # Concatenate features
        x = torch.cat((x1, x2, x3, x4), dim=1)

        # Global feature
        x_global = self.conv5(x)
        x_global = x_global.max(dim=-1, keepdim=True)[0]
        x_global = x_global.repeat(1, 1, num_points)

        # Concatenate local and global features
        x = torch.cat((x, x_global), dim=1)

        # Segmentation head
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.dp1(x)
        x = self.conv8(x)

        return x


class PointNetSetAbstraction(nn.Module):
    """Set Abstraction module for PointNet++."""

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        B, N, C = xyz.shape

        if self.group_all:
            new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
            grouped_xyz = xyz.view(B, 1, N, 3)
            if points is not None:
                grouped_points = points.view(B, 1, N, -1)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz
        else:
            new_xyz = self._farthest_point_sample(xyz, self.npoint)
            idx = self._ball_query(xyz, new_xyz, self.radius, self.nsample)
            grouped_xyz = self._index_points(xyz, idx)
            grouped_xyz -= new_xyz.view(B, self.npoint, 1, 3)

            if points is not None:
                grouped_points = self._index_points(points, idx)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz

        grouped_points = grouped_points.permute(0, 3, 2, 1)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))

        new_points = torch.max(grouped_points, 2)[0]
        new_points = new_points.permute(0, 2, 1)

        return new_xyz, new_points

    def _farthest_point_sample(self, xyz, npoint):
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
        distance = torch.ones(B, N, device=xyz.device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)
        batch_indices = torch.arange(B, dtype=torch.long, device=xyz.device)

        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            distance = torch.min(distance, dist)
            farthest = torch.max(distance, -1)[1]

        return self._index_points(xyz, centroids)

    def _ball_query(self, xyz, new_xyz, radius, nsample):
        B, N, C = xyz.shape
        _, S, _ = new_xyz.shape

        sqrdists = self._square_distance(new_xyz, xyz)
        idx = torch.arange(N, device=xyz.device).repeat(B, S, 1)
        idx[sqrdists > radius ** 2] = N
        idx = idx.sort(dim=-1)[0][:, :, :nsample]

        first_idx = idx[:, :, 0:1].repeat(1, 1, nsample)
        mask = idx == N
        idx[mask] = first_idx[mask]

        return idx

    def _square_distance(self, src, dst):
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist

    def _index_points(self, points, idx):
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points


class PointNetFeaturePropagation(nn.Module):
    """Feature Propagation module for PointNet++."""

    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = self._square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_points = torch.sum(
                self._index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2
            )

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        return new_points.permute(0, 2, 1)

    def _square_distance(self, src, dst):
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist

    def _index_points(self, points, idx):
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points


class PointNet2Segmentation(nn.Module):
    """
    PointNet++ for point cloud segmentation.
    Target: 1,402,437 parameters, 5.38 MB
    """

    def __init__(self, num_classes=5):
        super(PointNet2Segmentation, self).__init__()

        # Set Abstraction layers - tuned for 1.4M params
        self.sa1 = PointNetSetAbstraction(512, 0.1, 32, 3, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, 0.2, 64, 128 + 3, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 512, 1024], group_all=True)

        # Feature Propagation layers
        self.fp3 = PointNetFeaturePropagation(1280, [256, 256])
        self.fp2 = PointNetFeaturePropagation(384, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128 + 3, [128, 128, 128])

        # Segmentation head
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, x):
        xyz = x.permute(0, 2, 1).contiguous()
        B, N, _ = xyz.shape

        # Set Abstraction
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Feature Propagation
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, xyz, l1_points)

        # Segmentation head
        x = l0_points.permute(0, 2, 1)
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x)

        return x


def count_parameters(model):
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """Get model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test DGCNN Teacher
    teacher = DGCNNSegmentation(num_classes=5).to(device)
    print(f"\nDGCNN Teacher (H43):")
    print(f"  Parameters: {count_parameters(teacher):,} (target: 1,246,597)")
    print(f"  Size: {get_model_size_mb(teacher):.2f} MB (target: 4.77 MB)")

    # Test DGCNN Student
    student = DGCNNStudent(num_classes=5).to(device)
    print(f"\nDGCNN Student (Distilled):")
    print(f"  Parameters: {count_parameters(student):,} (target: 702,245)")
    print(f"  Size: {get_model_size_mb(student):.2f} MB (target: 2.69 MB)")

    # Test PointNet++
    pointnet2 = PointNet2Segmentation(num_classes=5).to(device)
    print(f"\nPointNet++ Baseline:")
    print(f"  Parameters: {count_parameters(pointnet2):,} (target: 1,402,437)")
    print(f"  Size: {get_model_size_mb(pointnet2):.2f} MB (target: 5.38 MB)")

    # Test forward pass
    batch_size = 4
    num_points = 2048
    x = torch.randn(batch_size, 3, num_points).to(device)

    print(f"\nTest forward pass with input shape: {x.shape}")

    with torch.no_grad():
        out_teacher = teacher(x)
        print(f"Teacher output: {out_teacher.shape}")

        out_student = student(x)
        print(f"Student output: {out_student.shape}")

        out_pn2 = pointnet2(x)
        print(f"PointNet++ output: {out_pn2.shape}")

    # Test inference time
    import time

    print("\nInference time (avg of 100 runs):")
    for name, model in [("Teacher", teacher), ("Student", student), ("PointNet++", pointnet2)]:
        model.eval()
        x = torch.randn(1, 3, 2048).to(device)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(x)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 100 * 1000
        print(f"  {name}: {elapsed:.2f} ms")
