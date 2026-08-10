"""
Training script for knowledge distillation.

Training configuration (from report):
- 20 epochs
- Adam optimizer, LR = 0.001
- Weighted cross-entropy loss
- Temperature T=4.0 for knowledge distillation
- Loss balance: α=0.7 (70% task loss, 30% distillation loss)
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import DGCNNSegmentation, DGCNNStudent, PointNet2Segmentation, count_parameters, get_model_size_mb
from src.dataset import PointWireDataset, create_dataloaders


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss.

    L = α * L_task + (1 - α) * L_distill * T^2

    Where:
        L_task: Weighted cross-entropy with ground truth
        L_distill: KL divergence between student and teacher soft targets
        T: Temperature for softening probabilities
        α: Balance between task loss and distillation loss
    """

    def __init__(self, temperature=4.0, alpha=0.7, class_weights=None):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.class_weights = class_weights

        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets):
        """
        Args:
            student_logits: (B, C, N) student model output
            teacher_logits: (B, C, N) teacher model output
            targets: (B, N) ground truth labels
        """
        # Task loss (hard targets)
        loss_task = self.ce_loss(student_logits, targets)

        # Distillation loss (soft targets)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)

        loss_distill = F.kl_div(
            soft_student, soft_teacher,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Combined loss
        loss = self.alpha * loss_task + (1 - self.alpha) * loss_distill

        return loss, loss_task, loss_distill


def compute_iou(pred, target, num_classes=5):
    """Compute per-class IoU and mean IoU."""
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union > 0:
            iou = intersection / union
        else:
            iou = torch.tensor(1.0, device=pred.device) if intersection == 0 else torch.tensor(0.0, device=pred.device)

        ious.append(iou)

    return torch.stack(ious)


def train_epoch(model, teacher, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    if teacher is not None:
        teacher.eval()

    total_loss = 0
    total_task_loss = 0
    total_distill_loss = 0
    total_correct = 0
    total_points = 0
    all_ious = []

    for batch_idx, (pcl, seg) in enumerate(train_loader):
        pcl, seg = pcl.to(device), seg.to(device)

        optimizer.zero_grad()

        # Forward pass
        student_logits = model(pcl)

        if teacher is not None:
            with torch.no_grad():
                teacher_logits = teacher(pcl)
            loss, loss_task, loss_distill = criterion(student_logits, teacher_logits, seg)
            total_task_loss += loss_task.item()
            total_distill_loss += loss_distill.item()
        else:
            # Standard training (no distillation)
            loss = criterion.ce_loss(student_logits, seg)
            total_task_loss += loss.item()

        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute metrics
        pred = student_logits.argmax(dim=1)
        total_correct += (pred == seg).sum().item()
        total_points += seg.numel()

        # Compute IoU
        ious = compute_iou(pred.view(-1), seg.view(-1))
        all_ious.append(ious)

    # Aggregate metrics
    avg_loss = total_loss / len(train_loader)
    avg_task_loss = total_task_loss / len(train_loader)
    avg_distill_loss = total_distill_loss / len(train_loader) if teacher is not None else 0
    accuracy = total_correct / total_points * 100
    mean_ious = torch.stack(all_ious).mean(dim=0)
    miou = mean_ious.mean().item() * 100

    return {
        'loss': avg_loss,
        'task_loss': avg_task_loss,
        'distill_loss': avg_distill_loss,
        'accuracy': accuracy,
        'miou': miou,
        'per_class_iou': (mean_ious * 100).tolist()
    }


@torch.no_grad()
def evaluate(model, data_loader, device, num_classes=5):
    """Evaluate model on a dataset."""
    model.eval()

    total_correct = 0
    total_points = 0
    all_ious = []

    for pcl, seg in data_loader:
        pcl, seg = pcl.to(device), seg.to(device)

        logits = model(pcl)
        pred = logits.argmax(dim=1)

        total_correct += (pred == seg).sum().item()
        total_points += seg.numel()

        ious = compute_iou(pred.view(-1), seg.view(-1), num_classes)
        all_ious.append(ious)

    accuracy = total_correct / total_points * 100
    mean_ious = torch.stack(all_ious).mean(dim=0)
    miou = mean_ious.mean().item() * 100

    return {
        'accuracy': accuracy,
        'miou': miou,
        'per_class_iou': (mean_ious * 100).tolist()
    }


@torch.no_grad()
def measure_inference_time(model, device, num_points=2048, num_runs=100):
    """Measure average inference time."""
    model.eval()
    x = torch.randn(1, 3, num_points).to(device)

    # Warmup
    for _ in range(10):
        _ = model(x)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        _ = model(x)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / num_runs * 1000

    return elapsed


def train_teacher(data_path, results_path, epochs=20, batch_size=16, lr=0.001):
    """Train the DGCNN teacher model."""
    print("=" * 60)
    print("Training DGCNN Teacher (H43)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path, batch_size=batch_size, num_workers=4
    )
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")
    print(f"Test: {len(test_loader.dataset)} samples")

    # Create model
    model = DGCNNSegmentation(num_classes=5).to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")
    print(f"Model size: {get_model_size_mb(model):.2f} MB")

    # Loss and optimizer
    class_weights = PointWireDataset.CLASS_WEIGHTS.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_miou = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_points = 0

        for pcl, seg in train_loader:
            pcl, seg = pcl.to(device), seg.to(device)

            optimizer.zero_grad()
            logits = model(pcl)
            loss = criterion(logits, seg)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            train_correct += (pred == seg).sum().item()
            train_points += seg.numel()

        train_acc = train_correct / train_points * 100
        train_loss = train_loss / len(train_loader)

        # Validate
        val_metrics = evaluate(model, val_loader, device)

        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val mIoU: {val_metrics['miou']:.2f}%, Acc: {val_metrics['accuracy']:.2f}%")

        if val_metrics['miou'] > best_miou:
            best_miou = val_metrics['miou']
            best_epoch = epoch
            torch.save(model.state_dict(), results_path / "teacher_best.pth")
            print(f"  -> New best model saved!")

        scheduler.step()

    print(f"\nBest validation mIoU: {best_miou:.2f}% at epoch {best_epoch}")

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(results_path / "teacher_best.pth"))
    test_metrics = evaluate(model, test_loader, device)
    inf_time = measure_inference_time(model, device)

    print(f"\nTest Results:")
    print(f"  mIoU: {test_metrics['miou']:.2f}%")
    print(f"  Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"  Inference time: {inf_time:.2f} ms")
    print(f"  Per-class IoU: {test_metrics['per_class_iou']}")

    return model, test_metrics


def train_student(teacher, data_path, results_path, epochs=20, batch_size=16, lr=0.001,
                 temperature=4.0, alpha=0.7):
    """Train the distilled student model."""
    print("\n" + "=" * 60)
    print("Training Distilled Student (DGCNN 4-layer)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path, batch_size=batch_size, num_workers=4
    )

    # Create student model
    student = DGCNNStudent(num_classes=5).to(device)
    print(f"\nStudent parameters: {count_parameters(student):,}")
    print(f"Student size: {get_model_size_mb(student):.2f} MB")

    # Move teacher to device and set to eval
    teacher = teacher.to(device)
    teacher.eval()

    # Loss and optimizer
    class_weights = PointWireDataset.CLASS_WEIGHTS.to(device)
    criterion = KnowledgeDistillationLoss(
        temperature=temperature,
        alpha=alpha,
        class_weights=class_weights
    )
    optimizer = Adam(student.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_miou = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Train
        train_metrics = train_epoch(student, teacher, train_loader, optimizer, criterion, device)

        # Validate
        val_metrics = evaluate(student, val_loader, device)

        print(f"  Train Loss: {train_metrics['loss']:.4f} (task: {train_metrics['task_loss']:.4f}, distill: {train_metrics['distill_loss']:.4f})")
        print(f"  Train mIoU: {train_metrics['miou']:.2f}%, Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Val mIoU: {val_metrics['miou']:.2f}%, Acc: {val_metrics['accuracy']:.2f}%")

        if val_metrics['miou'] > best_miou:
            best_miou = val_metrics['miou']
            best_epoch = epoch
            torch.save(student.state_dict(), results_path / "student_best.pth")
            print(f"  -> New best model saved!")

        scheduler.step()

    print(f"\nBest validation mIoU: {best_miou:.2f}% at epoch {best_epoch}")

    # Load best model and evaluate on test set
    student.load_state_dict(torch.load(results_path / "student_best.pth"))
    test_metrics = evaluate(student, test_loader, device)
    inf_time = measure_inference_time(student, device)

    print(f"\nTest Results:")
    print(f"  mIoU: {test_metrics['miou']:.2f}%")
    print(f"  Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"  Inference time: {inf_time:.2f} ms")
    print(f"  Per-class IoU: {test_metrics['per_class_iou']}")

    return student, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train knowledge distillation models")
    parser.add_argument("--data-path", type=str, default="./data/set2",
                       help="Path to dataset")
    parser.add_argument("--results-path", type=str, default="./results",
                       help="Path to save results")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--temperature", type=float, default=4.0,
                       help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.7,
                       help="Balance between task and distillation loss")
    parser.add_argument("--teacher-only", action="store_true",
                       help="Only train teacher model")
    parser.add_argument("--student-only", action="store_true",
                       help="Only train student model (requires pretrained teacher)")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    results_path = Path(args.results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    if args.student_only:
        # Load pretrained teacher
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        teacher = DGCNNSegmentation(num_classes=5).to(device)
        teacher.load_state_dict(torch.load(results_path / "teacher_best.pth"))
        print("Loaded pretrained teacher model")
    else:
        # Train teacher
        teacher, teacher_metrics = train_teacher(
            data_path, results_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )

    if not args.teacher_only:
        # Train student with knowledge distillation
        student, student_metrics = train_student(
            teacher, data_path, results_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            temperature=args.temperature,
            alpha=args.alpha
        )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
