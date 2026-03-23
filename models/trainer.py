"""
models/trainer.py
==================
Training loop for the GlaucomaResNet CNN.

Features
────────
- Mixed precision (torch.cuda.amp) — ~30% faster, halves VRAM on RTX GPUs
- Early stopping on validation AUC (not loss — more clinically relevant)
- Best model checkpointing — saves the epoch with highest val AUC
- TensorBoard logging — loss, AUC, LR per epoch
- Two-stage training:
    Stage 1: train head only (backbone frozen), fast convergence
    Stage 2: full fine-tune (backbone unfrozen), lower LR, careful optimisation

Usage
─────
    from models.trainer import Trainer

    trainer = Trainer(model, optimizer, scheduler, loss_fn)
    history = trainer.fit(train_loader, val_loader, epochs=20, stage=1)
    trainer.unfreeze_and_finetune(train_loader, val_loader, epochs=30)
"""

import time
import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from sklearn.metrics import roc_auc_score

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MODELS_DIR, LOGS_DIR, DEVICE, EARLY_STOP_PATIENCE


# ─────────────────────────────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────────────────────────────
class Trainer:
    """
    Manages the full training lifecycle for GlaucomaResNet.

    Args:
        model:      GlaucomaResNet instance (already on DEVICE)
        optimizer:  torch optimizer
        scheduler:  LR scheduler (stepped after each epoch)
        loss_fn:    Loss function (CrossEntropyLoss with class weights)
        run_name:   Name used for checkpoint files and TensorBoard logs
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        loss_fn,
        run_name: str = "glaucoma_resnet18",
    ):
        self.model      = model
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.loss_fn    = loss_fn
        self.run_name   = run_name

        self.scaler  = GradScaler(DEVICE)   # for mixed precision
        self.writer  = SummaryWriter(log_dir=str(LOGS_DIR / run_name))

        self.best_val_auc    = 0.0
        self.best_epoch      = 0
        self.patience_counter = 0
        self.history         = {
            "train_loss": [], "val_loss": [],
            "train_auc":  [], "val_auc":  [],
            "lr":         [],
        }

        self.ckpt_path = MODELS_DIR / f"{run_name}_best.pth"
        print(f"Checkpoint → {self.ckpt_path}")
        print(f"TensorBoard → {LOGS_DIR / run_name}")

    # ── Single epoch ──────────────────────────────────────────────────
    def _train_epoch(self, loader) -> tuple:
        self.model.train()
        total_loss, all_probs, all_labels = 0.0, [], []

        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(DEVICE):
                logits = self.model(imgs)
                loss   = self.loss_fn(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * imgs.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        auc      = roc_auc_score(all_labels, all_probs)
        return avg_loss, auc

    def _val_epoch(self, loader) -> tuple:
        self.model.eval()
        total_loss, all_probs, all_labels = 0.0, [], []

        with torch.no_grad():
            for imgs, labels in loader:
                imgs   = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                with autocast(DEVICE):
                    logits = self.model(imgs)
                    loss   = self.loss_fn(logits, labels)

                total_loss += loss.item() * imgs.size(0)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        auc      = roc_auc_score(all_labels, all_probs)
        return avg_loss, auc

    # ── Main training loop ────────────────────────────────────────────
    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int,
        stage: int = 1,
        patience: int = EARLY_STOP_PATIENCE,
    ) -> dict:
        """
        Train for up to `epochs` epochs with early stopping.

        Args:
            train_loader: DataLoader for training split
            val_loader:   DataLoader for validation split
            epochs:       Maximum epochs to run
            stage:        1 (frozen backbone) or 2 (full fine-tune)
            patience:     Early stopping patience (epochs without val AUC improvement)

        Returns:
            history dict with train/val loss and AUC per epoch
        """
        print(f"\n{'═'*60}")
        print(f"  Stage {stage} training  |  max {epochs} epochs  |  patience {patience}")
        print(f"{'═'*60}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss, train_auc = self._train_epoch(train_loader)
            val_loss,   val_auc   = self._val_epoch(val_loader)

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log to history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_auc"].append(train_auc)
            self.history["val_auc"].append(val_auc)
            self.history["lr"].append(current_lr)

            # TensorBoard
            step = epoch
            self.writer.add_scalar("Loss/train",  train_loss, step)
            self.writer.add_scalar("Loss/val",    val_loss,   step)
            self.writer.add_scalar("AUC/train",   train_auc,  step)
            self.writer.add_scalar("AUC/val",     val_auc,    step)
            self.writer.add_scalar("LR",          current_lr, step)

            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:>3}/{epochs} | "
                f"Train loss {train_loss:.4f}  AUC {train_auc:.4f} | "
                f"Val loss {val_loss:.4f}  AUC {val_auc:.4f} | "
                f"LR {current_lr:.6f} | {elapsed:.1f}s"
            )

            # Checkpoint + early stopping
            if val_auc > self.best_val_auc:
                self.best_val_auc     = val_auc
                self.best_epoch       = epoch
                self.patience_counter = 0
                torch.save({
                    "epoch":      epoch,
                    "stage":      stage,
                    "model_state_dict":     self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_auc":    val_auc,
                    "val_loss":   val_loss,
                }, self.ckpt_path)
                print(f"  ✓ Best model saved  (val AUC {val_auc:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\n  Early stopping at epoch {epoch}  "
                          f"(best val AUC {self.best_val_auc:.4f} at epoch {self.best_epoch})")
                    break

        self.writer.flush()
        print(f"\n  Stage {stage} done.  Best val AUC: {self.best_val_auc:.4f}")
        return self.history

    def unfreeze_and_finetune(
        self,
        train_loader,
        val_loader,
        epochs: int = 30,
        patience: int = EARLY_STOP_PATIENCE,
    ) -> dict:
        """
        Stage 2: unfreeze backbone, build a new optimizer with differential LR,
        and continue training from the best Stage 1 checkpoint.

        Returns:
            Updated history dict
        """
        from models.cnn_model import get_optimizer

        # Load best Stage 1 weights
        ckpt = torch.load(self.ckpt_path, map_location=DEVICE)
        self.model.load_state_dict(ckpt["model_state_dict"])
        print(f"\nLoaded Stage 1 checkpoint (epoch {ckpt['epoch']}, "
              f"val AUC {ckpt['val_auc']:.4f})")

        # Unfreeze and rebuild optimizer
        self.model.unfreeze_backbone()
        self.optimizer, self.scheduler = get_optimizer(self.model, stage=2)
        self.scaler          = GradScaler(DEVICE)
        self.patience_counter = 0
        # Keep best_val_auc from Stage 1 so we only save if Stage 2 improves further
        print(f"Stage 2 starts. Current best val AUC: {self.best_val_auc:.4f}")

        return self.fit(train_loader, val_loader, epochs=epochs, stage=2, patience=patience)

    def load_best(self):
        """Load the best checkpoint weights into self.model."""
        ckpt = torch.load(self.ckpt_path, map_location=DEVICE)
        self.model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best model: epoch {ckpt['epoch']}, val AUC {ckpt['val_auc']:.4f}")
        return self.model


# ─────────────────────────────────────────────────────────────────────
# EVALUATION  (test set)
# ─────────────────────────────────────────────────────────────────────
def evaluate_cnn(model, loader) -> dict:
    """
    Full evaluation of a trained model on any DataLoader.

    Returns dict with:
        auc, sensitivity, specificity, f1, accuracy,
        y_true, y_pred, y_prob   (for ROC curve plotting)
    """
    from sklearn.metrics import (
        roc_auc_score, confusion_matrix, f1_score, accuracy_score, roc_curve
    )

    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            with autocast(DEVICE):
                logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity    = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr, tpr, _    = roc_curve(y_true, y_prob)

    return {
        "auc":         roc_auc_score(y_true, y_prob),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1":          f1_score(y_true, y_pred),
        "accuracy":    accuracy_score(y_true, y_pred),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob,
        "fpr": fpr, "tpr": tpr,
    }
