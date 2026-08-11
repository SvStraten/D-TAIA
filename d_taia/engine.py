import copy
import random

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import accuracy, macro_f1, mae


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg):
    available = torch.cuda.is_available()
    device = torch.device(cfg.device if available else "cpu")
    print(f"torch.cuda.is_available() = {available}")
    if available:
        print(f"device count = {torch.cuda.device_count()}, using {torch.cuda.get_device_name(0)}")
    print(f"selected device = {device}")
    return device


def train_datl_stage(model, loader, cfg, device):
    if model.datl is None or cfg.no_datl:
        return

    optimizer = torch.optim.Adam(model.datl.parameters(), lr=cfg.datl_lr)
    model.datl.train()
    for epoch in range(cfg.datl_epochs):
        total_loss = 0.0
        n_batches = 0
        total_triplets = 0
        total_d_ap, total_d_an, n_stat_batches = (0.0, 0.0, 0)

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _, stats = model.datl(
                batch["activities"],
                batch["features"],
                batch["lengths"],
                batch["domain_id"],
                batch["rt_bucket"],
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.datl.parameters(), cfg.gradient_clip)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            total_triplets += stats["n_triplets"]
            if stats["d_ap_mean"] is not None:
                total_d_ap += stats["d_ap_mean"]
                total_d_an += stats["d_an_mean"]
                n_stat_batches += 1

        if n_batches:
            avg_d_ap = total_d_ap / max(n_stat_batches, 1)
            avg_d_an = total_d_an / max(n_stat_batches, 1)
            avg_triplets = total_triplets / n_batches
            print(
                f"[Stage 1: DATL] epoch {epoch + 1}/{cfg.datl_epochs}  loss={total_loss / n_batches:.4f}  "
                f"d_ap={avg_d_ap:.4f}  d_an={avg_d_an:.4f}  triplets/batch={avg_triplets:.1f}  "
                f"batches_with_no_triplets={n_batches - n_stat_batches}"
            )


@torch.no_grad()
def collect_datl_embeddings(model, loader, device):
    model.datl.eval()
    all_embeddings, all_rt = ([], [])
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        h = model.datl.encoder(batch["activities"], batch["features"], batch["lengths"])
        all_embeddings.append(h.cpu().numpy())
        all_rt.append(batch["remaining_time"].cpu().numpy())
    return (np.concatenate(all_embeddings), np.concatenate(all_rt))


def _loss_components(out, batch, cfg):
    ce_loss = F.cross_entropy(out["activity_logits"], batch["next_activity"])
    mse_loss = F.mse_loss(out["rt_direct"], batch["remaining_time"])
    total = ce_loss + cfg.loss_alpha * mse_loss
    return total, ce_loss, mse_loss


def train_finetune_stage(model, train_loader, val_loader, cfg, device):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=cfg.finetune_lr)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(cfg.finetune_epochs):
        model.train()
        total_loss, total_ce, total_mse = (0.0, 0.0, 0.0)
        total_correct, total_n = 0, 0
        n_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            loss, ce_loss, mse_loss = _loss_components(out, batch, cfg)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.gradient_clip)
            optimizer.step()
            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_mse += mse_loss.item()
            n_batches += 1

            pred = out["activity_logits"].argmax(dim=-1)
            total_correct += (pred == batch["next_activity"]).sum().item()
            total_n += pred.size(0)

        val_loss, val_ce, val_mse, val_acc = _eval_loss(model, val_loader, cfg, device)
        n_batches = max(n_batches, 1)
        train_acc = total_correct / max(total_n, 1)
        print(
            f"[Stage 2: Fine-tune] epoch {epoch + 1}/{cfg.finetune_epochs}  "
            f"train_loss={total_loss / n_batches:.4f} (ce={total_ce / n_batches:.4f} mse={total_mse / n_batches:.4f}) "
            f"train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} (ce={val_ce:.4f} mse={val_mse:.4f}) val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.early_stopping_patience:
                print(f"[Stage 2] early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[Stage 2] restored best checkpoint, val_loss={best_val_loss:.4f}")


@torch.no_grad()
def _eval_loss(model, loader, cfg, device):
    model.eval()
    total, total_ce, total_mse, n = (0.0, 0.0, 0.0, 0)
    total_correct, total_n = 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        loss, ce_loss, mse_loss = _loss_components(out, batch, cfg)
        total += loss.item()
        total_ce += ce_loss.item()
        total_mse += mse_loss.item()
        n += 1

        pred = out["activity_logits"].argmax(dim=-1)
        total_correct += (pred == batch["next_activity"]).sum().item()
        total_n += pred.size(0)
    n = max(n, 1)
    val_acc = total_correct / max(total_n, 1)
    return (total / n, total_ce / n, total_mse / n, val_acc)


def prepare_stage3(model, train_loader, device):
    if model.retrieval_index is not None:
        embeddings, rt_values = collect_datl_embeddings(model, train_loader, device)
        model.build_retrieval_index(embeddings, rt_values)
    model.apply_taia_inference()


@torch.no_grad()
def evaluate(model, loader, preprocessor, device, num_activities):
    model.eval()
    all_true_act, all_pred_act = ([], [])
    all_true_rt, all_pred_rt = ([], [])

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        pred_act = out["activity_logits"].argmax(dim=-1)
        all_true_act.append(batch["next_activity"].cpu().numpy())
        all_pred_act.append(pred_act.cpu().numpy())
        all_true_rt.append(batch["remaining_time"].cpu().numpy())
        all_pred_rt.append(out["rt_final"].cpu().numpy())

    y_true_act = np.concatenate(all_true_act)
    y_pred_act = np.concatenate(all_pred_act)
    y_true_rt = preprocessor.inverse_transform_rt(np.concatenate(all_true_rt))
    y_pred_rt = preprocessor.inverse_transform_rt(np.concatenate(all_pred_rt))
    y_pred_rt = np.clip(y_pred_rt, 0.0, None)

    return {
        "accuracy": accuracy(y_true_act, y_pred_act),
        "macro_f1": macro_f1(y_true_act, y_pred_act, num_activities),
        "mae": mae(y_true_rt, y_pred_rt),
    }