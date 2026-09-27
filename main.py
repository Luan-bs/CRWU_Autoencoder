from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "CWRU_48k_load_1_CNN_data.npz"


class ConvAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                8, 32, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                32, 16, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                16, 1, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de dados nao encontrado: {path}")

    with np.load(path, allow_pickle=False) as archive:
        if not {"data", "labels"}.issubset(archive.files):
            raise ValueError("O NPZ precisa conter os arrays 'data' e 'labels'.")
        samples = np.asarray(archive["data"], dtype=np.float32)
        labels = np.asarray(archive["labels"]).astype(str)

    if samples.shape[0] != labels.shape[0]:
        raise ValueError("A quantidade de amostras e rotulos no NPZ diverge.")

    # O arquivo CWRU fornecido armazena cada amostra como uma matriz 32x32.
    samples = samples.reshape(samples.shape[0], 1, -1)
    return samples, labels


def split_by_class(
    labels: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    calibration_indices: list[int] = []
    test_indices: list[int] = []

    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        train_end = int(len(indices) * 0.70)
        calibration_end = int(len(indices) * 0.85)
        if train_end == 0 or calibration_end == train_end:
            raise ValueError(f"Poucas amostras para dividir a classe {label!r}.")

        if label.lower().startswith("normal"):
            train_indices.extend(indices[:train_end])
            calibration_indices.extend(indices[train_end:calibration_end])
        test_indices.extend(indices[calibration_end:])

    return (
        np.asarray(train_indices, dtype=np.int64),
        np.asarray(calibration_indices, dtype=np.int64),
        np.asarray(test_indices, dtype=np.int64),
    )


def reconstruction_errors(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    mse_values: list[np.ndarray] = []
    mae_values: list[np.ndarray] = []

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            mse_values.append((batch - reconstructed).pow(2).mean(dim=(1, 2)).cpu().numpy())
            mae_values.append((batch - reconstructed).abs().mean(dim=(1, 2)).cpu().numpy())

    return np.concatenate(mse_values), np.concatenate(mae_values)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    calibration_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> list[tuple[float, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    history: list[tuple[float, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * batch.size(0)
            train_count += batch.size(0)

        train_mse = train_total / train_count
        calibration_mse, _ = reconstruction_errors(
            model, calibration_loader, device
        )
        calibration_mean = float(calibration_mse.mean())
        history.append((train_mse, calibration_mean))
        print(
            f"Epoca {epoch:03d}/{epochs} | "
            f"MSE treino: {train_mse:.6f} | "
            f"MSE calibracao normal: {calibration_mean:.6f}"
        )

    return history


def save_results(
    output_dir: Path,
    history: list[tuple[float, float]],
    threshold: float,
    test_mse: np.ndarray,
    test_mae: np.ndarray,
    test_targets: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = (test_mse > threshold).astype(int)
    auc = roc_auc_score(test_targets, test_mse)
    report = classification_report(
        test_targets,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Falha"],
        zero_division=0,
    )
    report_text = (
        "Avaliacao do autoencoder Conv1D\n"
        f"Threshold (media + 3 desvios, calibracao normal): {threshold:.8f}\n"
        f"MSE medio no teste: {test_mse.mean():.8f}\n"
        f"MAE medio no teste: {test_mae.mean():.8f}\n"
        f"ROC-AUC: {auc:.4f}\n\n"
        f"{report}"
    )
    (output_dir / "evaluation_report.txt").write_text(report_text, encoding="utf-8")
    print(f"\n{report_text}")

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot([item[0] for item in history], label="Treino normal")
    axis.plot([item[1] for item in history], label="Calibracao normal")
    axis.set(title="Erro de reconstrucao por epoca", xlabel="Epoca", ylabel="MSE")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_loss.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    sns.histplot(test_mse[test_targets == 0], bins=40, label="Normal", ax=axis)
    sns.histplot(test_mse[test_targets == 1], bins=40, label="Falha", ax=axis)
    axis.axvline(threshold, color="black", linestyle="--", label="Threshold")
    axis.set(title="Distribuicao do erro de reconstrucao", xlabel="MSE", ylabel="Contagem")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "error_distribution.png", dpi=150)
    plt.close(figure)

    matrix = confusion_matrix(test_targets, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Falha"],
        yticklabels=["Normal", "Falha"],
        ax=axis,
    )
    axis.set(title="Matriz de confusao", xlabel="Predito", ylabel="Real")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina e avalia um autoencoder Conv1D nos dados CWRU."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs e batch-size precisam ser positivos.")

    torch.manual_seed(args.seed)
    samples, labels = load_dataset(args.data)
    train_indices, calibration_indices, test_indices = split_by_class(
        labels, args.seed
    )
    test_targets = np.where(
        np.char.lower(labels[test_indices]).startswith("normal"), 0, 1
    )

    train_samples = samples[train_indices]
    scale = float(train_samples.std())
    if scale == 0.0:
        raise ValueError("Os dados normais de treino possuem variancia zero.")
    mean = float(train_samples.mean())
    samples = (samples - mean) / scale

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(samples[train_indices])),
        batch_size=args.batch_size,
        shuffle=True,
    )
    calibration_loader = DataLoader(
        TensorDataset(torch.from_numpy(samples[calibration_indices])),
        batch_size=args.batch_size,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(samples[test_indices])),
        batch_size=args.batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder().to(device)
    history = train_model(
        model,
        train_loader,
        calibration_loader,
        device,
        args.epochs,
        args.learning_rate,
    )

    calibration_mse, _ = reconstruction_errors(model, calibration_loader, device)
    threshold = float(calibration_mse.mean() + 3 * calibration_mse.std())
    test_mse, test_mae = reconstruction_errors(model, test_loader, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_length": int(samples.shape[-1]),
            "normalization_mean": mean,
            "normalization_std": scale,
            "threshold": threshold,
        },
        args.output_dir / "conv1d_autoencoder.pt",
    )
    save_results(
        args.output_dir, history, threshold, test_mse, test_mae, test_targets
    )
    print(f"Dispositivo: {device}")
    print(f"Resultados salvos em: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()