import torch
from torch import nn
from typing import Optional


from metrics import KNN

def asymmetric_loss(p, z):
    z = z.detach()  # stop gradient
    p = nn.functional.normalize(p, dim=1)
    z = nn.functional.normalize(z, dim=1)

    return -(p * z).sum(dim=1).mean()


def criterion(model_outputs: dict):
    loss1 = asymmetric_loss(model_outputs["p1"], model_outputs["z2"])
    loss2 = asymmetric_loss(model_outputs["p2"], model_outputs["z1"])
    return 0.5 * loss1 + 0.5 * loss2


class Trainer:
    def __init__(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.util.data.DataLoader],
        model: torch.nn.Module,
        epochs: int = 10,
        learning_rate: float = 0.06,
        weight_decay: float = 5e-4,
        device: str = "cuda",
    ):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def train_epoch(self):
        for aug1, aug2, _, _ in self.train_dataloader:
            aug1 = aug1.to(self.device)
            aug2 = aug2.to(self.device)
            model_outputs = self.model(aug1, aug2)

    def validation(self) -> float:
        train_features = []
        train_labels = []
        val_features = []
        val_labels = []

        with torch.no_grad():
            for batch in self.train_dataloader:
                _, _, img, label = batch
                img = img.cuda()
                train_features.append(self.model.backbone(img).cpu())
                train_labels.append(label.cpu())

            train_features = torch.concat(train_features, dim=0).numpy()
            train_labels = torch.concat(train_labels, dim=0).numpy()

            for batch in self.val_dataloader:
                img, label = batch
                img = img.cuda()
                val_features.append(self.model.backbone(img).cpu())
                val_labels.append(label.cpu())

            val_features = torch.concat(val_features, dim=0).numpy()
            val_labels = torch.concat(val_labels, dim=0).numpy()

            knn = KNN(n_classes=10, top_k=[1], knn_k=200)
            val_acc = knn.knn_acc(
                val_features, val_labels, train_features, train_labels
            )

        return list(val_acc.values())[0]

