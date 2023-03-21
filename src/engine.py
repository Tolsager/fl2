import torch
import tqdm
import torchmetrics
from torch import nn
from typing import Optional
import math


from src.metrics import KNN


# def asymmetric_loss(p, z):
#     z = z.detach()  # stop gradient

#     return -torch.nn.functional.cosine_similarity(p, z, dim=-1).mean()


# def criterion(model_outputs: dict):
#     loss1 = asymmetric_loss(model_outputs["p1"], model_outputs["z2"])
#     loss2 = asymmetric_loss(model_outputs["p2"], model_outputs["z1"])
#     return 0.5 * loss1 + 0.5 * loss2
class SimSiamLoss(nn.Module):
    def __init__(self, version="simplified"):
        super().__init__()
        self.ver = version

    def asymmetric_loss(self, p, z):
        if self.ver == "original":
            z = z.detach()  # stop gradient

            p = nn.functional.normalize(p, dim=1)
            z = nn.functional.normalize(z, dim=1)

            return -(p * z).sum(dim=1).mean()

        elif self.ver == "simplified":
            z = z.detach()  # stop gradient
            return -nn.functional.cosine_similarity(p, z, dim=-1).mean()

    def forward(self, z1, z2, p1, p2):

        loss1 = self.asymmetric_loss(p1, z2)
        loss2 = self.asymmetric_loss(p2, z1)

        return 0.5 * loss1 + 0.5 * loss2


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args["learning_rate"]
    # cosine lr schedule
    lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args["epochs"]))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class Trainer:
    def __init__(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader],
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
        self.optimizer = torch.optim.SGD(
            model.parameters(), learning_rate, momentum=0.9, weight_decay=weight_decay
        )
        self.avg_train_loss = torchmetrics.MeanMetric()
        self.criterion = SimSiamLoss()

    def train_epoch(self) -> None:
        self.model.train()
        # for aug1, aug2, _, _ in tqdm.tqdm(self.train_dataloader):
        for i, (images, _) in enumerate(self.train_dataloader):

            # aug1 = aug1.to(self.device)
            # aug2 = aug2.to(self.device)
            images[0] = images[0].cuda()
            images[1] = images[1].cuda()
            model_outputs = self.model(im_aug1=images[0], im_aug2=images[1])
            loss = self.criterion(
                model_outputs["z1"],
                model_outputs["z2"],
                model_outputs["p1"],
                model_outputs["p2"],
            )

            # optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss = loss.detach().cpu().item()
            self.avg_train_loss.update(loss)

    def train(self) -> None:
        for epoch in tqdm.trange(self.epochs):
            adjust_learning_rate(
                self.optimizer,
                epoch=epoch,
                args={"learning_rate": self.learning_rate, "epochs": self.epochs},
            )
            self.avg_train_loss.reset()
            self.train_epoch()
            val_acc = self.validation()
            avg_train_loss = self.avg_train_loss.compute()
            print(f"Epoch: {epoch}")
            print(f"Average train loss: {avg_train_loss}")
            print(f"Validation accuracy: {val_acc}")

    def validation(self) -> float:
        self.model.eval()
        train_features = []
        train_labels = []
        val_features = []
        val_labels = []

        with torch.no_grad():
            for batch in self.train_dataloader:
                images, label = batch
                img = images[-1]
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
