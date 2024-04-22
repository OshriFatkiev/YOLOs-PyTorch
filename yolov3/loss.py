import torch
import torch.nn as nn
from torchvision.ops import box_iou, box_convert


# Defining YOLO loss class
class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, target, anchors):
        # Identifying which cells in target have objects and which have no objects
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0

        # Calculating No object loss
        no_object_loss = self.bce((pred[..., 0:1][no_obj]), (target[..., 0:1][no_obj]))

        # Reshaping anchors to match predictions
        anchors = anchors.reshape(1, 3, 1, 1, 2)

        # Box prediction confidence
        box_preds = torch.cat([self.sigmoid(pred[..., 1:3]), torch.exp(pred[..., 3:5]) * anchors], dim=-1)

        # Calculating intersection over union for prediction and target
        # ious = iou(box_preds[obj], target[..., 1:5][obj]).detach()
        ious = box_iou(
            boxes1=box_convert(box_preds, in_fmt='cxcywh', out_fmt='xyxy')[obj],
            boxes2=box_convert(target[..., 1:5], in_fmt='cxcywh', out_fmt='xyxy')[obj]
        ).detach()
        diag_ious = torch.diag(ious).unsqueeze(1).detach()
        # Calculating Object loss
        object_loss = self.mse(self.sigmoid(pred[..., 0:1][obj]), diag_ious * target[..., 0:1][obj])

        # Predicted box coordinates
        pred[..., 1:3] = self.sigmoid(pred[..., 1:3])
        # Target box coordinates
        target[..., 3:5] = torch.log(1e-9 + target[..., 3:5] / anchors)
        # Calculating box coordinate loss
        box_loss = self.mse(pred[..., 1:5][obj], target[..., 1:5][obj])

        # Claculating class loss
        # class_loss = self.cross_entropy((pred[..., 5:][obj]), target[..., 5][obj].long())

        # Claculating stellar mass loss
        target[..., 5:6] = torch.log10(target[..., 5:6]) / 10
        mass_loss = self.mse(pred[..., 5:6][obj], target[..., 5:6][obj])

        # Total loss
        return box_loss, object_loss, no_object_loss, mass_loss  # , class_loss
