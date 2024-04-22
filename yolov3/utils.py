import torch
import random
import numpy as np


def iou(box1, box2, is_pred=True):
    if is_pred:
        # IoU score for prediction and label
        # box1 (prediction) and box2 (label) are both in [x, y, width, height] format

        # Box coordinates of prediction
        b1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        b1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        b1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        b1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2

        # Box coordinates of ground truth
        b2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        b2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        b2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        b2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2

        # Get the coordinates of the intersection rectangle
        x1 = torch.max(b1_x1, b2_x1)
        y1 = torch.max(b1_y1, b2_y1)
        x2 = torch.min(b1_x2, b2_x2)
        y2 = torch.min(b1_y2, b2_y2)
        # Make sure the intersection is at least 0
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

        # Calculate the union area
        box1_area = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
        box2_area = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1))
        union = box1_area + box2_area - intersection

        # Calculate the IoU score
        epsilon = 1e-6
        iou_score = intersection / (union + epsilon)

        # Return IoU score
        return iou_score

    else:
        # IoU score based on width and height of bounding boxes

        # Calculate intersection area
        intersection_area = torch.min(box1[..., 0], box2[..., 0]) * \
                            torch.min(box1[..., 1], box2[..., 1])

        # Calculate union area
        box1_area = box1[..., 0] * box1[..., 1]
        box2_area = box2[..., 0] * box2[..., 1]
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU score
        iou_score = intersection_area / union_area

        # Return IoU score
        return iou_score


def nms(bboxes, iou_threshold, threshold):
    # Filter out bounding boxes with confidence below the threshold.
    bboxes = [box for box in bboxes if box[1] > threshold]

    # Sort the bounding boxes by confidence in descending order.
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    # Initialize the list of bounding boxes after non-maximum suppression.
    bboxes_nms = []

    while bboxes:
        # Get the first bounding box.
        first_box = bboxes.pop(0)

        # Iterate over the remaining bounding boxes.
        for box in bboxes:
            # If the bounding boxes do not overlap or if the first bounding box has
            # a higher confidence, then add the second bounding box to the list of
            # bounding boxes after non-maximum suppression.
            if box[0] != first_box[0] or iou(
                    torch.tensor(first_box[2:]),
                    torch.tensor(box[2:]),
            ) < iou_threshold:
                # Check if box is not in bboxes_nms
                if box not in bboxes_nms:
                    # Add box to bboxes_nms
                    bboxes_nms.append(box)

                    # Return bounding boxes after non-maximum suppression.
    return bboxes_nms


def convert_cells_to_bboxes(predictions, anchors, s, is_predictions=True):  # mass_estimation=True
    batch_size = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]

    if is_predictions:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        # xy
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        # wh
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        mass = torch.pow(10, 10 * predictions[..., 5:6])
        best_class = torch.argmax(predictions[..., 6:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        mass = predictions[..., 5:6]
        best_class = predictions[..., 6:7]

    cell_indices = (torch.arange(s)
                    .repeat(predictions.shape[0], 3, s, 1)
                    .unsqueeze(-1)
                    .to(predictions.device))

    # Calculate x, y, width and height with proper scaling
    x = (1 / s) * (box_predictions[..., 0:1] + cell_indices)
    y = (1 / s) * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    width_height = (1 / s) * box_predictions[..., 2:4]

    converted_bboxes = torch.cat(
        (best_class, scores, x, y, width_height, mass), dim=-1
    ).reshape(batch_size, num_anchors * s * s, 7)

    # Filter out bboxes that do not represents an object
    # converted_bboxes = converted_bboxes[converted_bboxes[...,1] > 0]

    # Returning the reshaped and converted bounding box list
    return converted_bboxes.tolist()


# Function to save checkpoint
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("Saving checkpoint...")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print("Done saving checkpoint.\n")


# Function to load checkpoint
def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Done loading checkpoint.\n")


def seed_everything(seed=13):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    box1 = torch.tensor([2,2,2,2]).unsqueeze(0)
    box2 = torch.tensor([3,2,2,2]).unsqueeze(0)

    res1 = iou(box1=box1, box2=box2, is_pred=True)

    from torchvision.ops import box_iou, box_convert
    res2 = box_iou(
        boxes1=box_convert(box1, in_fmt='cxcywh', out_fmt='xyxy'),
        boxes2=box_convert(box2, in_fmt='cxcywh', out_fmt='xyxy'),
    )

    assert res1.item() - res2.item() < 1e-6, res1.item()
    print("Passed IoU test!")
