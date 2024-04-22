import os
import glob
import torch
import cv2

import numpy as np
from utils import iou

from astropy.visualization import MinMaxInterval, AsinhStretch


# transform = MinMaxInterval() # AsinhStretch()
# Create a dataset class to load the images and labels from the folder

#                      0       1  2    3      4          5             6
# bboxes format: [class_label, x, y, width, height, stellar_mass]
# target format: [probability, x, y, width, height, stellar_mass, class_label]

class Dataset(torch.utils.data.Dataset):
    def __init__(
            self, image_dir, label_dir, anchors,
            image_size=320, grid_sizes=[10, 20, 40],
            num_classes=1, augment=False
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.augment = augment
        self.transform = MinMaxInterval()

        self.label_name = self.label_dir.split('/')[-3]
        # print(self.label_name)

        labels = sorted([f for f in glob.glob(f"{label_dir}/*.txt")])
        self.images, self.labels = [], []
        for label in labels:
            im = label.replace(self.label_name, 'images').replace('txt', 'npy')
            if os.path.exists(im):
                self.labels.append(label)
                self.images.append(im)

        # Grid sizes for each scale
        self.grid_sizes = grid_sizes

        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.num_classes = num_classes
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = np.load(self.images[idx])
        # print(self.images[idx])

        filename = self.images[idx].split(os.sep)[-1][:-4]

        if image.shape[1] != self.image_size:
            image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
            image = self.transform(image)

        label_path = self.images[idx].replace('images', self.label_name).replace('.npy', '.txt')
        labels = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2)

        # Delete the baryonic mass column
        labels = np.delete(labels, 5, axis=1)

        # We are applying roll to move class label to the last column: [x, y, width, height, stellar_mass, class_label]
        bboxes = np.roll(labels, 5, axis=1)

        # ADD HERE AUGMENTATIONS
        if self.augment:
            # img = (img - mean * max_pixel_value) / (std * max_pixel_value)

            # 90 degrees rotations multiples
            n_rots = np.random.randint(low=1, high=5)
            for i in range(n_rots):
                image = np.rot90(image)
                if len(bboxes):
                    bboxes_ = np.zeros_like(bboxes)
                    bboxes_[..., 0] = bboxes[..., 1]  # Rotate x
                    bboxes_[..., 1] = 1 - bboxes[..., 0]  # Rotate y
                    bboxes_[..., 2] = bboxes[..., 3]  # Rotate width
                    bboxes_[..., 3] = bboxes[..., 2]  # Rotate height
                    bboxes_[..., 4] = bboxes[..., 4]  # Keep stellar_mass in place

                    bboxes = bboxes_

            # Flip up-down: y --> (1 - y)
            if np.random.random() < 0.5:
                image = np.flipud(image)
                if len(bboxes):
                    bboxes[:, 1] = 1 - bboxes[:, 1]

            # Flip left-right: x --> (1 - x)
            if np.random.random() < 0.5:
                image = np.fliplr(image)
                if len(bboxes):
                    bboxes[:, 0] = 1 - bboxes[:, 0]

            # TODO: Cutout augmentation:
            # if np.random.random() < 1:
            #     x1, x2 = np.random.random(2)
            #     y1, y2 = np.random.random(2)
            #
            #     # Top left coordinates
            #     tl = (
            #         int(min(x1, x2) * self.image_size),
            #         int(min(y1, y2) * self.image_size)
            #     )
            #     # Bottom right coordinates
            #     br = (
            #         int(max(x1, x2) * self.image_size),
            #         int(max(y1, y2) * self.image_size)
            #     )
            #
            #     image[tl[0]:br[0], tl[1]:br[1], ...] = 0
            #
            #     if len(bboxes):
            #         bboxes_ = []
            #         for row_idx, box in enumerate(bboxes):
            #             # print(tl[0], box[1] * self.image_size, br[0])
            #             # print(tl[1], box[0] * self.image_size, br[1])
            #             if not (tl[0] <= box[1] * self.image_size <= br[0] and
            #                 tl[1] <= box[0] * self.image_size <= br[1]):
            #                 bboxes_.append(box)
            #         bboxes = np.array(bboxes_)

        bboxes = torch.from_numpy(bboxes.copy())

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        # target : [probabilities, x, y, width, height, class_label, stellar_mass]
        targets = [torch.zeros((self.num_anchors_per_scale, s, s, 7)) for s in self.grid_sizes]

        # Identify anchor box and cell for each bounding box
        for box in bboxes:
            # Calculate iou of bounding box with anchor boxes
            iou_anchors = iou(box[2:4], self.anchors, is_pred=False)
            # Selecting the best anchor box
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)

            x, y, width, height, class_label, stellar_mass = box

            # At each scale, assigning the bounding box to the best matching anchor box
            has_anchor = [False] * 3
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale

                # Identifying the grid size for the scale
                s = self.grid_sizes[scale_idx]

                # Identifying the cell to which the bounding box belongs
                i, j = int(s * y), int(s * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                # Check if the anchor box is already assigned
                if not anchor_taken and not has_anchor[scale_idx]:

                    # Set the probability to 1
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1

                    # Calculating the center of the bounding box relative to the cell
                    x_cell, y_cell = s * x - j, s * y - i

                    # Calculating the width and height of the bounding box
                    # relative to the cell
                    width_cell, height_cell = (width * s, height * s)

                    # Identify the box coordinates
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                    # Assigning the box coordinates to the target
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates

                    # Assigning the class label to the target
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)

                    # Assigning the class label to the target
                    targets[scale_idx][anchor_on_scale, i, j, 6] = stellar_mass

                    # print(targets[scale_idx][anchor_on_scale, i, j, :])

                    # Set the anchor box as assigned for the scale
                    has_anchor[scale_idx] = True

                # If the anchor box is already assigned, check if the
                # IoU is greater than the threshold
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    # Set the probability to -1 to ignore the anchor box
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        # Return the image and the target
        return image.copy(), tuple(targets), filename



