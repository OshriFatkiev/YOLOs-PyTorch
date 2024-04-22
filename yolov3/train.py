import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from utils import seed_everything
from model import YOLOv3
from loss import YOLOLoss
from dataset import Dataset


def training_loop(loader, model, optimizer, loss_fn, scaler, scaled_anchors, device):
    model.train()
    progress_bar = tqdm(loader, leave=True)

    # Initializing a list to store the losses
    losses = list()
    box_losses = list()
    object_losses = list()
    no_object_losses = list()
    mass_losses = list()

    # Iterating over the training data
    for _, (x, y, _) in enumerate(progress_bar):
        x = x.permute(0, 3, 1, 2).float().to(device)
        # y0, y1, y2 = (y[0].to(device), y[1].to(device), y[2].to(device))
        ys = [y[0].to(device), y[1].to(device), y[2].to(device)]

        # TODO: Visualize just before the net
        # true_bboxes = [[] for _ in range(x.shape[0])]
        # for i in range(3):
        #    batch_size, _, grid_size, _, _ = y[i].shape
        #    anchor = anchors[i]
        #    boxes_scale_i = convert_cells_to_bboxes(
        #        y[i],
        #        anchor,
        #        s=grid_size,
        #        is_predictions=False
        #    )
        #    for idx, (box) in enumerate(boxes_scale_i):
        #        true_bboxes[idx] += box

        # TODO: Plotting the image with bounding boxes for each image in the batch
        # nms_true_bboxes = nms(true_bboxes[0], iou_threshold=1, threshold=0.99)
        # plot_image(x[0].permute(1,2,0).detach(), nms_true_bboxes)

        with torch.cuda.amp.autocast():
            # Getting the model predictions
            outputs = model(x)
            box_loss, object_loss, no_object_loss, mass_loss = 0, 0, 0, 0
            for i in range(3):
                res = loss_fn(outputs[i], ys[i], scaled_anchors[i])
                box_loss += res[0]
                object_loss += res[1]
                no_object_loss += res[2]
                mass_loss += res[3]

        box_losses.append(box_loss.item())
        object_losses.append(object_loss.item())
        no_object_losses.append(no_object_loss.item())
        mass_losses.append(mass_loss.item())

        # Add the loss to the list
        loss = box_loss + object_loss + no_object_loss  # + mass_loss
        # losses.append(loss.item())

        # Reset gradients
        optimizer.zero_grad()

        # Backpropagate the loss
        scaler.scale(loss).backward()

        # Optimization step
        scaler.step(optimizer)

        # Update the scaler for next iteration
        scaler.update()

        # update progress bar with loss
        # mean_loss = sum(losses) / len(losses)

        mean_box = sum(box_losses) / len(box_losses)
        mean_obj = sum(object_losses) / len(object_losses)
        mean_noobj = sum(no_object_losses) / len(no_object_losses)
        mean_mass = sum(mass_losses) / len(mass_losses)

        mean_loss = mean_box + mean_obj + mean_noobj + mean_mass
        progress_bar.set_postfix(loss=mean_loss)

    return mean_box, mean_obj, mean_noobj, mean_mass


def validation_loop(loader, model, loss_fn, scaled_anchors, device):
    model.eval()

    # Initializing a list to store the losses
    box_losses = list()
    object_losses = list()
    no_object_losses = list()
    mass_losses = list()

    # Iterating over the validation data
    with torch.no_grad():
        for _, (x, y, _) in enumerate(loader):
            x = x.permute(0, 3, 1, 2).float().to(device)
            ys = [y[0].to(device), y[1].to(device), y[2].to(device)]

            with torch.cuda.amp.autocast():
                # Getting the model predictions
                outputs = model(x)
                box_loss, object_loss, no_object_loss, mass_loss = 0, 0, 0, 0
                for i in range(3):
                    res = loss_fn(outputs[i], ys[i], scaled_anchors[i])
                    box_loss += res[0]
                    object_loss += res[1]
                    no_object_loss += res[2]
                    mass_loss += res[3]

            box_losses.append(box_loss.item())
            object_losses.append(object_loss.item())
            no_object_losses.append(no_object_loss.item())
            mass_losses.append(mass_loss.item())

    mean_box = sum(box_losses) / len(box_losses)
    mean_obj = sum(object_losses) / len(object_losses)
    mean_noobj = sum(no_object_losses) / len(no_object_losses)
    mean_mass = sum(mass_losses) / len(mass_losses)

    return mean_box, mean_obj, mean_noobj, mean_mass


def train(train_image_dir, train_label_dir, val_image_dir, val_label_dir, batch_size, learning_rate, epochs,
          num_workers, image_size, in_channels, num_classes, device):
    seed_everything()

    grid_sizes = [image_size // 32, image_size // 16, image_size // 8]
    anchors = [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ]

    model = YOLOv3(in_channels=in_channels, num_classes=num_classes).float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  #, weight_decay=5e-4)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = YOLOLoss()

    # Defining the scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    train_dataset = Dataset(
        image_dir=train_image_dir,
        label_dir=train_label_dir,
        image_size=image_size,
        grid_sizes=grid_sizes,
        num_classes=num_classes,
        anchors=anchors,
        augment=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )

    val_dataset = Dataset(
        image_dir=val_image_dir,
        label_dir=val_label_dir,
        image_size=image_size,
        grid_sizes=grid_sizes,
        num_classes=num_classes,
        anchors=anchors,
        augment=False
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )

    # Scaling the anchors
    scaled_anchors = (
            torch.tensor(anchors) *
            torch.tensor(grid_sizes).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)

    print(f"image size: {image_size}, batch size: {batch_size}, learning rate: {learning_rate}")

    train_loss = np.zeros(epochs)
    val_loss = np.zeros(epochs)

    train_box_loss = torch.zeros(epochs)
    train_obj_loss = torch.zeros(epochs)
    train_noobj_loss = torch.zeros(epochs)
    train_mass_loss = torch.zeros(epochs)

    val_box_loss = torch.zeros(epochs)
    val_obj_loss = torch.zeros(epochs)
    val_noobj_loss = torch.zeros(epochs)
    val_mass_loss = torch.zeros(epochs)

    for e in range(epochs):
        train = training_loop(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, device)

        train_box_loss[e] = train[0]
        train_obj_loss[e] = train[1]
        train_noobj_loss[e] = train[2]
        train_mass_loss[e] = train[3]

        train_loss[e] = sum(train)

        val = validation_loop(val_loader, model, loss_fn, scaled_anchors, device)

        val_box_loss[e] = val[0]
        val_obj_loss[e] = val[1]
        val_noobj_loss[e] = val[2]
        val_mass_loss[e] = val[3]

        val_loss[e] = sum(val)

        print(f"Epoch: {e + 1}/{epochs}, avg. train loss: {train_loss[e]:.3f}, avg. validation loss: {val_loss[e]:.3f}")

        return (model, optimizer, train_loss, train_box_loss, train_obj_loss, train_noobj_loss, train_mass_loss,
                val_loss, val_box_loss, val_obj_loss, val_noobj_loss, val_mass_loss)
