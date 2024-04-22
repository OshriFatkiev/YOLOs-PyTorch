import torch
import torch.optim as optim
import torchvision

import os
import astropy.units as u
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from model import YOLOv3
from loss import YOLOLoss
from dataset import Dataset
from utils import seed_everything, convert_cells_to_bboxes, load_checkpoint


# Function to plot images with bounding boxes and class labels
def plot_image(image, boxes, true_boxes=None, vmin=0, vmax=0.7, plot_channel=0, class_labels=None):
    # Getting the color map from matplotlib
    if class_labels is None:
        class_labels = ['clump']

    colour_map = plt.get_cmap("tab20b")
    # Getting 20 different colors from the color map for 20 different classes
    colors = [colour_map(i) for i in np.linspace(0, 1, len(class_labels))]

    img = np.array(image)
    height, width, _ = img.shape

    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(img[..., plot_channel], vmin=vmin, vmax=vmax)

    # Plotting the bounding boxes and labels over the image
    for box in boxes:
        class_pred = box[0]
        confidence = box[1]

        # Get the center x and y coordinates
        box = box[2:]

        # Get the upper left corner coordinates
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2

        # Create a Rectangle patch with the bounding box
        rect = patches.Rectangle(
            xy=(upper_left_x * width, upper_left_y * height),
            width=box[2] * width,
            height=box[3] * height,
            linewidth=2,
            edgecolor="white",  # colors[int(class_pred)],
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.text(
            upper_left_x * width, upper_left_y * height,
            f'{confidence:.2f}',
            fontsize=12,
            fontweight='bold',
            color='white'
        )

        # Add class name to the patch
        if len(class_labels) > 1:
            plt.text(
                upper_left_x * width,
                upper_left_y * height,
                s=class_labels[int(class_pred)],
                color="cyan",
                verticalalignment="top",
                bbox={"color": 'black', "pad": 0},  # colors[int(class_pred)]
            )

    if true_boxes is not None:
        for box in true_boxes:
            class_pred = box[0]

            # Get the center x and y coordinates
            box = box[2:]

            # Get the upper left corner coordinates
            upper_left_x = box[0] - box[2] / 2
            upper_left_y = box[1] - box[3] / 2

            # Create a Rectangle patch with the bounding box
            rect = patches.Rectangle(
                xy=(upper_left_x * width, upper_left_y * height),
                width=box[2] * width,
                height=box[3] * height,
                linewidth=2,
                edgecolor="magenta",
                facecolor="none",
            )
            ax.add_patch(rect)

    # plt.axis('off')
    plt.show()


def create_dict(images):
    d = dict()
    for bboxes in images:
        for bbox in bboxes:
            stellar_mass = bbox[6]
            m = np.round(np.log10(stellar_mass) * 2) / 2
            # m = np.floor(np.log10(bbox[6])) or np.round(np.log10(bbox[6]))
            if m not in d.keys():
                d[m] = 1
            else:
                d[m] += 1

    sorted_dict = dict(sorted(d.items()))
    return sorted_dict


def metrics_per_mass(true_bboxes, pred_bboxes, iou_thres=0.5, filenames=None, true_from_catalog=False, eps=0.25):
    """
    Compute the metrics per mass and plot the completeness and recall per mass.
    true_bboxes: list format [num images][bounding boxes][p, c, x, y, w, h, m].
    pred_bboxes: list format [num images][bounding boxes][p, c, x, y, w, h, m].
    """
    if true_from_catalog:
        print('Getting true clumps from the catalog...')
        true_bboxes = []
        for filename in filenames:
            _, galaxy, cam, scfctr = filename.split('_')
            cam = cam[3:]
            scfctr = scfctr[1:]
            # Get the true bboxes and append them to the list
            res = get_true_from_catalog(galaxy, cam, scfctr, fltr='f200w', img_size=64)
            true_bboxes.append(res)
        print('Done scanning the catalog.')

    true_dict = create_dict(true_bboxes)
    pred_dict = create_dict(pred_bboxes)

    masses = sorted(set(list(true_dict) + list(pred_dict)))

    # Create a dictionary or storing the true positives per mass bin
    tp_dict = dict()
    for mass in masses:
        tp_dict[mass] = 0

    num_images = len(true_bboxes)
    for img_idx in range(num_images):
        true_numpy = np.array(true_bboxes[img_idx])
        pred_numpy = np.array(pred_bboxes[img_idx])

        if not len(pred_bboxes[img_idx]):
            continue

        true_xyxy = torchvision.ops.box_convert(boxes=torch.tensor(true_numpy)[..., 2:6], in_fmt='cxcywh', out_fmt='xyxy')
        pred_xyxy = torchvision.ops.box_convert(boxes=torch.tensor(pred_numpy)[..., 2:6], in_fmt='cxcywh', out_fmt='xyxy')
        used_indices = list()

        for true_index, tbox in enumerate(true_xyxy):
            for pred_index, pbox in enumerate(pred_xyxy):
                res = torchvision.ops.box_iou(tbox.unsqueeze(0), pbox.unsqueeze(0))
                if res > iou_thres and pred_index not in used_indices:
                    # Check if the mass is close enough
                    if np.abs(np.log10(true_numpy[true_index][..., 6]) - np.log10(pred_numpy[pred_index][..., 6])) < eps:
                        true_mass = np.round(np.log10(true_numpy[true_index][..., 6]) * 2) / 2
                        tp_dict[true_mass] += 1

                        pred_mass = np.round(np.log10(pred_numpy[pred_index][..., 6]) * 2) / 2
                        if true_mass != pred_mass:
                            pred_dict[true_mass] += 1
                            pred_dict[pred_mass] -= 1

                        used_indices.append(pred_index)
                        break  # move to the next true bounding box

    # purity := tp / (tp + fp)
    # (tp + fp) = all the pred bboxes
    purity = dict()
    for mass in masses:
        if mass in pred_dict.keys():
            res = tp_dict[mass] / pred_dict[mass]
        else:
            res = 0
        purity[mass] = res

    # completeness := tp / (tp + fn)
    # (tp + fn) = all the true bboxes
    completeness = dict()
    for mass in masses:
        if mass in true_dict.keys():
            res = tp_dict[mass] / true_dict[mass]
        else:
            res = 0
        completeness[mass] = res

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # print(f'tp_dict {tp_dict}')
    # print(f'true_dict {true_dict}')
    # print(f'pred_dict {pred_dict}')

    names = list(purity.keys())
    values = np.array(list(purity.values()))
    ax[0].bar(range(len(purity)), values, tick_label=names, alpha=0.7)
    mean_purity = np.mean([item for item in purity.values() if item])
    ax[0].axhline(mean_purity, ls='--', c='r')
    ax[0].set_ylim([0, 1.1])
    ax[0].set_title(rf'Mean: ${mean_purity:.2f}$', fontsize=16)
    ax[0].set_ylabel(r'Purity', fontsize=16)
    ax[0].set_xlabel(r'$\log M_{stellar}$', fontsize=16)

    names = list(completeness.keys())
    values = np.array(list(completeness.values()))
    ax[1].bar(range(len(completeness)), values, tick_label=names, alpha=0.7, color='#ff7f0e')
    mean_comp = np.mean([item for item in completeness.values() if item])
    ax[1].axhline(mean_comp, ls='--', c='r')
    ax[1].set_ylim([0, 1.1])
    ax[1].set_title(rf'Mean: ${mean_comp:.2f}$', fontsize=16)
    ax[1].set_ylabel(r'Completeness', fontsize=16)
    ax[1].set_xlabel(r'$\log M_{stellar}$', fontsize=16)

    plt.tight_layout()
    plt.show()


def get_fit_data(header):
    cam_z = np.array([header['CAMDIRX'], header['CAMDIRY'], header['CAMDIRZ']])
    cam_y = np.array([header['CAMUPX'], header['CAMUPY'], header['CAMUPZ']])
    cam_x = np.cross(cam_y, cam_z)

    npix = header['NAXIS1']
    pixscale = header['PIXKPC']

    npix_real = npix
    pixscale_real = pixscale

    return npix_real, pixscale_real, np.array([cam_x, cam_y, cam_z])


def get_sunrise_center(header):
    return header['translate_originX'], header['translate_originY'], header['translate_originZ']


def get_true_from_catalog(galaxy, cam, scfctr, fltr='f200w', img_size=64):
    """
    Returns all the true boundix boxes of the clumps straight from the catalog.
    """
    CAT_PATH = '/sci/home/omryg/astro-home/old-omryg-home/mycatalogs/npy/'
    galaxy_cat = np.load(CAT_PATH + 'galaxy_cat_new_g3.npy')
    clump_cat = np.load(CAT_PATH + 'clump_cat_g3.npy')

    AU_TO_PC = 4.8481e-6
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.27, Ob0=0.045)

    VELA_JWST_PATH = f'/sci/archive/dekel/lab_share/vela_jwst_mock/vela3_mocks/{fltr}/cam{cam}/jwst/nircam/{fltr}/'
    path = VELA_JWST_PATH + f'hlsp_vela_jwst_nircam_vela{galaxy}-cam{cam}-a{scfctr}_{fltr}_v3-2_sim-smc.fits'

    if not os.path.exists(path):
        return

    with fits.open(path) as hdul:
        data = hdul[0].data * 1
        header_true = hdul[0].header
        header_sec = hdul[1].header
        header_sfrhist = hdul[6].header
        _, _, rotmat = get_fit_data(header_sec)

        pixscale = (cosmo.angular_diameter_distance(header_true['redshift']) * 0.03 * AU_TO_PC).to(u.kpc).value
        img_physical_size = pixscale * img_size

        npix = header_true['NAXIS1']
        if npix < img_size:
            return

        lbound = (npix // 2) - (img_size // 2)
        ubound = (npix // 2) + (img_size // 2)
        cropped_data = data[lbound:ubound, lbound:ubound]

        centers, radii, mbar, ms = [], [], [], []
        for clump in clump_cat[
            (clump_cat['galaxy'] == galaxy) &
            (clump_cat['a'] == scfctr)
        ]:

            shift = np.array(list(galaxy_cat[(galaxy_cat['galaxy'] == galaxy) & \
                                             (galaxy_cat['a'] == scfctr)][['center_x', 'center_y', 'center_z']][0])) \
                    - get_sunrise_center(header_sfrhist)

            pos = rotmat.dot(np.array(list(clump[['x', 'y', 'z']])) + shift)
            img_physical_size = pixscale * img_size  # in kpc

            # Skip if out of borders
            if np.abs(pos[0]) > img_physical_size / 2 or np.abs(pos[1]) > img_physical_size / 2:
                continue

            # Skip if bulge
            if clump['lifetime'] == 'Bulg':
                continue

            # Skip if not a sfZLC
            if clump['lifetime'] == 'ZLC' and clump['SFR'] <= 0:  # / (clump['disc_SFR'] + 1e-10) < 1e-3:
                continue

            x, y = (pos[0] / pixscale + img_size / 2), (pos[1] / pixscale + img_size / 2)
            r = 3 * clump['Rc'] / pixscale

            centers.append((x, y))
            radii.append(r)
            mbar.append(clump['Mbar'])
            ms.append(clump['Ms'])

        # Don't add image without any clumps
        if radii == [] or centers == []:
            return

    xywhm = [[
        0.0,
        1.0,
        centers[i][0] / img_size,
        centers[i][1] / img_size,
        radii[i] / img_size,
        radii[i] / img_size,
        # mbar[i],
        ms[i]]
        for i in range(len(centers))]

    return xywhm


def plot_scatter_mass(true_boxes, pred_boxes, iou_thres=0.5):
    plt.figure(figsize=(6, 6))

    counter = 0
    n_clumps = 0
    n_images = len(true_boxes)

    # for img_idx in range(n_images):
    #     n_clumps += len(true_boxes[img_idx])
    #     for tbox in true_boxes[img_idx]:
    #         for pbox in pred_boxes[img_idx]:
    #
    #             # Skip if there is no object
    #             if tbox[1] == 0 or pbox[1] == 0:
    #                 continue
    #
    #             t = torch.tensor(tbox).unsqueeze(0)
    #             box1_xyxy = torchvision.ops.box_convert(boxes=t[..., 2:6], in_fmt='cxcywh', out_fmt='xyxy')
    #
    #             p = torch.tensor(pbox).unsqueeze(0)
    #             box2_xyxy = torchvision.ops.box_convert(boxes=p[..., 2:6], in_fmt='cxcywh', out_fmt='xyxy')
    #
    #             # Calculate the IOU of the pred and the true bbox
    #             res = torchvision.ops.box_iou(boxes1=box1_xyxy, boxes2=box2_xyxy)
    #             # print(res)
    #
    #             # print(iou(torch.tensor(tbox), torch.tensor(pbox)))
    #             # print(res)
    #             if res > iou_thres:
    #                 counter += 1
    #                 plt.scatter(x=np.log10(tbox[6]), y=np.log10(pbox[6]), s=3, c='red')
    #                 break

    for img_idx in range(n_images):
        n_clumps += len(true_boxes[img_idx])

        if not len(pred_boxes[img_idx]):
            continue

        true_xyxy = torchvision.ops.box_convert(
            boxes=torch.tensor(true_boxes[img_idx])[..., 2:6],
            in_fmt='cxcywh',
            out_fmt='xyxy'
        )
        pred_xyxy = torchvision.ops.box_convert(
            boxes=torch.tensor(pred_boxes[img_idx])[..., 2:6],
            in_fmt='cxcywh',
            out_fmt='xyxy'
        )

        used_indices = list()
        for _, tbox in enumerate(true_xyxy):
            for pred_index, pbox in enumerate(pred_xyxy):

                res = torchvision.ops.box_iou(tbox.unsqueeze(0), pbox.unsqueeze(0))
                if res > iou_thres and pred_index not in used_indices:
                    counter += 1

                    true_mass = tbox[6]
                    pred_mass = pbox[6]
                    plt.scatter(x=np.log10(true_mass), y=np.log10(pred_mass), s=3, c='red')

                    used_indices.append(pred_index)
                    break  # move to the next true bounding box (tbox)

    print(f'Number of matches: {counter}')
    print(f'Total number of true clumps: {n_clumps}')

    plt.plot(np.linspace(0, 10, 100), np.linspace(0, 10, 100), '--', c='black', linewidth=1)

    plt.xlabel('true', fontsize=16)
    plt.ylabel('pred', fontsize=16)
    plt.xlim([5, 10])
    plt.ylim([5, 10])
    plt.show()


def visualize_losses(epochs, train_box_loss, train_object_loss, train_no_object_loss, train_mass_loss,
                     val_box_loss, val_object_loss, val_no_object_loss, val_mass_loss):
    fig, ax = plt.subplots(1, 4, figsize=(14, 4))

    x = np.arange(epochs)

    ax[0].plot(x, train_box_loss.detach().numpy(), label='train')
    ax[0].plot(x, val_box_loss.detach().numpy(), label='val')
    ax[0].legend()

    ax[1].plot(x, train_object_loss.detach().numpy(), label='train')
    ax[1].plot(x, val_object_loss.detach().numpy(), label='val')
    ax[1].legend()

    ax[2].plot(x, train_no_object_loss.detach().numpy(), label='train')
    ax[2].plot(x, val_no_object_loss.detach().numpy(), label='val')
    ax[2].legend()

    ax[3].plot(x, train_mass_loss.detach().numpy(), label='train')
    ax[3].plot(x, val_mass_loss.detach().numpy(), label='val')
    ax[3].legend()

    ax[0].set_title('box_loss', fontsize=14)
    ax[1].set_title('obj_loss', fontsize=14)
    ax[2].set_title('noobj_loss', fontsize=14)
    ax[3].set_title('mass_loss', fontsize=14)

    # ax[0].set_ylim([0.16, 0.25])
    # ax[1].set_ylim([0, 0.25])
    # ax[2].set_ylim([0.1, 0.4])
    # ax[3].set_ylim([0, 0.02])

    plt.tight_layout()
    plt.show()


def plot_true_vs_pred(image_dir, label_dir, ckpt, image_size, learning_rate, in_channels,
                      conf_thres, iou_thres, vmin, vmax, plot_channel, num_classes=1,
                      batch_size=32, num_workers=1, device='cpu', imgs_to_plot=10):
    seed_everything()

    grid_sizes = [image_size // 32, image_size // 16, image_size // 8]
    anchors = [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ]

    # Loading the checkpoint
    model = YOLOv3(in_channels=in_channels, num_classes=num_classes).float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = YOLOLoss()
    scaler = torch.cuda.amp.GradScaler()
    load_checkpoint(ckpt, model, optimizer, learning_rate, device)
    model.eval()

    # Defining the train dataset
    test_dataset = Dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        image_size=image_size,
        grid_sizes=grid_sizes,
        num_classes=num_classes,
        anchors=anchors,
        augment=False
    )

    # Defining the train data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    nms_true_bboxes = list()  # [[] for _ in range(n_images)]
    nms_pred_bboxes = list()  # [[] for _ in range(n_images)]
    filenames = list()
    images = list()

    scaled_anchors = (
            torch.tensor(anchors) *
            torch.tensor(grid_sizes).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)

    count = 0
    for batch_index, (x, y, z) in enumerate(test_loader):
        x = x.permute(0, 3, 1, 2).float().to(device)  # (batch_size,64,64,6) to (batch_size,6,64,64)
        filenames += z
        images += x

        with torch.no_grad():
            output = model(x)

            # Get pred bounding boxes for each scale:
            for image_index in range(x.shape[0]):

                pred_temp_list, true_temp_list = [], []

                for scale_index in range(3):
                    pred_boxes_scale_i = convert_cells_to_bboxes(
                        output[scale_index][image_index].unsqueeze(0),
                        scaled_anchors[scale_index],
                        s=grid_sizes[scale_index],
                        is_predictions=True
                    )
                    for box in pred_boxes_scale_i:
                        pred_temp_list += box

                    true_boxes_scale_i = convert_cells_to_bboxes(
                        y[scale_index][image_index].unsqueeze(0),
                        scaled_anchors[scale_index],
                        s=grid_sizes[scale_index],
                        is_predictions=False
                    )
                    for box in true_boxes_scale_i:
                        true_temp_list += box

                # Take only the boxes with confidence > conf_thres
                p = torch.tensor(pred_temp_list)
                p = p[p[..., 1] > conf_thres]

                # Preform Non-Maximal Suppression
                keep_indices_p = torchvision.ops.nms(
                    boxes=torchvision.ops.box_convert(p[..., 2:6], in_fmt='cxcywh', out_fmt='xyxy'),
                    scores=p[..., 1],
                    iou_threshold=iou_thres
                )

                # Store the relevant bounding boxes in the list
                nms_pred_bboxes.append(p[keep_indices_p].tolist())

                t = torch.tensor(true_temp_list)
                t = t[t[..., 1] > 0]

                # Preform Non-Maximal Suppression
                keep_indices_t = torchvision.ops.nms(
                    boxes=torchvision.ops.box_convert(t[..., 2:6], in_fmt='cxcywh', out_fmt='xyxy'),
                    scores=t[..., 1],
                    iou_threshold=0.9
                )
                # Store the relevant bounding boxes in the list
                nms_true_bboxes.append(t[keep_indices_t].tolist())

                if count < imgs_to_plot:
                    print(f"{z[image_index]}")
                    plot_image(
                        x[image_index].permute(1, 2, 0).detach().cpu(),
                        boxes=p[keep_indices_p].tolist(),
                        true_boxes=t[keep_indices_t].tolist(),
                        vmin=vmin,
                        vmax=vmax,
                        plot_channel=plot_channel
                    )
                count += 1
    return nms_true_bboxes, nms_pred_bboxes

