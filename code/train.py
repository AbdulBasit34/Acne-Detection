# # standard library
# import os, sys

# # third-party library
# import numpy as np
# import collections
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms
# from torch.utils.data import DataLoader
# from dataset import dataset_processing
# from timeit import default_timer as timer
# from utils.report import report_precision_se_sp_yi, report_mae_mse
# from utils.utils import Logger, AverageMeter, time_to_str, weights_init
# from utils.genLD import genLD
# from model.resnet50 import resnet50
# import torch.backends.cudnn as cudnn
# from transforms.affine_transforms import *
# from utils.metrics import sensitivity_score, specificity_score, geometric_mean_score
# import time
# import warnings
# warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from sklearn.metrics import confusion_matrix

# # Hyper Parameters
# BATCH_SIZE = 32
# BATCH_SIZE_TEST = 20
# LR = 0.001
# NUM_WORKERS = 4
# NUM_CLASSES = 4

# # Corrected log file name (no colons)
# LOG_FILE_NAME = './logs/log_' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.log'

# # Ensure the logs directory exists
# os.makedirs('./logs', exist_ok=True)

# lr_steps = [30, 60, 90, 120]

# np.random.seed(42)

# DATA_PATH = '/media/abdul/New Volume1/inventra/ACNE04/Classification/JPEGImages'

# log = Logger()
# log.open(LOG_FILE_NAME, mode="a")

# print("Training started...")

# def criterion(lesions_num):
#     if lesions_num <= 5:
#         return 0
#     elif lesions_num <= 20:
#         return 1
#     elif lesions_num <= 50:
#         return 2
#     else:
#         return 3

# def trainval_test(cross_val_index, sigma, lam):
#     TRAIN_FILE = f'/media/abdul/New Volume1/inventra/ACNE04/Detection/VOC2007/ImageSets/Main/NNEW_trainval_{cross_val_index}.txt'
#     TEST_FILE = f'/media/abdul/New Volume1/inventra/ACNE04/Detection/VOC2007/ImageSets/Main/NNEW_test_{cross_val_index}.txt'

#     normalize = transforms.Normalize(mean=[0.45815152, 0.361242, 0.29348266], std=[0.2814769, 0.226306, 0.20132513])

#     dset_train = dataset_processing.DatasetProcessing(
#         DATA_PATH, TRAIN_FILE, transform=transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.RandomCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             RandomRotate(rotation_range=20),
#             normalize,
#         ]))

#     dset_test = dataset_processing.DatasetProcessing(
#         DATA_PATH, TEST_FILE, transform=transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             normalize,
#         ]))

#     train_loader = DataLoader(dset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
#     test_loader = DataLoader(dset_test, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

#     cnn = resnet50().cuda()
#     cudnn.benchmark = True

#     optimizer = torch.optim.SGD(cnn.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

#     loss_func = nn.CrossEntropyLoss().cuda()
#     kl_loss_1 = nn.KLDivLoss().cuda()
#     kl_loss_2 = nn.KLDivLoss().cuda()
#     kl_loss_3 = nn.KLDivLoss().cuda()

#     best_gmean = 0.0
#     best_model_path = f'./logs/best_model_fold_{cross_val_index}.pth'
#     gmean_epoch=1
#     def adjust_learning_rate_new(optimizer, decay=0.5):
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = decay * param_group['lr']

#     start = timer()
#     for epoch in range(lr_steps[-1]):
#         if epoch in lr_steps:
#             adjust_learning_rate_new(optimizer, 0.5)

#         cnn.train()
#         print(f"Starting epoch {epoch + 1}/{lr_steps[-1]}")
#         for step, (b_x, b_y, b_l) in enumerate(train_loader):
#             b_x = b_x.cuda()
#             b_l = b_l.numpy() - 1

#             ld = genLD(b_l, sigma, 'klloss', 65)
#             ld_4 = np.vstack((np.sum(ld[:, :5], 1), np.sum(ld[:, 5:20], 1), np.sum(ld[:, 20:50], 1), np.sum(ld[:, 50:], 1))).transpose()
#             ld = torch.from_numpy(ld).cuda().float()
#             ld_4 = torch.from_numpy(ld_4).cuda().float()

#             cls, cou, cou2cls = cnn(b_x, None)
#             loss_cls = kl_loss_1(torch.log(cls), ld_4) * 4.0
#             loss_cou = kl_loss_2(torch.log(cou), ld) * 65.0
#             loss_cls_cou = kl_loss_3(torch.log(cou2cls), ld_4) * 4.0
#             loss = (loss_cls + loss_cls_cou) * 0.5 * lam + loss_cou * (1.0 - lam)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         cnn.eval()
#         y_true, y_pred = [], []
#         with torch.no_grad():
#             for test_x, test_y, _ in test_loader:
#                 test_x = test_x.cuda()
#                 outputs = cnn(test_x, None)[2]
#                 _, preds = torch.max(outputs, 1)
#                 y_true.extend(test_y.cpu().numpy())
#                 y_pred.extend(preds.cpu().numpy())

#         sensitivity = sensitivity_score(y_true, y_pred, average='macro')
#         specificity = specificity_score(y_true, y_pred, average='macro')
#         gmean = geometric_mean_score(y_true, y_pred, average='macro')

#         if isinstance(gmean, np.ndarray):
#             gmean = gmean.mean()

#         print(f"Epoch {epoch + 1} - Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, G-Mean: {gmean:.4f}")

#         conf_matrix = confusion_matrix(y_true, y_pred)
#         print("Confusion Matrix:\n", conf_matrix)

#         if gmean > best_gmean and epoch > 70:
#             best_gmean = gmean
#             torch.save(cnn.state_dict(), best_model_path)
#             print(f"New best model saved with G-Mean: {best_gmean:.4f}")

# cross_val_lists = ['0', '1', '2', '3', '4']
# for cross_val_index in cross_val_lists:
#     log.write(f'\n\ncross_val_index: {cross_val_index}\n\n')
#     print(f"Starting cross-validation fold {cross_val_index}")
#     trainval_test(cross_val_index, sigma=3.0, lam=0.6)

# print("Training completed.")
# import os, sys
# import time
# import warnings

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.backends.cudnn as cudnn
# from torch.utils.data import DataLoader, ConcatDataset
# from torchvision import transforms
# from sklearn.metrics import confusion_matrix

# from timeit import default_timer as timer
# from dataset import dataset_processing
# from utils.report import report_precision_se_sp_yi, report_mae_mse
# from utils.utils import Logger, AverageMeter, time_to_str, weights_init
# from utils.genLD import genLD
# from utils.metrics import sensitivity_score, specificity_score, geometric_mean_score
# from model.resnet50 import resnet50
# from transforms.affine_transforms import *
# from transforms.image_transforms import (RandomBrightness,
#     RandomContrast) # example custom affine
# from transforms.distortion_transforms import (
#     RandomChoiceScramble,
#     RandomChoiceBlur
# )
# # ^ import other transforms as needed

# warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# # -------------------
# # Hyperparameters
# # -------------------
# BATCH_SIZE = 32
# BATCH_SIZE_TEST = 20
# LR = 0.001
# NUM_WORKERS = 12
# NUM_CLASSES = 4
# lr_steps = [30, 60, 90, 120]
# np.random.seed(42)

# DATA_PATH = '/media/abdul/New Volume1/inventra/ACNE04/Classification/JPEGImages'

# # Logging
# os.makedirs('./logs', exist_ok=True)
# LOG_FILE_NAME = './logs/log_' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.log'
# log = Logger()
# log.open(LOG_FILE_NAME, mode="a")

# print("Training started...")

# # -------------------
# # Helper function
# # -------------------
# def criterion(lesions_num):
#     """Map raw lesion count to a class index 0..3"""
#     if lesions_num <= 5:
#         return 0
#     elif lesions_num <= 20:
#         return 1
#     elif lesions_num <= 50:
#         return 2
#     else:
#         return 3

# # -------------------
# # Main training/validation
# # -------------------
# def trainval_test(cross_val_index, sigma, lam):
#     TRAIN_FILE = f'/media/abdul/New Volume1/inventra/ACNE04/Detection/VOC2007/ImageSets/Main/NNEW_trainval_{cross_val_index}.txt'
#     TEST_FILE = f'/media/abdul/New Volume1/inventra/ACNE04/Detection/VOC2007/ImageSets/Main/NNEW_test_{cross_val_index}.txt'

#     # Normalization used after our custom transforms
#     normalize = transforms.Normalize(
#         mean=[0.45815152, 0.361242, 0.29348266],
#         std=[0.2814769, 0.226306, 0.20132513]
#     )

#     # -------------
#     # 1) Original transform (minimal)
#     # -------------
#     transform_original = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         normalize,
#     ])

#     # -------------
#     # 2) Augmented transform (uses your custom random transforms)
#     # -------------
#     transform_augmented = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.RandomCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         # -- Insert your custom transforms here:
#         RandomRotate(rotation_range=20),
#         RandomBrightness(min_val=-0.3, max_val=0.3),
#         RandomContrast(min_val=0.8, max_val=1.2),
#         RandomChoiceScramble(blocksizes=[8,16,32]),
#         RandomChoiceBlur(thresholds=[64.0, 32.0, 16.0, 8.0, 4.0], order=5),
#         # -- End custom transforms
#         normalize,
#     ])

#     # -------------
#     # Create Dataset(s)
#     # -------------
#     # We create TWO datasets on the same list of images:
#     #  - one with the original transforms
#     #  - one with the augmented transforms
#     dset_train_original = dataset_processing.DatasetProcessing(
#         data_path=DATA_PATH,
#         img_filename=TRAIN_FILE,
#         transform=transform_original
#     )

#     dset_train_augmented = dataset_processing.DatasetProcessing(
#         data_path=DATA_PATH,
#         img_filename=TRAIN_FILE,
#         transform=transform_augmented
#     )

#     # Combine them so each epoch sees original + augmented
#     dset_train = ConcatDataset([dset_train_original, dset_train_augmented])

#     # For testing, typically minimal transforms only
#     dset_test = dataset_processing.DatasetProcessing(
#         DATA_PATH, TEST_FILE,
#         transform=transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             normalize,
#         ])
#     )

#     # -------------
#     # DataLoaders
#     # -------------
#     train_loader = DataLoader(
#         dset_train,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=NUM_WORKERS,
#         pin_memory=True
#     )

#     test_loader = DataLoader(
#         dset_test,
#         batch_size=BATCH_SIZE_TEST,
#         shuffle=False,
#         num_workers=NUM_WORKERS,
#         pin_memory=True
#     )

#     # -------------
#     # Model, optim, etc.
#     # -------------
#     cnn = resnet50().cuda()
#     cudnn.benchmark = True

#     optimizer = torch.optim.SGD(cnn.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
#     kl_loss_1 = nn.KLDivLoss().cuda()
#     kl_loss_2 = nn.KLDivLoss().cuda()
#     kl_loss_3 = nn.KLDivLoss().cuda()

#     best_gmean = 0.0
#     best_model_path = f'./logs/best_model_fold_{cross_val_index}.pth'

#     def adjust_learning_rate_new(optimizer, decay=0.5):
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = decay * param_group['lr']

#     start = timer()

#     # Here we assume we train for 120 epochs, but you can change as needed
#     for epoch in range(lr_steps[-1]):
#         # Adjust LR if needed
#         if epoch in lr_steps:
#             adjust_learning_rate_new(optimizer, 0.5)

#         cnn.train()
#         print(f"\nFold {cross_val_index} | Starting epoch {epoch+1}/{lr_steps[-1]}")

#         for step, (b_x, b_y, b_l) in enumerate(train_loader):
#             b_x = b_x.cuda()
#             b_l = b_l.numpy() - 1  # shift to 0..3 if originally 1..4

#             # Generate label distributions (LDs)
#             ld = genLD(b_l, sigma, 'klloss', 65)
#             ld_4 = np.vstack((
#                 np.sum(ld[:, :5], 1),
#                 np.sum(ld[:, 5:20], 1),
#                 np.sum(ld[:, 20:50], 1),
#                 np.sum(ld[:, 50:], 1)
#             )).transpose()

#             ld = torch.from_numpy(ld).cuda().float()
#             ld_4 = torch.from_numpy(ld_4).cuda().float()

#             # Forward pass
#             cls, cou, cou2cls = cnn(b_x, None)
#             loss_cls = kl_loss_1(torch.log(cls), ld_4) * 4.0
#             loss_cou = kl_loss_2(torch.log(cou), ld) * 65.0
#             loss_cls_cou = kl_loss_3(torch.log(cou2cls), ld_4) * 4.0

#             # Combined loss
#             loss = (loss_cls + loss_cls_cou) * 0.5 * lam + loss_cou * (1.0 - lam)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         # -------------
#         # Validation
#         # -------------
#         cnn.eval()
#         y_true, y_pred = [], []
#         with torch.no_grad():
#             for test_x, test_y, _ in test_loader:
#                 test_x = test_x.cuda()
#                 outputs = cnn(test_x, None)[2]  # cou2cls predictions
#                 _, preds = torch.max(outputs, 1)
#                 y_true.extend(test_y.cpu().numpy())
#                 y_pred.extend(preds.cpu().numpy())

#         sensitivity = sensitivity_score(y_true, y_pred, average='macro')
#         specificity = specificity_score(y_true, y_pred, average='macro')
#         gmean = geometric_mean_score(y_true, y_pred, average='macro')

#         if isinstance(sensitivity, np.ndarray):
#             sensitivity = sensitivity.mean() 
#         if isinstance(specificity, np.ndarray):
#             specificity = specificity.mean() 
#         if isinstance(gmean, np.ndarray):
#             gmean = gmean.mean() 
#              # if it's array-like

#         print(f"Epoch {epoch+1} - Sensitivity: {sensitivity:.4f}, "
#               f"Specificity: {specificity:.4f}, G-Mean: {gmean:.4f}")

#         conf_matrix = confusion_matrix(y_true, y_pred)
#         print("Confusion Matrix:\n", conf_matrix)

#         # Save best model (if desired after epoch 70)
#         if gmean > best_gmean and epoch > 70:
#             best_gmean = gmean
#             torch.save(cnn.state_dict(), best_model_path)
#             print(f"New best model saved with G-Mean: {best_gmean:.4f}")

#     elapsed = timer() - start
#     print(f"Fold {cross_val_index} completed in {elapsed/60:.2f} min.")

import os, sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.metrics import confusion_matrix

from timeit import default_timer as timer
from dataset import dataset_processing
from utils.report import report_precision_se_sp_yi, report_mae_mse
from utils.utils import Logger, AverageMeter, time_to_str, weights_init
from utils.genLD import genLD
from utils.metrics import sensitivity_score, specificity_score, geometric_mean_score
from model.resnet50 import resnet50

from transforms.affine_transforms import RandomRotate, RandomTranslate, RandomShear, RandomZoom, RandomAffine
from transforms.image_transforms import RandomBrightness, RandomContrast, RandomSaturation, RandomGrayscale, RandomGamma
from transforms.distortion_transforms import RandomChoiceScramble, RandomChoiceBlur, Scramble, Blur

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# -------------------
# Hyperparameters
# -------------------
BATCH_SIZE = 32
BATCH_SIZE_TEST = 20
LR = 0.001
NUM_WORKERS = 12
NUM_CLASSES = 4
lr_steps = [30, 60, 90, 120]
np.random.seed(42)

DATA_PATH = '/media/abdul/New Volume1/inventra/ACNE04/Classification/JPEGImages'

# Logging
os.makedirs('./logs', exist_ok=True)
LOG_FILE_NAME = './logs/log_' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.log'
log = Logger()
log.open(LOG_FILE_NAME, mode="a")

print("Training started...")

# -------------------
# Helper function
# -------------------
def criterion(lesions_num):
    """Map raw lesion count to a class index 0..3"""
    if lesions_num <= 5:
        return 0
    elif lesions_num <= 20:
        return 1
    elif lesions_num <= 50:
        return 2
    else:
        return 3

# -------------------
# Main training/validation
# -------------------
def trainval_test(cross_val_index, sigma, lam):
    TRAIN_FILE = f'/media/abdul/New Volume1/inventra/ACNE04/Detection/VOC2007/ImageSets/Main/NNEW_trainval_{cross_val_index}.txt'
    TEST_FILE = f'/media/abdul/New Volume1/inventra/ACNE04/Detection/VOC2007/ImageSets/Main/NNEW_test_{cross_val_index}.txt'

    normalize = transforms.Normalize(
        mean=[0.45815152, 0.361242, 0.29348266],
        std=[0.2814769, 0.226306, 0.20132513]
    )

# 1) Original transform (minimal)
    transform_original = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # Converts to [0,1] float32
        transforms.ConvertImageDtype(torch.float),  # Ensure float dtype
        normalize,
    ])

    # 2) Augmented transform (with random distortions and affine/image transforms)
    transform_augmented = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Converts to [0,1] float32
        transforms.ConvertImageDtype(torch.float),  # Ensure float dtype

        # ---- Distortion Transforms (4) ----
        RandomChoiceScramble(blocksizes=[8, 16, 32]),
        RandomChoiceBlur(thresholds=[64.0, 32.0, 16.0, 8.0, 4.0], order=5),
        Scramble(blocksize=16),
        Blur(threshold=8.0, order=3),

        # ---- Affine Transforms (5) ----
        RandomRotate(rotation_range=20),
        RandomTranslate(translation_range=(0.1, 0.1)),
        RandomShear(shear_range=10),
        RandomZoom(zoom_range=(0.8, 1.2)),
        RandomAffine(rotation_range=15, translation_range=(0.1, 0.1)),

        # ---- Image Transforms (5) ----
        RandomBrightness(min_val=-0.3, max_val=0.3),
        RandomContrast(min_val=0.8, max_val=1.2),
        RandomSaturation(min_val=-0.2, max_val=0.2),
        RandomGrayscale(p=0.2),
        RandomGamma(min_val=0.8, max_val=1.2),

        transforms.ConvertImageDtype(torch.float),  # Ensure float dtype after all transforms
        normalize,
    ])



    dset_train_original = dataset_processing.DatasetProcessing(
        data_path=DATA_PATH,
        img_filename=TRAIN_FILE,
        transform=transform_original
    )

    dset_train_augmented = dataset_processing.DatasetProcessing(
        data_path=DATA_PATH,
        img_filename=TRAIN_FILE,
        transform=transform_augmented
    )

    dset_train = ConcatDataset([dset_train_original, dset_train_augmented])

    dset_test = dataset_processing.DatasetProcessing(
        DATA_PATH, TEST_FILE,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = DataLoader(
        dset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        dset_test,
        batch_size=BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    cnn = resnet50().cuda()
    cudnn.benchmark = True

    optimizer = torch.optim.SGD(cnn.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    kl_loss_1, kl_loss_2, kl_loss_3 = nn.KLDivLoss().cuda(), nn.KLDivLoss().cuda(), nn.KLDivLoss().cuda()

    best_gmean = 0.0
    best_model_path = f'./logs/best_model_fold_{cross_val_index}.pth'

    def adjust_learning_rate_new(optimizer, decay=0.5):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay

    start = timer()

    for epoch in range(lr_steps[-1]):
        # Adjust LR if needed
        if epoch in lr_steps:
            adjust_learning_rate_new(optimizer, 0.5)

        cnn.train()
        print(f"\nFold {cross_val_index} | Starting epoch {epoch+1}/{lr_steps[-1]}")

        for step, (b_x, b_y, b_l) in enumerate(train_loader):
            b_x = b_x.cuda()
            b_l = b_l.numpy() - 1  # shift to 0..3 if originally 1..4

            # Generate label distributions (LDs)
            ld = genLD(b_l, sigma, 'klloss', 65)
            ld_4 = np.vstack((
                np.sum(ld[:, :5], 1),
                np.sum(ld[:, 5:20], 1),
                np.sum(ld[:, 20:50], 1),
                np.sum(ld[:, 50:], 1)
            )).transpose()

            ld = torch.from_numpy(ld).cuda().float()
            ld_4 = torch.from_numpy(ld_4).cuda().float()

            # Forward pass
            cls, cou, cou2cls = cnn(b_x, None)
            loss_cls = kl_loss_1(torch.log(cls), ld_4) * 4.0
            loss_cou = kl_loss_2(torch.log(cou), ld) * 65.0
            loss_cls_cou = kl_loss_3(torch.log(cou2cls), ld_4) * 4.0

            # Combined loss
            loss = (loss_cls + loss_cls_cou) * 0.5 * lam + loss_cou * (1.0 - lam)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ✅ Print progress every 10 steps
            if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                print(f"Epoch [{epoch+1}/{lr_steps[-1]}] | Step [{step+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        # -------------
        # Validation
        # -------------
        cnn.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for test_x, test_y, _ in test_loader:
                test_x = test_x.cuda()
                outputs = cnn(test_x, None)[2]  # cou2cls predictions
                _, preds = torch.max(outputs, 1)
                y_true.extend(test_y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        sensitivity = sensitivity_score(y_true, y_pred, average='macro')
        specificity = specificity_score(y_true, y_pred, average='macro')
        gmean = geometric_mean_score(y_true, y_pred, average='macro')

        if isinstance(sensitivity, np.ndarray):
            sensitivity = sensitivity.mean() 
        if isinstance(specificity, np.ndarray):
            specificity = specificity.mean() 
        if isinstance(gmean, np.ndarray):
            gmean = gmean.mean()  # if it's array-like

        print(f"Epoch {epoch+1} - Sensitivity: {sensitivity:.4f}, "
            f"Specificity: {specificity:.4f}, G-Mean: {gmean:.4f}")

        conf_matrix = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:\n", conf_matrix)

        # Save best model (if desired after epoch 70)
        if gmean > best_gmean and epoch > 70:
            best_gmean = gmean
            torch.save(cnn.state_dict(), best_model_path)
            print(f"New best model saved with G-Mean: {best_gmean:.4f}")

    elapsed = timer() - start
    print(f"Fold {cross_val_index} completed in {elapsed/60:.2f} min.")

# -------------------
# Run Cross-Validation
# -------------------
cross_val_lists = ['0', '1', '2', '3', '4']
for cross_val_index in cross_val_lists:
    log.write(f'\n\ncross_val_index: {cross_val_index}\n\n')
    print(f"Starting cross-validation fold {cross_val_index}")
    trainval_test(cross_val_index, sigma=3.0, lam=0.6)

print("Training completed.")
