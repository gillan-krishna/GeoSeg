from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.openearth_dataset import *
from geoseg.models.UNetFormer import UNetFormer
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# training hparam
max_epoch = 1000
ignore_index = 0
train_batch_size = 108
val_batch_size = 20
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES
# class_weightage = Tensor([0.16666667*1.15, 0.16666667*1.5, 0.16666667*2.5, 0.16666667, 0.16666667,
#        0.16666667]).cuda()

weights_name = "unetformer-512-ms-crop"
weights_path = f"model_weights/openearth/{weights_name}"
test_weights_name = "unetformer-512-ms-crop"
log_name = f'openearth/{weights_name}'
monitor = 'val/mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
# pretrained_ckpt_path = 'pretrain_weights/stseg_base.pth'
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None
# resume_ckpt_path = 'model_weights/openearth/ftunetformer-512-ms-crop/ftunetformer-512-ms-crop.ckpt'

#  define the network
net = UNetFormer(num_classes=num_classes)

# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = True

train_dataset = OpenEarthDataset(data_root='data/openearth/train', mode='train',
                                 mosaic_ratio=0.25, transform=train_aug)

val_dataset = OpenEarthDataset(data_root='data/openearth/val', transform=val_aug)
# test_dataset = OpenEarthDataset(data_root='data/openearth/test',
#                                 transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=1,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=1,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
