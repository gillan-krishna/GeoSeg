import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from tools.cfg import py2cfg
import os 
import torch
from torch import nn
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from lightning.pytorch.loggers import CSVLogger, NeptuneLogger
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from neptune.types import File

# torch.set_float32_matmul_precision("medium")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net

        self.loss = config.loss

        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']

        prediction = self.net(img)
        loss = self.loss(prediction, mask)

        if self.config.use_aux_loss:
            # pre_mask = nn.Softmax(dim=1)(prediction[0])
            pre_mask = nn.Softmax(dim=1)(prediction)
        else:
            pre_mask = nn.Softmax(dim=1)(prediction)

        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        return {"loss": loss}

    def on_train_epoch_end(self):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'cropland' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'openearth' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_train.F1())

        OA = np.nanmean(self.metrics_train.OA())
        iou_per_class = self.metrics_train.Intersection_over_Union()
        f1_per_class = self.metrics_train.F1()
        pr_per_class = self.metrics_train.Precision()
        re_per_class = self.metrics_train.Recall()
        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA}
        print('train:', eval_value)

        iou_value = dict(zip(self.config.classes, iou_per_class))
        f1_value = dict(zip(self.config.classes, f1_per_class))
        pr_value = dict(zip(self.config.classes, pr_per_class))
        re_value = dict(zip(self.config.classes, re_per_class))
        # cm = self.metrics_train.confusion_matrix
        # print(iou_value)
        self.metrics_train.reset()

        log_dict = {'train/mIoU': mIoU, 'train/F1': F1, 'train/OA': OA} | {
            f'train/IoU_{class_name}': iou_value[class_name]
            for class_name in self.config.classes
        } | {
            f'train/f1_{class_name}': f1_value[class_name]
            for class_name in self.config.classes
        } | {
            f'train/pr_{class_name}': pr_value[class_name]
            for class_name in self.config.classes
        } | {
            f'train/re_{class_name}': re_value[class_name]
            for class_name in self.config.classes
        }
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)

        # cm_df = pd.DataFrame(cm,
        #             index = self.config.classes,
        #             columns = self.config.classes
        #              )

        # plt.figure(figsize=(10,8))
        # sns.heatmap(cm_df, annot=True)
        # plt.title('Recall Matrix')
        # plt.ylabel('Actual Values')
        # plt.xlabel('Predicted Values')
        # save_path = f'/home/gillan/unetformer/reports/train_cm_{self.trainer.current_epoch}.png'
        # plt.savefig(save_path, bbox_inches='tight')
        # plt.close()
        # self.logger.experiment["train/precision_matrix"].append(File.from_path(save_path))

    def validation_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        prediction = self.forward(img)
        pre_mask = nn.Softmax(dim=1)(prediction)
        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        loss_val = self.loss(prediction, mask)
        return {"loss_val": loss_val}

    def on_validation_epoch_end(self):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'cropland' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'openearth' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_val.F1())

        OA = np.nanmean(self.metrics_val.OA())
        iou_per_class = self.metrics_val.Intersection_over_Union()
        f1_per_class = self.metrics_val.F1()
        pr_per_class = self.metrics_train.Precision()
        re_per_class = self.metrics_train.Recall()

        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA}
        print('val:', eval_value)
        iou_value = dict(zip(self.config.classes, iou_per_class))
        f1_value = dict(zip(self.config.classes, f1_per_class))
        pr_value = dict(zip(self.config.classes, pr_per_class))
        re_value = dict(zip(self.config.classes, re_per_class))
        # cm = self.metrics_val.confusion_matrix
        # print(iou_value)

        self.metrics_val.reset()
        log_dict = {'val/mIoU': mIoU, 'val/F1': F1, 'val/OA': OA} | {
            f'val/IoU_{class_name}': iou_value[class_name]
            for class_name in self.config.classes
        }| {
            f'val/f1_{class_name}': f1_value[class_name]
            for class_name in self.config.classes
        } | {
            f'val/pr_{class_name}': pr_value[class_name]
            for class_name in self.config.classes
        } | {
            f'val/re_{class_name}': re_value[class_name]
            for class_name in self.config.classes
        }
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)
        # cm_df = pd.DataFrame(cm,
        #             index = self.config.classes,
        #             columns = self.config.classes
        #              )
        # cm_df = cm_df.div(cm_df.sum(axis=1), axis=0).fillna(0)

        # plt.figure(figsize=(10,8))
        # sns.heatmap(cm_df, annot=True)
        # plt.title('Recall Matrix')
        # plt.ylabel('Actual Values')
        # plt.xlabel('Predicted Values')
        # save_path = f'/home/gillan/unetformer/reports/val_cm_{self.trainer.current_epoch}.png'
        # plt.savefig(save_path, bbox_inches='tight')
        # plt.close()
        # self.logger.experiment["val/precision_matrix"].append(File.from_path(save_path))

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):

        return self.config.train_loader

    def val_dataloader(self):

        return self.config.val_loader


# training
def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)

    checkpoint_callback = ModelCheckpoint(save_top_k=config.save_top_k, monitor=config.monitor,
                                          save_last=config.save_last, mode=config.monitor_mode,
                                          dirpath=config.weights_path,
                                          filename=config.weights_name)
    # logger = CSVLogger('lightning_logs', name=config.log_name)
    logger = NeptuneLogger(project="gillan-k/OEM-UF",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyYWRiZGVlNC04NjA2LTRlMmYtODE4OS0zYWQ4NjFhYTEyMDIifQ==",
)

    model = Supervision_Train(config)
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)

    trainer = pl.Trainer(devices=config.gpus, max_epochs=config.max_epoch, accelerator='auto',
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=[checkpoint_callback], strategy='auto',
                         logger=logger)
    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)


if __name__ == "__main__":
   main()
