import contextlib
import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
import os
from torch.optim.lr_scheduler import MultiStepLR
import wandb
from pytorch_lightning.loggers import WandbLogger
from configs import constants as _C
# resolved dynamically by globals() in get_decoder() below — keep this import.
from ..models_2d.mvhead import MvCameraHMR, MLP, MammaNetDecoder # noqa: F401
from ..models_2d.loss import JointGNLLLoss
from ..models_2d.mask_proc import MaskEmbedding
from .utils.visualization import compare_results_denseldmks2d
from ..backbone.vit import ViT
from ...datasets import MixedWebDataset

import torch.nn as nn


def get_decoder(cfg):
    class_name = cfg.model['decoder']['layer_name']
    class_params = {k: v for k, v in cfg.model['decoder'].items() if k != 'layer_name'}

    # Retrieve the class by its name (assuming it's globally accessible)
    decoder_class = globals()[class_name]

    # Instantiate the class with the parameters from cfg
    return decoder_class(**class_params), class_name


def _focal_contact_loss(inputs, targets):
    return torchvision.ops.sigmoid_focal_loss(
        inputs, targets, alpha=0.9, gamma=2.0, reduction='mean')


class DenseLdmks2DViT(pl.LightningModule):
    def __init__(self, cfg, viz_dir=None):
        super(DenseLdmks2DViT, self).__init__()

        self.cfg = cfg
        self.optimizer_cfg = cfg.optimizer.optimizer
        self.lr_cfg = cfg.optimizer.lr_config
        self.train_data_cfg = cfg.data['train']
        self.val_data_cfg = cfg.data['val'] if 'val' in cfg.data else None

        backbone_cfg = {k: v for k, v in cfg.model['backbone'].items() if k != 'type'}
        self.backbone_cfg = backbone_cfg
        self.freeze_backbone = cfg.model['freeze_backbone']
        self.backbone = ViT(**backbone_cfg).to(self.device)
        vitpose_ckpt = _C.PATHS.PRETRAINED_VITPOSE_CKPT_PTH
        if vitpose_ckpt and os.path.isfile(vitpose_ckpt):
            self.map_state_dict_from_vitpose(vitpose_ckpt)
        else:
            # Inference path: the trained checkpoint loaded right after
            # fully overwrites the backbone, so the pretrained init is a
            # no-op. Training callers that genuinely need this file will
            # fail later with a more specific error.
            print(f"Skipping pretrained ViTPose init (not found at: {vitpose_ckpt}).")
        patch_size_mask = self.backbone_cfg['patch_size']//4

        if self.freeze_backbone:
            print("Freezing backbone!")
            self.backbone.eval()
        else:
            print("backbone is trainable!")

        self.visibility = cfg.model['decoder'].get('visibility', False)
        self.contact = cfg.model['decoder'].get('contact', False)
        self.floor_contact = cfg.model['decoder'].get('floor_contact', False)

        self.decoder, self.decoder_name = get_decoder(cfg)
        print(f"Decoder: {self.decoder_name}")

        self.criterion = JointGNLLLoss(loss_weights=cfg.loss_weights)
        self.vis_criteria = torch.nn.BCEWithLogitsLoss()

        self.viz_dir = viz_dir
        self.validation_outputs = []
        self.show_results = None

        self.use_mask = cfg.model["use_mask"]
        if self.use_mask:
            self.mask_downscaling = MaskEmbedding(self.backbone_cfg['embed_dim'], patch_size_mask)

    def map_state_dict_from_vitpose(self, state_dict_pth=None):
        if state_dict_pth is None:
            print("No state dict provided")
            return
        state_dict = torch.load(state_dict_pth)['state_dict']
        keys = list(state_dict.keys())
        keys = [k for k in keys if 'keypoint_head' not in k]

        # Load state dict until final_layer
        original_pos_embed = state_dict["backbone.pos_embed"]
        img_h, img_w = self.cfg.model["backbone"]["img_size"]
        # It has a wrong input size, but it doesn't matter, as interpolate_pos_encoding has also the wrong scale size
        patch_size = self.cfg.model["backbone"]["patch_size"]
        new_npatch = self.backbone.pos_embed.shape[1] - 1
        state_dict["backbone.pos_embed"] = self.interpolate_pos_encoding(img_w, img_h, original_pos_embed, patch_size, new_npatch)

        self.resize_patch_embed(state_dict)

        backbone_state_dict = {k: state_dict[k] for k in keys}
        self.load_state_dict(backbone_state_dict, strict=False)

    def interpolate_pos_encoding(self, img_w, img_h, original_pos_embed, patch_size, new_npatch, curr_patch_h=16, curr_patch_w=12):
        npatch = original_pos_embed.shape[1] - 1
        if npatch == new_npatch:# and img_w == img_h:
            return original_pos_embed
        class_pos_embed = original_pos_embed[:, :1]
        patch_pos_embed = original_pos_embed[:, 1:]
        dim = self.backbone.pos_embed.shape[-1]
        w0 = img_w // patch_size
        h0 = img_h // patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        # scale_factor is wrong, but it doesn't matter as the img_w and img_h are inverted anyways

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(curr_patch_h), int(curr_patch_w), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / curr_patch_h, h0 / curr_patch_w),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def resize_patch_embed(self, state_dict):
        # Resize the patch embedding layer
        pretrained_conv_weight = state_dict['backbone.patch_embed.proj.weight']  # Shape: [768, 3, 16, 16]

        if pretrained_conv_weight.shape[2] != self.backbone_cfg["patch_size"]:
            # Interpolate the weights to fit the new patch size
            pretrained_conv_weight_resized = torch.nn.functional.interpolate(pretrained_conv_weight, size=( self.backbone_cfg["patch_size"],  self.backbone_cfg["patch_size"]), mode='bilinear', align_corners=False)

            # Update the state dictionary with the resized weights
            state_dict['backbone.patch_embed.proj.weight'] = pretrained_conv_weight_resized

    def configure_optimizers(self):
        # get params and set different learning rate for the backbone ones
        param_groups = [
            {'params': self.decoder.parameters()},
        ]
        if not self.freeze_backbone:
            param_groups.append({'params': self.backbone.parameters(), 'lr': self.optimizer_cfg['lr']})

        optimizer = eval(f'torch.optim.{self.optimizer_cfg["type"]}')(
            param_groups,
            lr=self.optimizer_cfg['lr'],
            betas=self.optimizer_cfg['betas'],
            weight_decay=self.optimizer_cfg['weight_decay']
        )

        # Learning rate scheduler (MultiStepLR)
        milestones = self.lr_cfg['step']
        gamma = 0.1
        scheduler = MultiStepLR(optimizer, milestones, gamma)

        return [optimizer], [scheduler]

    def forward(self, x, masks):
        # Extract features from ViT. An unfrozen backbone follows the caller's
        # grad mode, so inference under no_grad() does not build a graph.
        with torch.no_grad() if self.freeze_backbone else contextlib.nullcontext():
            features = self.backbone(x)

        if self.use_mask:
            masks_feats = self.mask_downscaling(masks)
            features = features + masks_feats

        if 'MammaNet' in self.decoder_name:
            pred = self.decoder(features, self.backbone.pos_embed, None)
        else:
            pred = self.decoder(features, None)
        return pred

    def training_step(self, batch, batch_idx):
        target = batch
        images = target['image'].to(self.device)
        target_weights = target['target_weight'].to(self.device)
        masks = target['mask'].to(self.device) if self.use_mask else None

        pred = self(images, masks)

        loss = self.criterion(pred, target, target_weights)
        if self.visibility:
            loss['loss_visibility'] = 0.0
            loss['loss_visibility'] = self.vis_criteria(pred['visibility'], target["joints_visibility"])
            loss['loss'] = loss['loss'] + loss['loss_visibility']

        if self.contact:
            loss['loss_contact'] = 0.0
            loss['loss_contact'] = _focal_contact_loss(pred['contact'], target["joints_contact"])
            loss['loss'] = loss['loss'] + loss['loss_contact']
        if self.floor_contact:
            loss['loss_floor_contact'] = 0.0
            loss['loss_floor_contact'] = _focal_contact_loss(pred['floor_contact'], target["joints_floor_contact"])
            loss['loss'] = loss['loss'] + loss['loss_floor_contact']

        steps = 10 if self.cfg.debug_mode else 2000
        train_path = os.path.join(self.viz_dir, 'train')
        os.makedirs(train_path, exist_ok=True)
        if self.global_step > 0 and self.global_step % steps == 0:
        # if self.global_step % 2 == 0:
            with torch.no_grad():
                video_path = compare_results_denseldmks2d(images, pred, target, masks, self.global_step, train_path, self.train_data_cfg["normalize_plus_min_one"])
                self.wandb_video_log(video_path, 'train')
        if self.global_step > 0 and self.global_step % self.cfg.log_steps == 0:
            self.tensorboard_logging(loss, self.global_step, train=True)
        self.log('train/loss', loss["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=images.size(0))
        self.log('train/loss_joints2d', loss["loss_joints2d"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=images.size(0))
        self.log('train/loss_sigma', loss["loss_sigma"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=images.size(0))
        if "loss_visibility" in loss:
            self.log('train/loss_visibility', loss["loss_visibility"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=images.size(0))
        if "loss_contact" in loss:
            self.log('train/loss_contact', loss["loss_contact"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=images.size(0))
        if "loss_floor_contact" in loss:
            self.log('train/loss_floor_contact', loss["loss_floor_contact"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=images.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        target = batch
        images = target['image'].to(self.device)
        target_weights = target['target_weight'].to(self.device)
        masks = target['mask'].to(self.device) if self.use_mask else None

        pred = self(images, masks)

        loss = self.criterion(pred, target, target_weights)
        if self.visibility:
            loss['loss_visibility'] = 0.0
            loss['loss_visibility'] = self.vis_criteria(pred['visibility'], target["joints_visibility"])
            loss['loss'] = loss['loss'] + loss['loss_visibility']

        if self.contact:
            loss['loss_contact'] = 0.0
            loss['loss_contact'] = _focal_contact_loss(pred['contact'], target["joints_contact"])
            loss['loss'] = loss['loss'] + loss['loss_contact']
        if self.floor_contact:
            loss['loss_floor_contact'] = 0.0
            loss['loss_floor_contact'] = _focal_contact_loss(pred['floor_contact'], target["joints_floor_contact"])
            loss['loss'] = loss['loss'] + loss['loss_floor_contact']

        self.tensorboard_logging(loss, self.global_step, train=False,)
        self.log('val/loss', loss["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True, batch_size=images.size(0))  # NOTE: , sync_dist=True MAYBE?
        self.log('val/loss_joints2d', loss["loss_joints2d"], on_step=True, on_epoch=True, prog_bar=True, logger=False, sync_dist=True, batch_size=images.size(0))
        self.log('val/loss_sigma', loss["loss_sigma"], on_step=True, on_epoch=True, prog_bar=True, logger=False, sync_dist=True, batch_size=images.size(0))
        if "loss_visibility" in loss:
            self.log('val/loss_visibility', loss["loss_visibility"], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=images.size(0))
        if "loss_contact" in loss:
            self.log('val/loss_contact', loss["loss_contact"], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=images.size(0))
        if "loss_floor_contact" in loss:
            self.log('val/loss_floor_contact', loss["loss_floor_contact"], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=images.size(0))

        self.show_results = [images, pred, target, masks]
        return loss

    def on_after_backward(self):
        unused = [n for n, p in self.named_parameters() if p.requires_grad and p.grad is None]
        if unused:
            self.print(f"[Unused this step] {unused[:25]}{' …' if len(unused) > 25 else ''}")

    def on_validation_epoch_end(self):
        val_path = os.path.join(self.viz_dir, 'val')
        os.makedirs(val_path, exist_ok=True)
        if self.show_results is not None:
            with torch.no_grad():
                images, pred, target, masks = self.show_results
                video_path = compare_results_denseldmks2d(images, pred, target, masks, self.global_step, val_path, self.val_data_cfg["normalize_plus_min_one"])
                self.wandb_video_log(video_path, 'val')
            self.show_results = None
        self.validation_outputs.clear()

    def wandb_video_log(self, video_path, log_name='train') -> None:
        video = wandb.Video(video_path, format="mp4")
        if isinstance(self.loggers, WandbLogger):
            self.loggers.experiment.log({f"{log_name}/video": video})
        elif isinstance(self.loggers, list):  # in case you're using multiple loggers
            for l in self.loggers:
                if isinstance(l, WandbLogger):
                    l.experiment.log({f"{log_name}/video": video})

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if self.train_data_cfg['type'] == 'BEDLAM_WD':
                # with_epoch tells how many data samples are in the dataset per each gpu and it's multiplied by the number of workers
                # To find num of iterations per epoch, we need to divide
                # we divided by the number of GPUs as well to get the number of iterations per epoch
                self.mamma_train = MixedWebDataset(
                                             self.train_data_cfg,
                                             is_train=True,
                                             ).with_epoch(1_000_000//(self.cfg.workers_per_gpu*self.cfg.gpus_n)).shuffle(2000)
                print("Number of iterations per epoch: ", self.mamma_train.nsamples,
                      np.ceil(1_000_000/(self.cfg.workers_per_gpu*self.cfg.gpus_n*self.cfg.samples_per_gpu)*self.cfg.workers_per_gpu))

                self.mamma_val = MixedWebDataset(
                                             self.val_data_cfg,
                                             is_train=False,)  # change so only augmentation is activated
            else:
                raise NotImplementedError(f"Dataset type {self.train_data_cfg['type']} not implemented yet!")

    def train_dataset(self):
        return self.mamma_train

    def train_dataloader(self):
        self.train_ds = self.train_dataset()
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.cfg.samples_per_gpu,
            num_workers=self.cfg.workers_per_gpu,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataset(self):
        if self.train_data_cfg['type'] == 'BEDLAM_WD':
            return self.mamma_val

        return torch.utils.data.TensorDataset(torch.empty(0))

    def val_dataloader(self):
        self.val_ds = self.val_dataset()
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=16 if self.cfg.samples_per_gpu > 20 else 1,#self.cfg.data['samples_per_gpu'],
            num_workers=2 if self.cfg.samples_per_gpu > 20 else 1,# #self.cfg.data['workers_per_gpu'],
            pin_memory=True
        )

    # Tensoroboard logging should run from first rank only
    @pl.utilities.rank_zero.rank_zero_only
    def tensorboard_logging(self, losses, step_count: int, train: bool = True, write_to_summary_writer: bool = True) -> None:
        """
        Log results to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            step_count (int): Global training step count
            train (bool): Flag indicating whether it is training or validation mode
        """

        mode = 'train' if train else 'val'

        if write_to_summary_writer:
            summary_writer = self.logger.experiment
            for loss_name, val in losses.items():
                summary_writer.add_scalar(mode +'/' + loss_name, val.detach().item(), step_count)