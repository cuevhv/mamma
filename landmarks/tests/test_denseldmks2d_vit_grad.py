"""Grad-mode behaviour of DenseLdmks2DViT.forward.

An unfrozen backbone must receive gradients during training and must follow
the caller's no_grad() at inference; a frozen backbone never tracks gradients.
Uses tiny stand-in backbone/decoder modules, so no weights or GPU are needed.

Run from the repo root:  python -m pytest landmarks/tests -q
"""
import os
import sys

import pytest
import torch

_LANDMARKS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _LANDMARKS not in sys.path:
    sys.path.insert(0, _LANDMARKS)

vit_mod = pytest.importorskip("lib.models.models_2d.denseldmks2d_vit")


class _Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(4, 3)

    def forward(self, features, mask=None):
        return {"joints2d": self.head(features.mean(dim=(2, 3)))[:, None]}


def _model(freeze_backbone):
    m = vit_mod.DenseLdmks2DViT.__new__(vit_mod.DenseLdmks2DViT)
    torch.nn.Module.__init__(m)
    m.backbone = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2)
    m.decoder = _Decoder()
    m.decoder_name = "TinyDecoder"
    m.use_mask = False
    m.freeze_backbone = freeze_backbone
    return m


def _backbone_output(model, x):
    out = {}
    handle = model.backbone.register_forward_hook(lambda mod, inp, o: out.setdefault("f", o))
    pred = model(x, None)
    handle.remove()
    return pred, out["f"]


def test_unfrozen_backbone_gets_gradients_in_training():
    model = _model(freeze_backbone=False).train()
    pred, _ = _backbone_output(model, torch.randn(2, 3, 8, 8))
    pred["joints2d"].sum().backward()
    assert model.backbone.weight.grad is not None
    assert model.backbone.weight.grad.abs().sum() > 0


def test_unfrozen_backbone_follows_caller_no_grad():
    model = _model(freeze_backbone=False).eval()
    with torch.no_grad():
        pred, feats = _backbone_output(model, torch.randn(2, 3, 8, 8))
    assert not feats.requires_grad and feats.grad_fn is None
    assert not pred["joints2d"].requires_grad


def test_frozen_backbone_never_tracks_gradients():
    model = _model(freeze_backbone=True).train()
    pred, feats = _backbone_output(model, torch.randn(2, 3, 8, 8))
    assert not feats.requires_grad
    pred["joints2d"].sum().backward()
    assert model.backbone.weight.grad is None
    assert model.decoder.head.weight.grad is not None
