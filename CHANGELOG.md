# Changelog

## v1.1.0

### Optimization (ma_3d)

- **Occlusion-aware gating (new, opt-in).** Fixes implausible fits when a body part is hidden in most views (e.g. legs behind a table coming out with impossible rotations). Before, occluded keypoints were held at a minimum weight, so the many views that *couldn't* see a part out-voted the one that could; gating instead drives occluded detections toward zero so the view that actually sees the part wins. Enable with `--occlusion-aware-weights` or an `occlusion_gating` config block — **off by default, existing runs are unchanged.**
- **VPoser pose prior (new addition).** A gentle regulariser (`vposer_recon_loss`) that keeps joints with little image evidence on realistic poses by penalizing how far they sit from VPoser's learned pose manifold. Requires the VPoser `V02_05` weights.
- **Recommended config for occluded captures.** `config_real_gmf_small_vals_detr_exp_no_vtemplate_occlusion_vposer.yaml` bundles both of the above at validated settings — the go-to setup for sequences with heavily occluded subjects.

### Visualization (ma_vis)

- **Generic lens undistortion.** When overlaying the 3D fit on the original footage, `--undistort` now correctly undistorts `radtan` / `opencv_brown` lens models (previously a silent no-op for these), so the render lines up with the image.

