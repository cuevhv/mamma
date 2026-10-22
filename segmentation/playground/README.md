# playground

This directory is **not** part of the main pipeline. It contains throwaway
scripts used for:

- verifying that SAM2 / SAM3 installed correctly in your environment,
- isolating issues when the main pipeline misbehaves (is it SAM, or is it us?),
- small reference demos against the upstream SAM repos.

For actual usage of the segmentation pipeline, see the top-level `README.md`
and the entry point `run_ma_masks.py`.

Scripts here are not kept in sync with pipeline refactors and may lag behind.
