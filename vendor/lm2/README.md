# LM2 archival detector (vendored slice)

- `component_detector/` — YOLOv5 inference (GPLv3, viz [`NOTICE.md`](../../NOTICE.md))
- `weights/best.pt` — archivní detektor štítků (LM2 release v-2-3, `acd/best.pt`), součást Docker image

Výchozí cesta v kódu: `vendor/lm2/weights/best.pt` (`Settings.default_lm2_weights()`).  
Override: env `LM2_WEIGHTS_PATH`.

`weights/*.pt` jsou v **Git LFS** — po klonu: `git lfs pull`.
