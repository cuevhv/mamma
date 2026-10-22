import numpy as np
import matplotlib.pyplot as plt


def show_mask_cv2(mask, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask after squeezing, got shape {mask.shape}")
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return (mask_image * 255).astype(np.uint8)


def show_box(box, ax, color='green', linewidth=2, label=None):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=linewidth))
    if label:
        ax.text(
            x0,
            max(0, y0 - 4),
            label,
            color='white',
            fontsize=10,
            fontweight='bold',
            bbox=dict(facecolor=color, edgecolor='none', alpha=0.75, pad=2),
        )
