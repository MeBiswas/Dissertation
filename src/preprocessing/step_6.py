# preprocessing/step_6.py

# ── Numeric / image ───────────────────────────────────────────────────────────
import cv2
import numpy as np

# ── Visualisation ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1.6 — Visualise  [Figure 2]
# ─────────────────────────────────────────────────────────────────────────────

def visualize_preprocessing(
    original_color : np.ndarray,
    without_scale  : np.ndarray,
    grayscale      : np.ndarray,
    bg_removed     : np.ndarray,
    pb             : np.ndarray,
    image_name     : str = ''
) -> None:
    """
    Six-panel figure matching Figure 2 of the paper.
    (a) Original  (b) Without scale  (c) Grayscale  (d) BG removed
    (e) p_b  (f) Histogram of p_b
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f'Pre-processing pipeline — {image_name}\n'
        '(matches Figure 2 of Pramanik et al. 2018)',
        fontsize=13, fontweight='bold'
    )

    panels = [
        (axes[0,0], cv2.cvtColor(original_color, cv2.COLOR_BGR2RGB),
         '(a) Original pseudo-colour TBI',        False),
        (axes[0,1], cv2.cvtColor(without_scale,  cv2.COLOR_BGR2RGB),
         '(b) Overlay stripped + no colour bar',  False),
        (axes[0,2], grayscale,
         '(c) Blue channel — hot = bright',       True),
        (axes[1,0], bg_removed,
         '(d) Background removed',                True),
        (axes[1,1], pb,
         '(e) p_b — after reconstruction\n(input to SCH-CS)', True),
    ]
    for ax, img, title, is_gray in panels:
        ax.imshow(img, cmap='gray' if is_gray else None)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    ax_h = axes[1, 2]
    nz   = pb[pb > 0].ravel()
    ax_h.hist(nz, bins=128, range=(1, 256), color='steelblue', alpha=0.8)
    ax_h.axvline(nz.mean(), color='red', ls='--',
                 label=f'mean = {nz.mean():.1f}')
    ax_h.set_title('(f) Histogram of p_b\n(non-zero pixels only)', fontsize=10)
    ax_h.set_xlabel('Pixel intensity')
    ax_h.set_ylabel('Frequency')
    ax_h.legend(fontsize=8)

    plt.tight_layout()
    plt.show()