# sch_cs/step_10.py

# ── Standard library ──────────────────────────────────────────────────────────
from typing import Dict, List

# ── Numeric / image ───────────────────────────────────────────────────────────
import cv2
import numpy as np

# ── Visualisation ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2.10 — Visualise SCH-CS result  [Figure 3]
# ─────────────────────────────────────────────────────────────────────────────

def visualize_schcs(
    pb          : np.ndarray,
    hist_data   : Dict,
    rho_data    : Dict,
    th_data     : Dict,
    label_data  : Dict,
    sr_regions  : List[Dict],
    all_regions : List[Dict],
    image_name  : str = ''
) -> None:
    """
    Four-panel figure matching Figure 3 of the paper.
    (a) p_b  (b) Histogram  (c) All candidates  (d) Final SRs
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(
        f'SCH-CS result — {image_name}\n'
        '(matches Figure 3 of Pramanik et al. 2018)',
        fontsize=12, fontweight='bold'
    )

    th        = th_data['th']
    rho_eff   = rho_data['rho_effective']
    full_hist = hist_data['full_hist'].copy()
    full_hist[0] = 0

    # (a) p_b
    axes[0].imshow(pb, cmap='gray')
    axes[0].set_title('(a) p_b — input to SCH-CS', fontsize=10)
    axes[0].axis('off')

    # (b) Histogram
    axes[1].bar(range(256), full_hist, color='steelblue', alpha=0.6, width=1)
    axes[1].axhline(rho_eff, color='red',    lw=2, ls='-.',
                    label=f'rho_eff={rho_eff:.0f}')
    axes[1].axvline(th,      color='orange', lw=2, ls='--',
                    label=f'th={th:.1f}')
    axes[1].set_title('(b) Histogram of p_b', fontsize=10)
    axes[1].set_xlabel('Gray level')
    axes[1].set_ylabel('Frequency')
    axes[1].legend(fontsize=8)
    axes[1].set_xlim([0, 255])

    # (c) All candidate regions (before CS isolation)
    overlay_c = np.stack([pb, pb, pb], axis=2).copy()
    for reg in all_regions:
        cx  = int(round(reg['centroid'][1]))
        cy  = int(round(reg['centroid'][0]))
        r_px = max(5, int(np.sqrt(reg['size'] / np.pi)))
        cv2.circle(overlay_c, (cx, cy), r_px, (0, 200, 0), 2)
    axes[2].imshow(overlay_c)
    axes[2].set_title(
        f'(c) All {len(all_regions)} candidates\n(after edge filter)',
        fontsize=10
    )
    axes[2].axis('off')

    # (d) Final SRs
    overlay_d = np.stack([pb, pb, pb], axis=2).copy()
    for reg in sr_regions:
        overlay_d[reg['mask'], 0] = 255
        overlay_d[reg['mask'], 1] = 0
        overlay_d[reg['mask'], 2] = 0
        cx  = int(round(reg['centroid'][1]))
        cy  = int(round(reg['centroid'][0]))
        r_px = max(5, int(np.sqrt(reg['size'] / np.pi)))
        cv2.circle(overlay_d, (cx, cy), r_px, (255, 80, 0), 2)
    axes[3].imshow(overlay_d)
    axes[3].set_title(
        f'(d) Final {len(sr_regions)} SR(s) after CS isolation\n(red = SR pixels)',
        fontsize=10
    )
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()

    # Summary table
    print('\n Final SR summary:')
    print(f'  {"Label":<8} {"Size(px)":<10} {"Centroid(row,col)"}')
    print(f'  {"-"*8} {"-"*10} {"-"*20}')
    for reg in sr_regions:
        X, Y = reg['centroid']
        print(f'  {reg["label"]:<8} {reg["size"]:<10} ({X:.1f}, {Y:.1f})')