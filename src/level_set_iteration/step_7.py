# src/level_set_iteration/step_7.py

import matplotlib.pyplot as plt

# =========================================================================
# STEP 7: Visualization
# =========================================================================

def visualize_results(preprocessed_img, phi_init, phi_final, segmented_sr, history, config):

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    axes[0].imshow(preprocessed_img, cmap='gray')
    axes[0].contour(phi_init, levels=[0], colors=['cyan'])

    axes[1].imshow(preprocessed_img, cmap='gray')
    axes[1].contour(phi_final, levels=[0], colors=['red'])

    axes[2].imshow(segmented_sr, cmap='gray')

    if history:
        axes[3].plot(history['iteration'], history['l1'])
        axes[3].plot(history['iteration'], history['l2'])

        axes[4].plot(history['iteration'], history['dev_prev'])
        axes[4].plot(history['iteration'], history['dev_curr'])

    plt.tight_layout()
    plt.show()