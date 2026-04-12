# src/features_extraction/step_1.py

def extract_sr_region(segmented_sr, original_gray):
    sr_region = original_gray * segmented_sr

    if sr_region.max() == 0 or segmented_sr.sum() == 0:
        print("[Warn] SR empty → returning None")
        return None

    return sr_region