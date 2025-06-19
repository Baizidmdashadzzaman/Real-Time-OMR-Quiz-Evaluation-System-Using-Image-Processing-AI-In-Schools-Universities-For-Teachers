import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def split_info_section_into_combined_sections_v18(info_section_image_path):
    """
    Splits the info section image into three combined sections:
    1. Student ID (header + input box)
    2. Subject Code (header + input box)
    3. Total Number (header + input area)

    This version aggressively fine-tune adaptive valley/peak and jump detection thresholds,
    and adds a deduplication step for potential dividers.
    """
    info_image = cv2.imread(info_section_image_path)

    if info_image is None:
        print(f"❌ Error: Could not load image from {info_section_image_path}")
        return

    height, width, _ = info_image.shape
    gray = cv2.cvtColor(info_image, cv2.COLOR_BGR2GRAY)

    # --- Step 1: Create a binary image where white areas are dark (0) ---
    _, pure_white_mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    inverted_mask = cv2.bitwise_not(pure_white_mask)

    # --- Step 2: Analyze Horizontal Projection Profile ---
    horizontal_proj = np.sum(inverted_mask, axis=1, dtype=np.int32)

    # --- Debugging Visualization ---
    plt.figure(figsize=(10, 6))
    plt.plot(horizontal_proj, range(height), color='blue')
    plt.ylabel("Y-coordinate (Image Height)")
    plt.xlabel("Sum of Pixel Intensities (Horizontal Projection)")
    plt.title("Horizontal Projection Profile of `info_section.png`")
    plt.gca().invert_yaxis()  # Y-axis goes top-down in images
    plt.grid(True)
    plt.show()

    # --- Step 3: Find Dividers based on Projection Profile ---
    min_proj_val = np.min(horizontal_proj)
    max_proj_val = np.max(horizontal_proj)
    proj_range = max_proj_val - min_proj_val

    VALLEY_RELATIVE_THRESHOLD = 0.15  # Keep this consistent
    # **TUNE:** Lowering this further to be even more lenient for peak start
    PEAK_RELATIVE_THRESHOLD = 0.50  # Point is considered a peak if >= 50% into the proj range from min

    # **TUNE:** Lowering this further to be more sensitive to smaller jumps
    min_jump_threshold = proj_range * 0.28  # Try 28% of the range.

    print(f"Calculated adaptive min_jump_threshold (relative to range): {min_jump_threshold:.2f}")
    print(f"Min Proj: {min_proj_val}, Max Proj: {max_proj_val}, Range: {proj_range}")
    print(f"Valley threshold (abs): {min_proj_val + proj_range * VALLEY_RELATIVE_THRESHOLD:.2f}")
    print(f"Peak threshold (abs): {min_proj_val + proj_range * PEAK_RELATIVE_THRESHOLD:.2f}")

    potential_dividers_y = []

    for y in range(1, height):
        current_val = horizontal_proj[y]
        prev_val = horizontal_proj[y - 1]

        jump_magnitude = current_val - prev_val

        # Check for a significant upward jump AND a transition from valley to peak
        if jump_magnitude > min_jump_threshold:
            is_prev_a_valley = prev_val < (min_proj_val + proj_range * VALLEY_RELATIVE_THRESHOLD)
            is_current_a_peak = current_val > (min_proj_val + proj_range * PEAK_RELATIVE_THRESHOLD)

            if is_prev_a_valley and is_current_a_peak:
                potential_dividers_y.append(y)

    potential_dividers_y = sorted(list(set(potential_dividers_y)))

    # --- NEW: Filter out very close potential dividers ---
    # This helps if noise causes multiple closely-spaced "jumps" to be detected.
    filtered_potential_dividers = []
    if potential_dividers_y:
        filtered_potential_dividers.append(potential_dividers_y[0])
        for i in range(1, len(potential_dividers_y)):
            # If the current divider is sufficiently far from the last added divider
            if (potential_dividers_y[i] - filtered_potential_dividers[-1]) > (
                    height / 20):  # Minimum 1/20th height separation
                filtered_potential_dividers.append(potential_dividers_y[i])
    potential_dividers_y = filtered_potential_dividers

    main_dividers_y = []

    # We still rely on selecting the best two from the filtered list.
    if len(potential_dividers_y) >= 2:
        ideal_div1_pos = height / 3
        ideal_div2_pos = 2 * height / 3

        best_score = float('inf')
        best_pair = []

        for i in range(len(potential_dividers_y)):
            for j in range(i + 1, len(potential_dividers_y)):
                div1_candidate = potential_dividers_y[i]
                div2_candidate = potential_dividers_y[j]

                # Ensure minimum separation
                if (div2_candidate - div1_candidate) < (height / 8):
                    continue

                current_score = abs(div1_candidate - ideal_div1_pos) + abs(div2_candidate - ideal_div2_pos)

                if current_score < best_score:
                    best_score = current_score
                    best_pair = [div1_candidate, div2_candidate]

        if best_pair:
            main_dividers_y = sorted(best_pair)

    if len(main_dividers_y) == 2:
        section_boundaries = [0] + main_dividers_y + [height]
        print(f"✅ Found 2 distinct main dividers at Y: {main_dividers_y}")
    else:
        print(
            f"❌ Failed to find 2 distinct main dividers. Found: {len(main_dividers_y)} at {main_dividers_y}. Falling back to fixed approximate division.")
        section_boundaries = [
            0,
            int(height * 0.38),
            int(height * 0.72),
            height
        ]

    section_names = ["student_id_section", "subject_code_section", "total_number_section"]
    saved_files = []

    for i in range(len(section_boundaries) - 1):
        if i < len(section_names):
            y_start = section_boundaries[i]
            y_end = section_boundaries[i + 1]

            if y_end > y_start + 20:
                section_part = info_image[y_start:y_end, 0:width]
                output_filename = f"{section_names[i]}.png"
                cv2.imwrite(output_filename, section_part)
                saved_files.append(output_filename)
                print(f"✅ Saved: {output_filename} (Y: {y_start}-{y_end})")
            else:
                print(f"Skipping very small section (Y: {y_start}-{y_end})")
        else:
            print(f"Warning: More sections found than expected. Skipping extra section beyond {len(section_names)}.")

    if saved_files:
        print(f"Successfully extracted {len(saved_files)} combined sections from info_section.png.")
    else:
        print("❌ No combined sections found. Adjust parameters.")


# --- Main call ---
if __name__ == "__main__":
    info_section_file = "info_section.png"
    if os.path.exists(info_section_file):
        split_info_section_into_combined_sections_v18(info_section_file)
    else:
        print(f"Error: {info_section_file} not found. Please ensure your initial script generates it first.")
