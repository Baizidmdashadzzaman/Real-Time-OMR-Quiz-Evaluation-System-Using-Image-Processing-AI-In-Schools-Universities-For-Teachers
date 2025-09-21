import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def split_info_section_into_combined_sections_v9(info_section_image_path):
    """
    Splits the info section image into three combined sections:
    1. Student ID (header + input box)
    2. Subject Code (header + input box)
    3. Total Number (header + input area)

    This version uses a simplified and more direct approach to selecting the two main dividers.
    """
    info_image = cv2.imread(info_section_image_path)

    if info_image is None:
        print(f"❌ Error: Could not load image from {info_section_image_path}")
        return

    height, width, _ = info_image.shape
    gray = cv2.cvtColor(info_image, cv2.COLOR_BGR2GRAY)

    # --- Step 1: Create a binary image where white areas are dark (0) ---
    # This makes peaks for pink/text headers and valleys for white input areas.
    # Pixels > 230 are considered "pure white" and become 0 in inverted_mask, others (pink/text) become 255.
    _, pure_white_mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    inverted_mask = cv2.bitwise_not(pure_white_mask)

    # --- Step 2: Analyze Horizontal Projection Profile ---
    horizontal_proj = np.sum(inverted_mask, axis=1, dtype=np.int32)  # Use int32 to avoid overflow warning

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
    # Calculate an adaptive jump threshold based on the observed peaks in the profile.
    avg_proj = np.mean(horizontal_proj)
    # Consider values significantly above average as part of peaks.
    peak_values = horizontal_proj[horizontal_proj > avg_proj * 1.5]

    min_jump_threshold = 40000  # Fallback default
    if len(peak_values) > 0:
        median_peak_value = np.median(peak_values)
        # Use a slightly lower factor if 0.7 was too strict and missed jumps.
        # Let's try 0.6 or 0.5 to be more inclusive of "jumps"
        min_jump_threshold = median_peak_value * 0.6
        print(f"Calculated adaptive min_jump_threshold: {min_jump_threshold:.2f}")
    else:
        print(f"No clear peaks found for adaptive threshold, using fixed: {min_jump_threshold}")

    potential_dividers_y = []
    # Iterate through the profile to find upward transitions
    for y in range(1, height):  # Iterate up to height-1
        # A significant positive difference indicates a jump from a valley to a peak.
        if horizontal_proj[y] - horizontal_proj[y - 1] > min_jump_threshold:
            potential_dividers_y.append(y)

    # Sort and remove duplicates
    potential_dividers_y = sorted(list(set(potential_dividers_y)))

    main_dividers_y = []

    # Strategy: Select the two potential dividers that are closest to the ideal 1/3 and 2/3 marks.
    # This is more direct and less prone to IndexErrors from iterative filtering.

    ideal_div1_pos = height / 3
    ideal_div2_pos = 2 * height / 3

    # If we have enough potential dividers, select the best two
    if len(potential_dividers_y) >= 2:
        # Find the two potential dividers that minimize the sum of absolute differences
        # to the ideal 1/3 and 2/3 positions.

        # Calculate scores for each pair of potential dividers
        best_score = float('inf')
        best_pair = []

        for i in range(len(potential_dividers_y)):
            for j in range(i + 1, len(potential_dividers_y)):
                div1_candidate = potential_dividers_y[i]
                div2_candidate = potential_dividers_y[j]

                # Ensure div1_candidate is before div2_candidate
                if div1_candidate >= div2_candidate:
                    continue

                # Calculate score for this pair (how close they are to ideal positions)
                score1 = abs(div1_candidate - ideal_div1_pos) + abs(div2_candidate - ideal_div2_pos)
                score2 = abs(div1_candidate - ideal_div2_pos) + abs(div2_candidate - ideal_div1_pos)

                # Take the minimum score (as they might be assigned to ideal1 or ideal2 interchangeably)
                current_score = min(score1, score2)

                # Check for minimum separation, to ensure they're distinct sections
                if (div2_candidate - div1_candidate) > (height / 8) and current_score < best_score:
                    best_score = current_score
                    best_pair = sorted([div1_candidate, div2_candidate])

        if best_pair:
            main_dividers_y = best_pair

    # Final check: we need exactly two dividers for three sections
    if len(main_dividers_y) == 2:
        section_boundaries = [0] + main_dividers_y + [height]
        print(f"✅ Found 2 distinct main dividers at Y: {main_dividers_y}")
    else:
        print(
            f"❌ Failed to find 2 distinct main dividers. Found: {len(main_dividers_y)} at {main_dividers_y}. Falling back to refined approximate division.")
        # This fallback is a last resort.
        section_boundaries = [
            0,
            int(height * 0.38),  # Empirically tuned based on previous successful crops
            int(height * 0.72),
            height
        ]

    section_names = ["student_id_section", "subject_code_section", "total_number_section"]
    saved_files = []

    for i in range(len(section_boundaries) - 1):
        if i < len(section_names):
            y_start = section_boundaries[i]
            y_end = section_boundaries[i + 1]

            if y_end > y_start + 20:  # Minimum height check
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
    info_section_file = "info_section.png"  # Example filename
    if os.path.exists(info_section_file):
        split_info_section_into_combined_sections_v9(info_section_file)
    else:
        print(f"Error: {info_section_file} not found. Please ensure your initial script generates it first.")