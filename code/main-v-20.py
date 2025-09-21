import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont  # Ensure PIL is imported


# --- Section 1: Image Preprocessing and Splitting ---

def display_and_split_biggest_rectangle_part(image_path):
    """
    Loads an image, finds the biggest rectangular part (assumed to be the OMR sheet),
    and splits it into the answer section (left) and info section (right),
    returning them as NumPy arrays and the bounding box of the main OMR part.

    Args:
        image_path (str): The path to the input image file.

    Returns:
        tuple: (answer_section_img, info_section_img, biggest_rect_bbox)
               where biggest_rect_bbox is (x, y, w, h) of the detected OMR sheet,
               or (None, None, None) if the image cannot be loaded or no rectangle is found.
    """
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return None, None, None  # Return None for all on failure

    # Convert to grayscale for contour detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary inverse threshold to make text/marks white and background black
    # This helps in finding contours of the main OMR area. Adjust threshold if needed.
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest_rect_bbox = None
    max_area = 0

    # Iterate through contours to find the largest quadrilateral (rectangle)
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        # Check if the approximated contour has 4 vertices (a quadrilateral)
        # and if its area is significant enough to be considered the OMR sheet.
        # Area threshold (1000) might need adjustment based on image resolution.
        if len(approx) == 4 and cv2.contourArea(approx) > 1000:
            x, y, w, h = cv2.boundingRect(approx)  # Get bounding rectangle coordinates
            area = w * h
            if area > max_area:  # Keep track of the biggest rectangle found
                max_area = area
                biggest_rect_bbox = (x, y, w, h)

    if biggest_rect_bbox:
        x, y, w, h = biggest_rect_bbox
        # Crop the biggest rectangular part from the original image
        rectangular_part = image[y:y + h, x:x + w]

        # Split into initial answer section (left 75% width) and info section (right 25% width)
        # These percentages are assumptions based on typical OMR layouts and might need tuning.
        answer_section_with_outer = rectangular_part[:, 0:int(0.75 * w)]
        info_section_with_outer = rectangular_part[:, int(0.75 * w):]

        # Function to find the main content within a section (removing surrounding white borders)
        def crop_white_borders(section_image):
            if section_image is None or section_image.size == 0:
                return section_image, (0, 0, 0, 0)  # Return empty if input is empty
            gray_section = cv2.cvtColor(section_image, cv2.COLOR_BGR2GRAY)
            # Re-threshold the section to find its internal content
            _, thresh_section = cv2.threshold(gray_section, 200, 255, cv2.THRESH_BINARY_INV)
            contours_section, _ = cv2.findContours(thresh_section, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours_section:
                # Find the largest contour within the section and crop to its bounding box
                largest_contour_section = max(contours_section, key=cv2.contourArea)
                x_sec, y_sec, w_sec, h_sec = cv2.boundingRect(largest_contour_section)
                cropped_img = section_image[y_sec:y_sec + h_sec, x_sec:x_sec + w_sec]
                return cropped_img, (x_sec, y_sec, w_sec, h_sec)
            return section_image, (0, 0, section_image.shape[1],
                                   section_image.shape[0])  # Return original if no significant contours found

        # Apply border cropping to both sections
        answer_section_without_outer, _ = crop_white_borders(answer_section_with_outer)
        info_section_without_outer, info_section_inner_bbox = crop_white_borders(
            info_section_with_outer)  # Get inner bbox too

        print("✅ Main OMR sheet sections extracted (in memory).")
        return answer_section_without_outer, info_section_without_outer, biggest_rect_bbox
    else:
        print("❌ No significant rectangular OMR sheet found. Try adjusting the threshold or contour area.")
        return None, None, None  # Return None for all on failure


def extract_info_sections(info_section_image):
    """
    Extracts the 'শিক্ষার্থী আইডি', 'বিষয় কোড', and 'মোট নম্বর' sections
    from the info_section_image (NumPy array) based on assumed vertical percentages.
    Returns them as NumPy arrays and their relative y-coordinates within the info_section_image.

    Note: This function is primarily for demonstrating the info section split.
    Further processing (OCR) would be needed to read the actual text/numbers.

    Args:
        info_section_image (np.array): The NumPy array of the info section.

    Returns:
        tuple: (student_id_part, subject_code_part, total_marks_part, total_marks_rel_bbox_y)
               where total_marks_rel_bbox_y is (start_y, end_y) relative to info_section_image,
               or (None, None, None, (0,0)) if the input is invalid.
    """
    if info_section_image is None or info_section_image.size == 0:
        print("❌ Error: Info section image is empty or None.")
        return None, None, None, (0, 0)

    img_height, img_width, _ = info_section_image.shape

    # --- ADJUST THESE PERCENTAGES based on your OMR sheet layout ---
    student_id_section_height_percentage = 1 / 3
    subject_code_section_height_percentage = 1 / 3
    # The remaining height will be for total_marks_section_height_percentage

    student_id_start_y = 0
    student_id_end_y = int(img_height * student_id_section_height_percentage)

    subject_code_start_y = student_id_end_y
    subject_code_end_y = subject_code_start_y + int(img_height * subject_code_section_height_percentage)

    total_marks_start_y = subject_code_end_y
    total_marks_end_y = img_height

    student_id_part = None
    subject_code_part = None
    total_marks_part = None
    total_marks_rel_bbox_y = (0, 0)  # Store relative Y coordinates for total marks section

    # Perform cropping, ensuring valid dimensions
    if student_id_end_y > student_id_start_y:
        student_id_part = info_section_image[student_id_start_y:student_id_end_y, 0:img_width]
        print("✅ Student ID section extracted (in memory).")
    else:
        print("❌ Student ID section crop failed.")

    if subject_code_end_y > subject_code_start_y:
        subject_code_part = info_section_image[subject_code_start_y:subject_code_end_y, 0:img_width]
        print("✅ Subject Code section extracted (in memory).")
    else:
        print("❌ Subject Code section crop failed.")

    if total_marks_end_y > total_marks_start_y:
        total_marks_part = info_section_image[total_marks_start_y:total_marks_end_y, 0:img_width]
        total_marks_rel_bbox_y = (total_marks_start_y, total_marks_end_y)
        print("✅ Total Marks section extracted (in memory).")
    else:
        print("❌ Total Marks section crop failed.")

    return student_id_part, subject_code_part, total_marks_part, total_marks_rel_bbox_y


def split_answer_section_horizontally(answer_section_image, num_columns=15):
    """
    Splits the answer_section_image (NumPy array) horizontally into two equal halves,
    and then each half into a specified number of columns.
    Returns all these parts as two lists of NumPy arrays (top half columns, bottom half columns).

    Args:
        answer_section_image (np.array): The NumPy array of the answer section.
        num_columns (int): The number of columns to split each half into.

    Returns:
        tuple: (top_half_cols, bottom_half_cols) as lists of NumPy arrays,
               or ([], []) if the input is invalid.
    """
    if answer_section_image is None or answer_section_image.size == 0:
        print(f"❌ Error: Answer section image is empty or None.")
        return [], []  # Return empty lists on failure

    img_height, img_width, _ = answer_section_image.shape

    middle_y = img_height // 2

    # Split into top and bottom halves
    top_half = answer_section_image[0:middle_y, 0:img_width]
    bottom_half = answer_section_image[middle_y:img_height, 0:img_width]

    print("✅ Answer section split into two halves (in memory).")

    # Split top_half into 'num_columns' columns
    # Using np.array_split to handle cases where width is not perfectly divisible by num_columns
    top_half_cols = np.array_split(top_half, num_columns, axis=1)
    print(f"✅ Top half split into {len(top_half_cols)} columns.")

    # Split bottom_half into 'num_columns' columns
    bottom_half_cols = np.array_split(bottom_half, num_columns, axis=1)
    print(f"✅ Bottom half split into {len(bottom_half_cols)} columns.")

    return top_half_cols, bottom_half_cols


# --- Section 2: Circle Detection and Option Numbering (Revised and Fixed for Debugging) ---

def detect_filled_circles_in_segment(image_np_array, debug_mode=True, column_key="Unknown Column"):
    """
    Detects which option (1, 2, 3, or 4) is selected in an image segment (column).
    This function is adapted from the provided detect_selected_option, using
    predefined ROIs and intensity analysis for robustness against faint marks.

    Args:
        image_np_array (np.array): A NumPy array representing a single column segment.
        debug_mode (bool): If True, will display ROIs and print detailed debugging info.
        column_key (str): Identifier for the column being processed (e.g., "Top_Col_1").

    Returns:
        tuple: (selected_option_numbers, all_option_details)
               where selected_option_numbers is a list of integers (e.g., [1])
               and all_option_details is a list of dicts. Each dict contains:
               'option_num', 'center', 'radius', 'is_filled'.
    """
    if image_np_array is None or image_np_array.size == 0:
        if debug_mode:
            print(f"Debug ({column_key}): Warning: Empty image_np_array provided.")
        return [], []

    try:
        # Convert NumPy array (BGR) to PIL Image (L for grayscale)
        # Create a copy for drawing if in debug mode
        if debug_mode:
            img_to_draw = image_np_array.copy()
            gray_for_pil = cv2.cvtColor(image_np_array, cv2.COLOR_BGR2GRAY)
        else:
            gray_for_pil = cv2.cvtColor(image_np_array, cv2.COLOR_BGR2GRAY)

        # --- Optional: Adaptive Thresholding for Robustness ---
        # Uncomment and experiment with these parameters if simple grayscale isn't enough
        # blockSize: Should be an odd number (e.g., 11, 15, 21).
        # C: Constant subtracted from the mean or weighted mean. Smaller values mean more pixels are black.
        # if adaptive_thresholding:
        #    thresh_img_np = cv2.adaptiveThreshold(gray_for_pil, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                          cv2.THRESH_BINARY_INV, 15, 5) # Example parameters
        #    img_pil = Image.fromarray(thresh_img_np)
        # else:
        img_pil = Image.fromarray(gray_for_pil)  # Default: use simple grayscale

    except Exception as e:
        print(f"Debug ({column_key}): Error converting NumPy array to PIL Image in segment detection: {e}")
        return [], []

    width, height = img_pil.size

    # --- Define Regions of Interest (ROIs) for each option within the segment ---
    # These parameters are critical and might need fine-tuning for your specific OMR.
    # They are relative to the *current segment image* (col_img).

    # Horizontal bounds: Take a central slice of the segment's width.
    # This factor determines how wide the ROI is as a percentage of the segment's width.
    roi_width_factor = 0.5  # Example: 50% of the segment width
    x_center = width // 2
    roi_width = int(width * roi_width_factor)
    x_left = max(0, x_center - roi_width // 2)
    x_right = min(width, x_center + roi_width // 2)

    # Vertical bounds: Divide the segment height into 4 sections for 4 options.
    # `vertical_padding_factor` shrinks the ROI to focus on the bubble,
    # avoiding surrounding text or lines.
    option_section_height = height // 4
    vertical_padding_factor = 0.1  # 10% padding from top/bottom of each option's section

    rois = []  # List to store (left, upper, right, lower) tuples for each ROI
    for i in range(4):  # For options 1 through 4
        y_upper = i * option_section_height + int(option_section_height * vertical_padding_factor)
        y_lower = (i + 1) * option_section_height - int(option_section_height * vertical_padding_factor)
        # Ensure coordinates are within image bounds
        y_upper = max(0, y_upper)
        y_lower = min(height, y_lower)
        rois.append((x_left, y_upper, x_right, y_lower))

    avg_pixel_values = []
    all_option_details = []

    for i, roi in enumerate(rois):
        option_num = i + 1  # 1-indexed option number

        # Calculate approximate center and radius for drawing purposes
        # These are relative to the current segment (col_img)
        approx_center = ((roi[0] + roi[2]) // 2, (roi[1] + roi[3]) // 2)
        approx_radius = int(min((roi[2] - roi[0]), (roi[3] - roi[1])) / 2)

        current_option_detail = {
            'option_num': option_num,
            'center': approx_center,
            'radius': approx_radius,
            'is_filled': False,  # Will be determined later
            'mean_intensity': float('inf')
        }
        # Append this detail now, will update 'mean_intensity' and 'is_filled' later
        all_option_details.append(current_option_detail)

        # Check for invalid ROI dimensions BEFORE attempting to crop
        if roi[2] <= roi[0] or roi[3] <= roi[1]:
            if debug_mode:
                print(
                    f"Debug ({column_key}): Warning: ROI for option {option_num} has invalid dimensions: {roi}. Skipping.")
            avg_pixel_values.append(float('inf'))
            continue  # Skip to next iteration

        try:
            cropped_img_pil = img_pil.crop(roi)
            # Now check the size of the *actual cropped image*
            if cropped_img_pil.size[0] == 0 or cropped_img_pil.size[1] == 0:
                if debug_mode:
                    print(
                        f"Debug ({column_key}): Warning: Cropped image for option {option_num} is empty (0 size). Skipping.")
                avg_pixel_values.append(float('inf'))
                continue  # Skip to next iteration

            pixel_data = list(cropped_img_pil.getdata())

            if not pixel_data:  # Handle case where cropped image might somehow be empty after getdata()
                avg_pixel = float('inf')
            else:
                avg_pixel = sum(pixel_data) / len(pixel_data)

        except Exception as e:
            if debug_mode:
                print(
                    f"Debug ({column_key}): Error during cropping or pixel data extraction for option {option_num} (ROI: {roi}): {e}. Marking as invalid.")
            avg_pixel = float('inf')  # Mark as invalid if any error during crop

        avg_pixel_values.append(avg_pixel)
        # Update mean intensity for the detail that was already appended
        all_option_details[i]['mean_intensity'] = avg_pixel

        if debug_mode:
            # Draw the ROI rectangle on the debug image
            cv2.rectangle(img_to_draw, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 255), 1)  # Yellow rectangle
            # Put text for average pixel value
            cv2.putText(img_to_draw, f"{avg_pixel:.0f}", (roi[0], roi[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)  # Black text

    if debug_mode:
        print(f"Debug ({column_key}): Raw average pixel values for options: {avg_pixel_values}")
        # Display the column with ROIs drawn
        window_name = f"Debug: {column_key} with ROIs"
        cv2.imshow(window_name, img_to_draw)
        cv2.waitKey(0)  # Wait for a key press to close and proceed

        # Check if the window exists before trying to destroy it
        # This is a bit of a workaround; in some OpenCV versions/OS,
        # cv2.getWindowProperty might return -1 if the window was just closed manually.
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow(window_name)

    if not avg_pixel_values or all(val == float('inf') for val in avg_pixel_values):
        if debug_mode:
            print(f"Debug ({column_key}): No valid ROIs found or all ROIs are empty/invalid in this segment.")
        return [], []

    # Find the option with the minimum average pixel value (darkest region).
    min_avg_value = float('inf')
    selected_option_index = -1

    # Ensure to only consider valid (non-inf) pixel values for selection
    valid_avg_pixel_values = [val for val in avg_pixel_values if val != float('inf')]

    if not valid_avg_pixel_values:
        if debug_mode:
            print(f"Debug ({column_key}): No valid options to select from after filtering infinities.")
        return [], []  # No valid options to select from

    min_avg_value = min(valid_avg_pixel_values)
    # Find the index of this minimum value in the original avg_pixel_values list
    for i, val in enumerate(avg_pixel_values):
        if val == min_avg_value:
            selected_option_index = i
            break

    # If no distinct option found (e.g., all were infinity or no unique min), return empty
    if selected_option_index == -1:
        if debug_mode:
            print(f"Debug ({column_key}): Could not determine a distinct minimum average pixel value.")
        return [], []

    # --- Heuristic for determining distinct selection ---
    sorted_values = sorted(valid_avg_pixel_values)

    selected_numbers = []

    # Threshold 1: The darkest option's average value should be significantly
    # lower than the second darkest option.
    threshold_difference_ratio = 0.35  # Selected option must be 35% darker than the next one

    # Threshold 2: Fallback - if the ratio isn't met, check if there's a large
    # absolute difference between the darkest and the brightest option.
    min_max_diff_threshold = 40  # An absolute difference of 40 in grayscale values (out of 255)

    is_distinctly_selected = False

    if len(sorted_values) >= 2 and min_avg_value == sorted_values[0]:  # Ensure min_avg_value is indeed the darkest
        if sorted_values[1] > 0:  # Avoid division by zero
            if (sorted_values[1] - min_avg_value) / sorted_values[1] > threshold_difference_ratio:
                is_distinctly_selected = True
        else:  # If sorted_values[1] is 0, then min_avg_value must also be 0 for it to be the darkest
            if min_avg_value == 0:
                is_distinctly_selected = True

    # If not distinctly selected by ratio, try the absolute difference fallback
    # Also handles cases where there might be fewer than 4 valid options, as long as there's more than one.
    if not is_distinctly_selected and len(sorted_values) > 1:
        max_avg_value_overall = sorted_values[-1]  # The largest (lightest) valid average
        if (max_avg_value_overall - min_avg_value) > min_max_diff_threshold:
            is_distinctly_selected = True

    # If there's only one valid option in the segment, and it's the darkest (by definition),
    # assume it's selected. This handles cases where a column might only contain one bubble due to cropping.
    if len(valid_avg_pixel_values) == 1:
        is_distinctly_selected = True

    if is_distinctly_selected:
        selected_numbers.append(selected_option_index + 1)
        # Update the is_filled status in all_option_details for the selected option
        for detail in all_option_details:
            if detail['option_num'] == (selected_option_index + 1):
                detail['is_filled'] = True
                if debug_mode:
                    print(
                        f"Debug ({column_key}): Option {detail['option_num']} marked as FILLED (Mean Intensity: {detail['mean_intensity']:.2f})")
                break
    else:
        if debug_mode:
            print(f"Debug ({column_key}): No distinct option detected as filled based on heuristics.")

    return selected_numbers, all_option_details


# --- Section 3: Main Execution ---

if __name__ == "__main__":
    # Specify the path to your OMR image
    original_omr_image_path = "omr4.png"  # Changed to original image for fresh processing

    print("--- Starting OMR Sheet Processing ---")

    # Define the correct answers for each question/column.
    # The keys should match the column keys generated by the script (e.g., 'Top_Col_1', 'Bottom_Col_1').
    # The values should be a list containing the correct option number(s) for that question.
    # IMPORTANT: REPLACE THESE WITH YOUR ACTUAL CORRECT ANSWERS.
    correct_answers = {
        'Top_Col_1': [2],
        'Top_Col_2': [2],
        'Top_Col_3': [3],
        'Top_Col_4': [4],
        'Top_Col_5': [3],
        'Top_Col_6': [2],
        'Top_Col_7': [2],
        'Top_Col_8': [3],
        'Top_Col_9': [3],
        'Top_Col_10': [4],
        'Top_Col_11': [3],
        'Top_Col_12': [2],
        'Top_Col_13': [2],
        'Top_Col_14': [2],
        'Top_Col_15': [2],
        'Bottom_Col_1': [1],
        'Bottom_Col_2': [2],
        'Bottom_Col_3': [3],
        'Bottom_Col_4': [3],
        'Bottom_Col_5': [1],
        'Bottom_Col_6': [1],
        'Bottom_Col_7': [1],
        'Bottom_Col_8': [1],  # This is the target for debugging
        'Bottom_Col_9': [2],
        'Bottom_Col_10': [2],
        'Bottom_Col_11': [2],
        'Bottom_Col_12': [1],
        'Bottom_Col_13': [2],
        'Bottom_Col_14': [3],
        'Bottom_Col_15': [4]
    }

    # Initialize total marks
    total_marks_gained = 0

    # Step 1: Extract the main answer and info sections from the OMR sheet,
    # and get the bounding box of the overall OMR part.
    answer_section_img, info_section_img, omr_bbox = display_and_split_biggest_rectangle_part(original_omr_image_path)

    # Dictionary to store all detected selected options
    all_selected_options_by_column = {}

    # Load the original OMR image for drawing annotations
    omr_image_for_drawing = None
    try:
        omr_image_for_drawing = cv2.imread(original_omr_image_path)
        if omr_image_for_drawing is None:
            raise FileNotFoundError(f"Image not found at {original_omr_image_path} for drawing.")
    except Exception as e:
        print(f"❌ Error loading image for drawing: {e}")

    if answer_section_img is not None and answer_section_img.size > 0 and omr_image_for_drawing is not None:
        print("\n--- Splitting Answer Section into Halves and then Columns ---")
        answer_top_half_cols, answer_bottom_half_cols = split_answer_section_horizontally(answer_section_img,
                                                                                          num_columns=15)

        # Calculate absolute offsets for the answer section
        answer_section_abs_x = omr_bbox[0]
        answer_section_abs_y = omr_bbox[1]
        answer_section_width = int(0.75 * omr_bbox[2])
        answer_section_height = omr_bbox[3]
        num_columns = 15  # Consistent with split_answer_section_horizontally
        col_width_actual = answer_section_width / num_columns

        # Define reduced circle size for drawing
        drawing_radius_scale = 0.6
        drawing_thickness = 3  # Line thickness for circles

        # Process top half columns
        for i, col_img in enumerate(answer_top_half_cols):
            column_key = f"Top_Col_{i + 1}"
            selected_numbers, all_options_in_column = detect_filled_circles_in_segment(col_img, debug_mode=True,
                                                                                       column_key=column_key)  # DEBUG MODE ON
            all_selected_options_by_column[column_key] = selected_numbers
            print(f"  {column_key}: Student selected = {selected_numbers}")

            col_abs_x_offset = answer_section_abs_x + int(i * col_width_actual)
            col_abs_y_offset = answer_section_abs_y  # Top half starts at omr_y

            # Drawing logic for highlighting
            if column_key in correct_answers:
                correct_ans_for_col = correct_answers[column_key]

                # Check if the student's overall selection for this question is incorrect
                is_question_incorrect_overall = not (selected_numbers == correct_ans_for_col)

                for option_detail in all_options_in_column:
                    option_num = option_detail['option_num']
                    center_rel = option_detail['center']
                    radius = int(option_detail['radius'] * drawing_radius_scale)  # Apply scale to radius
                    is_filled = option_detail['is_filled']  # Use the is_filled status from new logic

                    center_abs = (center_rel[0] + col_abs_x_offset, center_rel[1] + col_abs_y_offset)

                    if option_num in correct_ans_for_col:
                        if is_filled:
                            # Student correctly marked this option
                            cv2.circle(omr_image_for_drawing, center_abs, radius, (0, 255, 0),
                                       drawing_thickness)  # Green: Correctly selected correct answer
                        elif is_question_incorrect_overall:  # Only highlight correct answer in blue if student got it wrong
                            # Student missed this correct option, and the question overall is wrong - highlight correct in BLUE
                            cv2.circle(omr_image_for_drawing, center_abs, radius, (255, 0, 0),
                                       drawing_thickness)  # Blue: Missed correct answer
                    elif is_filled:  # Student marked an option that is not correct - highlight in RED
                        cv2.circle(omr_image_for_drawing, center_abs, radius, (0, 0, 255),
                                   drawing_thickness)  # Red: Incorrectly selected option

                if selected_numbers == correct_ans_for_col:
                    total_marks_gained += 1
                    print(f"    ⭐ Correct! (+1 mark)")
                else:
                    print(f"    ❌ Incorrect. Correct answer: {correct_ans_for_col}")
            else:
                print(f"    ⚠️ Warning: No correct answer defined for {column_key}")

        # Process bottom half columns
        for i, col_img in enumerate(answer_bottom_half_cols):
            column_key = f"Bottom_Col_{i + 1}"
            selected_numbers, all_options_in_column = detect_filled_circles_in_segment(col_img, debug_mode=True,
                                                                                       column_key=column_key)  # DEBUG MODE ON
            all_selected_options_by_column[column_key] = selected_numbers
            print(f"  {column_key}: Student selected = {selected_numbers}")

            col_abs_x_offset = answer_section_abs_x + int(i * col_width_actual)
            col_abs_y_offset = answer_section_abs_y + (answer_section_height // 2)  # Bottom half offset

            # Drawing logic for highlighting

            if column_key in correct_answers:
                correct_ans_for_col = correct_answers[column_key]

                # Check if the student's overall selection for this question is incorrect
                is_question_incorrect_overall = not (selected_numbers == correct_ans_for_col)

                for option_detail in all_options_in_column:
                    option_num = option_detail['option_num']
                    center_rel = option_detail['center']
                    radius = int(option_detail['radius'] * drawing_radius_scale)  # Apply scale to radius
                    is_filled = option_detail['is_filled']  # Use the is_filled status from new logic

                    center_abs = (center_rel[0] + col_abs_x_offset, center_rel[1] + col_abs_y_offset)

                    if option_num in correct_ans_for_col:
                        if is_filled:
                            # Student correctly marked this option
                            cv2.circle(omr_image_for_drawing, center_abs, radius, (0, 255, 0),
                                       drawing_thickness)  # Green: Correctly selected correct answer
                        elif is_question_incorrect_overall:  # Only highlight correct answer in blue if student got it wrong
                            # Student missed this correct option, and the question overall is wrong - highlight correct in BLUE
                            cv2.circle(omr_image_for_drawing, center_abs, radius, (255, 0, 0),
                                       drawing_thickness)  # Blue: Missed correct answer
                    elif is_filled:  # Student marked an option that is not correct - highlight in RED
                        cv2.circle(omr_image_for_drawing, center_abs, radius, (0, 0, 255),
                                   drawing_thickness)  # Red: Incorrectly selected option

                if selected_numbers == correct_ans_for_col:
                    total_marks_gained += 1
                    print(f"    ⭐ Correct! (+1 mark)")
                else:
                    print(f"    ❌ Incorrect. Correct answer: {correct_ans_for_col}")
            else:
                print(f"    ⚠️ Warning: No correct answer defined for {column_key}")

    else:
        print("\nAnswer section not available for splitting and processing or image for drawing not loaded.")

    # --- Process the info section and write total marks ---
    if info_section_img is not None and info_section_img.size > 0 and omr_image_for_drawing is not None:
        print("\n--- Extracting Info Sub-sections and Writing Total Marks ---")
        student_id_img, subject_code_img, total_marks_img, total_marks_rel_bbox_y = extract_info_sections(
            info_section_img)

        try:
            # Define font parameters for OpenCV (cv2.putText)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5  # Adjust font scale as needed
            font_thickness = 4  # Adjust thickness for visibility
            font_color = (0, 0, 0)  # Black color (BGR)

            if omr_bbox:
                omr_x, omr_y, omr_w, omr_h = omr_bbox
                # Calculate the absolute coordinates for the info section within the original image
                info_section_abs_x_start = omr_x + int(0.75 * omr_w)
                info_section_abs_y_start = omr_y

                # Calculate the absolute coordinates for the total marks section.
                total_marks_abs_y_start = info_section_abs_y_start + total_marks_rel_bbox_y[0]
                total_marks_abs_y_end = info_section_abs_y_start + total_marks_rel_bbox_y[1]

                # Calculate the center point for placing the text
                text_center_x = info_section_abs_x_start + (info_section_img.shape[1] // 2)
                text_center_y = total_marks_abs_y_start + ((total_marks_abs_y_end - total_marks_abs_y_start) // 2)

                # Get text size to properly center it
                text_string = str(total_marks_gained)
                text_size = cv2.getTextSize(text_string, font, font_scale, font_thickness)[0]

                # Calculate origin to truly center the text within the bounding box
                text_origin_x = int(text_center_x - text_size[0] / 2)
                text_origin_y = int(text_center_y + text_size[1] / 2)  # Adjust for baseline

                # Draw the total marks onto the original image
                cv2.putText(omr_image_for_drawing, text_string, (text_origin_x, text_origin_y),
                            font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                print(f"✅ Total marks '{total_marks_gained}' written onto the 'মোট নম্বর' section.")

            # Save the modified image
            output_image_path = "omr4_with_marks.png"
            cv2.imwrite(output_image_path, omr_image_for_drawing)
            print(f"✅ Final OMR image with marks saved to {output_image_path}")

            # --- Display the image in a popup window ---
            cv2.imshow("OMR Sheet with Total Marks", omr_image_for_drawing)
            print("\n💡 Displaying OMR image with marks in a popup. Close the window to continue.")
            cv2.waitKey(0)  # Wait indefinitely until a key is pressed (window closed)
            cv2.destroyAllWindows()  # Close all OpenCV windows

        except Exception as e:
            print(f"❌ An error occurred while drawing marks on the image or displaying it: {e}")

    print("\n--- OMR Processing Complete ---")
    print("\nSummary of all selected options per answer column:")
    for col_name, selected_opts in all_selected_options_by_column.items():
        print(f"{col_name}: {selected_opts}")

    print(f"\nTotal marks gained by the student: {total_marks_gained}")