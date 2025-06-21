import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

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
        return None, None, None # Return None for all on failure

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
            x, y, w, h = cv2.boundingRect(approx) # Get bounding rectangle coordinates
            area = w * h
            if area > max_area: # Keep track of the biggest rectangle found
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
                return section_image, (0, 0, 0, 0) # Return empty if input is empty
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
            return section_image, (0, 0, section_image.shape[1], section_image.shape[0]) # Return original if no significant contours found

        # Apply border cropping to both sections
        answer_section_without_outer, _ = crop_white_borders(answer_section_with_outer)
        info_section_without_outer, info_section_inner_bbox = crop_white_borders(info_section_with_outer) # Get inner bbox too

        print("✅ Main OMR sheet sections extracted (in memory).")
        return answer_section_without_outer, info_section_without_outer, biggest_rect_bbox
    else:
        print("❌ No significant rectangular OMR sheet found. Try adjusting the threshold or contour area.")
        return None, None, None # Return None for all on failure

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
        return None, None, None, (0,0)

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
    total_marks_rel_bbox_y = (0, 0) # Store relative Y coordinates for total marks section

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
        return [], [] # Return empty lists on failure

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

# --- Section 2: Circle Detection and Option Numbering ---

def detect_filled_circles_in_segment(image_np_array, min_circle_area=150, max_circle_area=2000, min_circularity=0.7, fill_intensity_threshold=140):
    """
    Detects filled circles in a given image segment (NumPy array).
    Assigns option numbers (1, 2, 3, 4...) based on vertical position of the circles.

    Args:
        image_np_array (np.array): The image segment (NumPy array) to process.
        min_circle_area (int): Minimum area for a contour to be considered a circle.
        max_circle_area (int): Maximum area for a contour to be considered a circle.
        min_circularity (float): Minimum circularity (0 to 1, 1 is perfect circle).
        fill_intensity_threshold (int): Max mean pixel intensity for a circle to be considered filled (0-255).

    Returns:
        tuple: (selected_option_numbers, all_option_details)
            - selected_option_numbers (list): A list of integers representing the numbers of the selected options within this segment.
            - all_option_details (list): A list of dictionaries, each containing
                                        {'option_num': int, 'center': (x, y), 'radius': int, 'is_filled': bool}
                                        for all detected circular regions, sorted by y-coordinate.
    """
    # Convert image to grayscale for processing
    gray = cv2.cvtColor(image_np_array, cv2.COLOR_BGR2GRAY)

    # For debugging: Uncomment to see the grayscale segment (inside function)
    # cv2.imshow("Grayscale Segment (inside detect_filled_circles_in_segment)", gray)
    # cv2.waitKey(1)

    # Apply adaptive thresholding to enhance circles.
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    # For debugging: Uncomment to see the thresholded segment (inside function)
    # cv2.imshow("Thresholded Segment (inside detect_filled_circles_in_segment)", thresh)
    # cv2.waitKey(1)


    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    potential_options = []

    # Loop over the found contours to identify potential circles
    for c in contours:
        # Calculate perimeter and approximate polygon
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        area = cv2.contourArea(c)

        # Filter contours based on shape (number of vertices for a circle-like shape)
        # and size (area constraints).
        if len(approx) > 5 and area > min_circle_area and area < max_circle_area:
            # Get the minimum enclosing circle for the contour (for initial radius estimation)
            (x_enc, y_enc), radius = cv2.minEnclosingCircle(c)

            # Calculate centroid of the contour for potentially more accurate centering
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center = (cX, cY) # Use centroid as center
            else: # Fallback to minEnclosingCircle center if moment is zero (e.g., tiny contour)
                center = (int(x_enc), int(y_enc))

            radius = int(radius) # Keep radius from minEnclosingCircle for consistency

            # Calculate circularity to ensure it's round
            if peri > 0:
                circularity = (4 * np.pi * area) / (peri ** 2)
            else:
                circularity = 0

            # Filter by circularity
            if circularity > min_circularity:
                # Create a mask for the inner region of the detected circle
                mask = np.zeros(gray.shape, dtype=np.uint8)
                # Draw a slightly smaller filled circle on the mask to avoid border noise
                cv2.circle(mask, center, radius - 2, (255), -1)

                # Calculate the mean intensity of the original grayscale image within this mask.
                # A lower mean intensity means the area is darker (filled mark).
                mean_intensity = cv2.mean(gray, mask=mask)[0]

                is_filled = False
                # If the mean intensity is below the threshold, consider it filled.
                if mean_intensity < fill_intensity_threshold:
                    is_filled = True

                potential_options.append({
                    'center': center,
                    'radius': radius,
                    'is_filled': is_filled,
                    'mean_intensity': mean_intensity # Useful for debugging/tuning
                })

    # Sort the potential options by their y-coordinate to assign numbers vertically (1, 2, 3, ...)
    potential_options.sort(key=lambda opt: opt['center'][1])

    all_option_details = []
    selected_option_numbers = []

    # Assign option numbers and collect selected ones
    for option_num, opt in enumerate(potential_options, 1):
        option_detail = {
            'option_num': option_num,
            'center': opt['center'],
            'radius': opt['radius'],
            'is_filled': opt['is_filled']
        }
        all_option_details.append(option_detail)
        if opt['is_filled']:
            selected_option_numbers.append(option_num)

    return selected_option_numbers, all_option_details

# --- Section 3: Main Execution ---

if __name__ == "__main__":
    # Specify the path to your OMR image
    original_omr_image_path = "omr4.png" # Make sure this image exists in the same directory

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
        'Bottom_Col_8': [1], # This is the target for debugging
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
        answer_top_half_cols, answer_bottom_half_cols = split_answer_section_horizontally(answer_section_img, num_columns=15)

        # Calculate absolute offsets for the answer section
        answer_section_abs_x = omr_bbox[0]
        answer_section_abs_y = omr_bbox[1]
        answer_section_width = int(0.75 * omr_bbox[2])
        answer_section_height = omr_bbox[3]
        num_columns = 15 # Consistent with split_answer_section_horizontally
        col_width_actual = answer_section_width / num_columns

        # Define reduced circle size for drawing
        drawing_radius_scale = 0.6 # Kept at 0.6
        drawing_thickness = 3 # Line thickness for circles

        # Process top half columns
        for i, col_img in enumerate(answer_top_half_cols):
            selected_numbers, all_options_in_column = detect_filled_circles_in_segment(col_img)
            column_key = f"Top_Col_{i+1}"
            all_selected_options_by_column[column_key] = selected_numbers
            print(f"  {column_key}: Student selected = {selected_numbers}")

            col_abs_x_offset = answer_section_abs_x + int(i * col_width_actual)
            col_abs_y_offset = answer_section_abs_y # Top half starts at omr_y

            # Drawing logic for highlighting
            if column_key in correct_answers:
                correct_ans_for_col = correct_answers[column_key]

                # Check if the student's overall selection for this question is incorrect
                is_question_incorrect_overall = not (selected_numbers == correct_ans_for_col)

                for option_detail in all_options_in_column:
                    option_num = option_detail['option_num']
                    center_rel = option_detail['center']
                    radius = int(option_detail['radius'] * drawing_radius_scale) # Apply scale to radius
                    is_filled = option_detail['is_filled']

                    center_abs = (center_rel[0] + col_abs_x_offset, center_rel[1] + col_abs_y_offset)

                    if option_num in correct_ans_for_col:
                        if is_filled:
                            # Student correctly marked this option
                            cv2.circle(omr_image_for_drawing, center_abs, radius, (0, 255, 0), drawing_thickness) # Green: Correctly selected correct answer
                        elif is_question_incorrect_overall: # Only highlight correct answer in blue if student got it wrong
                            # Student missed this correct option, and the question overall is wrong - highlight correct in BLUE
                            cv2.circle(omr_image_for_drawing, center_abs, radius, (255, 0, 0), drawing_thickness) # Blue: Missed correct answer
                    elif is_filled: # Student marked an option that is not correct - highlight in RED
                        cv2.circle(omr_image_for_drawing, center_abs, radius, (0, 0, 255), drawing_thickness) # Red: Incorrectly selected option


                if selected_numbers == correct_ans_for_col:
                    total_marks_gained += 1
                    print(f"    ⭐ Correct! (+1 mark)")
                else:
                    print(f"    ❌ Incorrect. Correct answer: {correct_ans_for_col}")
            else:
                print(f"    ⚠️ Warning: No correct answer defined for {column_key}")

        # Process bottom half columns
        for i, col_img in enumerate(answer_bottom_half_cols):
            selected_numbers, all_options_in_column = detect_filled_circles_in_segment(col_img)
            column_key = f"Bottom_Col_{i+1}"
            all_selected_options_by_column[column_key] = selected_numbers
            print(f"  {column_key}: Student selected = {selected_numbers}")

            # --- Debugging specific column for visual inspection (UNCOMMENT TO USE) ---
            if column_key == 'Bottom_Col_8': # Adjust this to the column you want to debug
                print(f"--- Debugging {column_key} ---")
                # Show the raw image segment for this column
                cv2.imshow(f"Debug: {column_key} Raw Segment", col_img)
                # Convert to grayscale and threshold to see what contours are being found
                gray_debug = cv2.cvtColor(col_img, cv2.COLOR_BGR2GRAY)
                # Note: Using the same adaptive thresholding parameters as the function
                # FIX: Explicitly pass blockSize and C as keyword arguments to avoid `cv2.error`
                # FIX: Added thresholdType (cv2.THRESH_BINARY_INV) which was missing
                thresh_debug = cv2.adaptiveThreshold(gray_debug, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=11, C=2)
                cv2.imshow(f"Debug: {column_key} Thresholded", thresh_debug)

                # Draw detected contours on a copy of the segment for visual verification
                debug_contours_img = col_img.copy()
                debug_contours, _ = cv2.findContours(thresh_debug, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(debug_contours_img, debug_contours, -1, (0, 255, 255), 1) # Draw all contours in yellow
                cv2.imshow(f"Debug: {column_key} All Contours", debug_contours_img)

                # Draw detected circles (after filtering by area, circularity)
                debug_img_with_circles = col_img.copy()
                if all_options_in_column:
                    for option_detail in all_options_in_column:
                        center_rel = option_detail['center']
                        radius = int(option_detail['radius'] * drawing_radius_scale)
                        color = (0, 255, 0) # Green for detected circles
                        if option_detail['is_filled']:
                            color = (0, 165, 255) # Orange for detected FILLED circles
                        cv2.circle(debug_img_with_circles, center_rel, radius, color, 2)
                else:
                    print(f"No potential circles found in {column_key} based on current parameters.")
                cv2.imshow(f"Debug: {column_key} Detected Circles (Filtered)", debug_img_with_circles)

                cv2.waitKey(0) # Wait for a key press to close these debug windows
                cv2.destroyAllWindows()
            # ----------------------------------------------------

            col_abs_x_offset = answer_section_abs_x + int(i * col_width_actual)
            col_abs_y_offset = answer_section_abs_y + (answer_section_height // 2) # Bottom half offset

            # Drawing logic for highlighting
            if column_key in correct_answers:
                correct_ans_for_col = correct_answers[column_key]

                # Check if the student's overall selection for this question is incorrect
                is_question_incorrect_overall = not (selected_numbers == correct_ans_for_col)

                for option_detail in all_options_in_column:
                    option_num = option_detail['option_num']
                    center_rel = option_detail['center']
                    radius = int(option_detail['radius'] * drawing_radius_scale) # Apply scale to radius
                    is_filled = option_detail['is_filled']

                    center_abs = (center_rel[0] + col_abs_x_offset, center_rel[1] + col_abs_y_offset)

                    if option_num in correct_ans_for_col:
                        if is_filled:
                            # Student correctly marked this option
                            cv2.circle(omr_image_for_drawing, center_abs, radius, (0, 255, 0), drawing_thickness) # Green: Correctly selected correct answer
                        elif is_question_incorrect_overall: # Only highlight correct answer in blue if student got it wrong
                            # Student missed this correct option, and the question overall is wrong - highlight correct in BLUE
                            cv2.circle(omr_image_for_drawing, center_abs, radius, (255, 0, 0), drawing_thickness) # Blue: Missed correct answer
                    elif is_filled: # Student marked an option that is not correct - highlight in RED
                        cv2.circle(omr_image_for_drawing, center_abs, radius, (0, 0, 255), drawing_thickness) # Red: Incorrectly selected option


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
        student_id_img, subject_code_img, total_marks_img, total_marks_rel_bbox_y = extract_info_sections(info_section_img)

        try:
            # Define font parameters for OpenCV (cv2.putText)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5 # Adjust font scale as needed
            font_thickness = 4 # Adjust thickness for visibility
            font_color = (0, 0, 0) # Black color (BGR)

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
                # FIX: Corrected typo 'total_marks_abs_y_y_start' to 'total_marks_abs_y_start'
                text_center_y = total_marks_abs_y_start + ((total_marks_abs_y_end - total_marks_abs_y_start) // 2)

                # Get text size to properly center it
                text_string = str(total_marks_gained)
                text_size = cv2.getTextSize(text_string, font, font_scale, font_thickness)[0]

                # Calculate origin to truly center the text within the bounding box
                text_origin_x = int(text_center_x - text_size[0] / 2)
                text_origin_y = int(text_center_y + text_size[1] / 2) # Adjust for baseline


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
            cv2.waitKey(0) # Wait indefinitely until a key is pressed (window closed)
            cv2.destroyAllWindows() # Close all OpenCV windows

        except Exception as e:
            print(f"❌ An error occurred while drawing marks on the image or displaying it: {e}")

    print("\n--- OMR Processing Complete ---")
    print("\nSummary of all selected options per answer column:")
    for col_name, selected_opts in all_selected_options_by_column.items():
        print(f"{col_name}: {selected_opts}")

    print(f"\nTotal marks gained by the student: {total_marks_gained}")
