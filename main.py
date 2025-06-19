import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

# --- Section 1: Image Preprocessing and Splitting ---

def display_and_split_biggest_rectangle_part(image_path):
    """
    Loads an image, finds the biggest rectangular part (assumed to be the OMR sheet),
    and splits it into the answer section (left) and info section (right),
    returning them as NumPy arrays.

    Args:
        image_path (str): The path to the input image file.

    Returns:
        tuple: (answer_section_img, info_section_img) as NumPy arrays,
               or (None, None) if the image cannot be loaded or no rectangle is found.
    """
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return None, None # Return None for both parts on failure

    # Convert to grayscale for contour detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary inverse threshold to make text/marks white and background black
    # This helps in finding contours of the main OMR area. Adjust threshold if needed.
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest_rect = None
    max_area = 0

    # Iterate through contours to find the largest quadrilateral (rectangle)
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        # Approximate the contour to a polygon. 0.02 * perimeter is a common epsilon.
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # Check if the approximated contour has 4 vertices (a quadrilateral)
        # and if its area is significant enough to be considered the OMR sheet.
        # Area threshold (1000) might need adjustment based on image resolution.
        if len(approx) == 4 and cv2.contourArea(approx) > 1000:
            x, y, w, h = cv2.boundingRect(approx) # Get bounding rectangle coordinates
            area = w * h
            if area > max_area: # Keep track of the biggest rectangle found
                max_area = area
                biggest_rect = (x, y, w, h)

    if biggest_rect:
        x, y, w, h = biggest_rect
        # Crop the biggest rectangular part from the original image
        rectangular_part = image[y:y + h, x:x + w]

        # Split into initial answer section (left 75% width) and info section (right 25% width)
        # These percentages are assumptions based on typical OMR layouts and might need tuning.
        answer_section_with_outer = rectangular_part[:, 0:int(0.75 * w)]
        info_section_with_outer = rectangular_part[:, int(0.75 * w):]

        # Function to find the main content within a section (removing surrounding white borders)
        def crop_white_borders(section_image):
            if section_image is None or section_image.size == 0:
                return section_image # Return empty if input is empty
            gray_section = cv2.cvtColor(section_image, cv2.COLOR_BGR2GRAY)
            # Re-threshold the section to find its internal content
            _, thresh_section = cv2.threshold(gray_section, 200, 255, cv2.THRESH_BINARY_INV)
            contours_section, _ = cv2.findContours(thresh_section, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours_section:
                # Find the largest contour within the section and crop to its bounding box
                largest_contour_section = max(contours_section, key=cv2.contourArea)
                x_sec, y_sec, w_sec, h_sec = cv2.boundingRect(largest_contour_section)
                return section_image[y_sec:y_sec + h_sec, x_sec:x_sec + w_sec]
            return section_image # Return original if no significant contours found

        # Apply border cropping to both sections
        answer_section_without_outer = crop_white_borders(answer_section_with_outer)
        info_section_without_outer = crop_white_borders(info_section_with_outer)

        print("✅ Main OMR sheet sections extracted (in memory).")
        return answer_section_without_outer, info_section_without_outer
    else:
        print("❌ No significant rectangular OMR sheet found. Try adjusting the threshold or contour area.")
        return None, None # Return None for both parts on failure

def extract_info_sections(info_section_image):
    """
    Extracts the 'শিক্ষার্থী আইডি', 'বিষয় কোড', and 'মোট নম্বর' sections
    from the info_section_image (NumPy array) based on assumed vertical percentages.
    Returns them as NumPy arrays.

    Note: This function is primarily for demonstrating the info section split.
    Further processing (OCR) would be needed to read the actual text/numbers.

    Args:
        info_section_image (np.array): The NumPy array of the info section.

    Returns:
        tuple: (student_id_part, subject_code_part, total_marks_part) as NumPy arrays,
               or (None, None, None) if the input is invalid.
    """
    if info_section_image is None or info_section_image.size == 0:
        print("❌ Error: Info section image is empty or None.")
        return None, None, None

    img_height, img_width, _ = info_section_image.shape

    # --- ADJUST THESE PERCENTAGES based on your OMR sheet layout ---
    # Assuming info section is divided roughly into three equal vertical parts
    # for Student ID, Subject Code, and Total Marks.
    section_height_percentage = 1 / 3 # Each section takes roughly 1/3rd height

    student_id_start_y = 0
    student_id_end_y = int(img_height * section_height_percentage)

    subject_code_start_y = student_id_end_y
    subject_code_end_y = subject_code_start_y + int(img_height * section_height_percentage)

    total_marks_start_y = subject_code_end_y
    total_marks_end_y = img_height

    student_id_part = None
    subject_code_part = None
    total_marks_part = None

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
        print("✅ Total Marks section extracted (in memory).")
    else:
        print("❌ Total Marks section crop failed.")

    return student_id_part, subject_code_part, total_marks_part

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

def detect_filled_circles_in_segment(image_np_array, min_circle_area=200, max_circle_area=2000, min_circularity=0.7, fill_intensity_threshold=150):
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
        list: A list of integers representing the numbers of the selected options within this segment.
    """
    # Convert image to grayscale for processing
    gray = cv2.cvtColor(image_np_array, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to enhance circles.
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

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
            # Get the minimum enclosing circle for the contour
            (x, y), radius = cv2.minEnclosingCircle(c)
            center = (int(x), int(y))
            radius = int(radius)

            # Calculate circularity to ensure it's round
            if peri > 0:
                circularity = (4 * np.pi * area) / (peri ** 2)
            else:
                circularity = 0

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

    selected_option_numbers = []

    # Assign option numbers and collect selected ones
    for option_num, opt in enumerate(potential_options, 1):
        if opt['is_filled']:
            selected_option_numbers.append(option_num)

    return selected_option_numbers

# --- Section 3: Main Execution ---

if __name__ == "__main__":
    # Specify the path to your OMR image
    original_omr_image_path = "omr4.png" # Make sure this image exists in the same directory

    print("--- Starting OMR Sheet Processing ---")

    # Step 1: Extract the main answer and info sections from the OMR sheet
    answer_section_img, info_section_img = display_and_split_biggest_rectangle_part(original_omr_image_path)

    # Dictionary to store all detected selected options
    all_selected_options_by_column = {}

    if answer_section_img is not None and answer_section_img.size > 0:
        print("\n--- Splitting Answer Section into Halves and then Columns ---")
        answer_top_half_cols, answer_bottom_half_cols = split_answer_section_horizontally(answer_section_img, num_columns=15)

        # Process top half columns
        for i, col_img in enumerate(answer_top_half_cols):
            # Convert OpenCV image (NumPy array) back to PIL Image if you need to draw on it
            # For this combined script, detect_filled_circles_in_segment just returns numbers
            selected_numbers = detect_filled_circles_in_segment(col_img)
            column_key = f"Top_Col_{i+1}"
            all_selected_options_by_column[column_key] = selected_numbers
            print(f"  {column_key}: Selected options = {selected_numbers}")

        # Process bottom half columns
        for i, col_img in enumerate(answer_bottom_half_cols):
            selected_numbers = detect_filled_circles_in_segment(col_img)
            column_key = f"Bottom_Col_{i+1}"
            all_selected_options_by_column[column_key] = selected_numbers
            print(f"  {column_key}: Selected options = {selected_numbers}")

    else:
        print("\nAnswer section not available for splitting and processing.")

    # Optional: Process the info section if available
    # This part doesn't involve circle detection but shows the split.
    if info_section_img is not None and info_section_img.size > 0:
        print("\n--- Extracting Info Sub-sections (for visualization/further OCR) ---")
        student_id_img, subject_code_img, total_marks_img = extract_info_sections(info_section_img)

        # Display extracted info sub-sections for debugging/verification
        # You would typically use OCR libraries (e.g., Tesseract) here to read text/numbers.
        # cv2.imshow("Student ID Section", student_id_img)
        # cv2.imshow("Subject Code Section", subject_code_img)
        # cv2.imshow("Total Marks Section", total_marks_img)
        # cv2.waitKey(1) # Small wait key to allow windows to pop up
        print("Note: Info sections are extracted. OCR would be the next step to read text.")


    print("\n--- OMR Processing Complete ---")
    print("\nSummary of all selected options per answer column:")
    for col_name, selected_opts in all_selected_options_by_column.items():
        print(f"{col_name}: {selected_opts}")

    # Keep windows open until a key is pressed (only if cv2.imshow was used for debug)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
