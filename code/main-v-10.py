import cv2
import numpy as np
import os

def display_and_split_biggest_rectangle_part(image_path):
    """
    Loads an image, finds the biggest rectangular part, and splits it into
    the answer section and info section, returning them as NumPy arrays.
    """
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return None, None # Return None for both parts on failure

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest_rect = None
    max_area = 0

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4 and cv2.contourArea(approx) > 1000:
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            if area > max_area:
                max_area = area
                biggest_rect = (x, y, w, h)

    if biggest_rect:
        x, y, w, h = biggest_rect
        rectangular_part = image[y:y + h, x:x + w]

        # Optional: Display the biggest rectangle
        # cv2.imshow("Biggest Rectangle Found", rectangular_part)
        # cv2.waitKey(0)

        # Split into initial answer section and info section
        # Assuming the answer section is on the left (75% width) and info on the right (25% width)
        answer_section_with_outer = rectangular_part[:, 0:int(0.75 * w)]
        info_section_with_outer = rectangular_part[:, int(0.75 * w):]

        # Function to find the main content within a section (removing white borders)
        def crop_white_borders(section_image):
            if section_image.size == 0:
                return section_image # Return empty if input is empty
            gray_section = cv2.cvtColor(section_image, cv2.COLOR_BGR2GRAY)
            # Adjust threshold if needed for better results
            _, thresh_section = cv2.threshold(gray_section, 200, 255, cv2.THRESH_BINARY_INV)
            contours_section, _ = cv2.findContours(thresh_section, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours_section:
                largest_contour_section = max(contours_section, key=cv2.contourArea)
                x_sec, y_sec, w_sec, h_sec = cv2.boundingRect(largest_contour_section)
                return section_image[y_sec:y_sec + h_sec, x_sec:x_sec + w_sec]
            return section_image # Return original if no contours found

        answer_section_without_outer = crop_white_borders(answer_section_with_outer)
        info_section_without_outer = crop_white_borders(info_section_with_outer)

        print("✅ Main sections extracted (in memory).")

        return answer_section_without_outer, info_section_without_outer
    else:
        print("❌ No significant rectangle found. Try adjusting the threshold or contour area.")
        return None, None # Return None for both parts on failure

def extract_info_sections(info_section_image):
    """
    Extracts the 'শিক্ষার্থী আইডি', 'বিষয় কোড', and 'মোট নম্বর' sections
    from the info_section_image (NumPy array) and returns them as NumPy arrays.
    """
    if info_section_image is None or info_section_image.size == 0:
        print("❌ Error: Info section image is empty or None.")
        return None, None, None

    img_height, img_width, _ = info_section_image.shape

    # --- ADJUST THESE PERCENTAGES ---
    student_id_section_height_percentage = 0.25
    subject_code_section_height_percentage = 0.25

    student_id_start_y = 0
    student_id_end_y = int(img_height * student_id_section_height_percentage)

    subject_code_start_y = student_id_end_y
    subject_code_end_y = subject_code_start_y + int(img_height * subject_code_section_height_percentage)

    total_marks_start_y = subject_code_end_y
    total_marks_end_y = img_height

    student_id_part = None
    subject_code_part = None
    total_marks_part = None

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

def split_answer_section_horizontally(answer_section_image):
    """
    Splits the answer_section_image (NumPy array) horizontally into two equal halves,
    returning them as NumPy arrays.
    """
    if answer_section_image is None or answer_section_image.size == 0:
        print(f"❌ Error: Answer section image is empty or None.")
        return None, None

    img_height, img_width, _ = answer_section_image.shape

    middle_y = img_height // 2

    top_half = answer_section_image[0:middle_y, 0:img_width]
    bottom_half = answer_section_image[middle_y:img_height, 0:img_width]

    print("✅ Answer section split into two halves (in memory).")
    return top_half, bottom_half


# === Main execution block ===
if __name__ == "__main__":
    original_omr_image_path = "omr2.png"  # Your original OMR image file

    print("--- Starting OMR Sheet Processing ---")

    # Step 1: Get the main answer and info sections
    answer_section_img, info_section_img = display_and_split_biggest_rectangle_part(original_omr_image_path)

    # Step 2: Process the info section if available
    if info_section_img is not None and info_section_img.size > 0:
        print("\n--- Extracting Info Sub-sections ---")
        student_id_img, subject_code_img, total_marks_img = extract_info_sections(info_section_img)

        # Optional: Display info sub-sections (no saving)
        if student_id_img is not None: cv2.imshow("Student ID", student_id_img)
        if subject_code_img is not None: cv2.imshow("Subject Code", subject_code_img)
        if total_marks_img is not None: cv2.imshow("Total Marks", total_marks_img)
        cv2.waitKey(1) # Small wait key to allow windows to pop up

    # Step 3: Split the answer section if available
    if answer_section_img is not None and answer_section_img.size > 0:
        print("\n--- Splitting Answer Section into Halves ---")
        answer_top_half_img, answer_bottom_half_img = split_answer_section_horizontally(answer_section_img)

        # Optional: Display answer halves (no saving)
        if answer_top_half_img is not None: cv2.imshow("Answer Top Half", answer_top_half_img)
        if answer_bottom_half_img is not None: cv2.imshow("Answer Bottom Half", answer_bottom_half_img)
        cv2.waitKey(0) # Wait for a key press to close all windows
        cv2.destroyAllWindows()
    else:
        print("\nAnswer section not available for splitting.")

    print("\n--- Processing Complete ---")