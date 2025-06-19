import cv2
import numpy as np
import os

def display_and_split_biggest_rectangle_part(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return

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

        cv2.imwrite("biggest_rectangle.png", rectangular_part)

        # Split into initial answer section and info section (which might have white borders)
        answer_section_with_white_outer = rectangular_part[:, 0:int(0.75 * w)]
        info_section_with_white_outer = rectangular_part[:, int(0.75 * w):]

        # --- Process info_section_with_white_outer to remove white outer part ---
        info_section_without_outer = info_section_with_white_outer # Initialize with full section as fallback
        gray_info = cv2.cvtColor(info_section_with_white_outer, cv2.COLOR_BGR2GRAY)
        # Adjust threshold if needed for better results
        _, thresh_info = cv2.threshold(gray_info, 200, 255, cv2.THRESH_BINARY_INV)
        contours_info, _ = cv2.findContours(thresh_info, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours_info:
            # Find the largest contour which should be the main content of the info section
            largest_contour_info = max(contours_info, key=cv2.contourArea)
            x_info, y_info, w_info, h_info = cv2.boundingRect(largest_contour_info)
            info_section_without_outer = info_section_with_white_outer[y_info:y_info + h_info, x_info:x_info + w_info]

        # --- Process answer_section_with_white_outer to remove white outer part ---
        answer_section_without_outer = answer_section_with_white_outer # Initialize with full section as fallback
        gray_answer = cv2.cvtColor(answer_section_with_white_outer, cv2.COLOR_BGR2GRAY)
        # Assuming a light background to dark content. Adjust threshold if needed.
        _, thresh_answer = cv2.threshold(gray_answer, 200, 255, cv2.THRESH_BINARY_INV)
        contours_answer, _ = cv2.findContours(thresh_answer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours_answer:
            # Find the largest contour which should be the main content of the answer section
            largest_contour_answer = max(contours_answer, key=cv2.contourArea)
            x_answer, y_answer, w_answer, h_answer = cv2.boundingRect(largest_contour_answer)
            answer_section_without_outer = answer_section_with_white_outer[y_answer:y_answer + h_answer, x_answer:x_answer + w_answer]


        # Save the processed parts
        cv2.imwrite("answer_section.png", answer_section_without_outer)
        cv2.imwrite("info_section.png", info_section_without_outer)

        print("✅ Saved: biggest_rectangle.png, answer_section.png, info_section.png")

        # Optional: Show the result
        cv2.imshow("Answer Section (without outer white)", answer_section_without_outer)
        cv2.imshow("Info Section (without outer white)", info_section_without_outer)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return True # Indicate that info_section.png was successfully created
    else:
        print("❌ No significant rectangle found. Try adjusting the threshold or contour area.")
        return False # Indicate failure

def extract_info_sections(image_path):
    """
    Extracts the 'শিক্ষার্থী আইডি', 'বিষয় কোড', and 'মোট নম্বর' sections based on height percentages.
    The 'মোট নম্বর' section now takes up the remaining height.
    """
    info_section_image = cv2.imread(image_path)

    if info_section_image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return

    img_height, img_width, _ = info_section_image.shape

    # --- ADJUST THESE PERCENTAGES ---
    # These are crucial parameters for this method.
    # The sum of percentages for each section should roughly correspond to
    # the portion of the info_section_image they occupy.

    # 1. Student ID Section (শিক্ষার্থী আইডি)
    # This is the height of the student ID section as a percentage of the *total info_section height*.
    student_id_section_height_percentage = 0.25

    # 2. Subject Code Section (বিষয় কোড)
    # This is the height of the subject code section as a percentage of the *total info_section height*.
    subject_code_section_height_percentage = 0.25

    # Calculate starting and ending y-coordinates for each section

    # Student ID Section: from 0 to its calculated height
    student_id_start_y = 0
    student_id_end_y = int(img_height * student_id_section_height_percentage)

    # Subject Code Section: starts immediately after student ID
    subject_code_start_y = student_id_end_y
    subject_code_end_y = subject_code_start_y + int(img_height * subject_code_section_height_percentage)

    # Total Marks Section: starts immediately after Subject Code and goes to the bottom of the info_section_image
    total_marks_start_y = subject_code_end_y
    total_marks_end_y = img_height # Takes the rest of the height!


    # Ensure coordinates do not exceed image boundaries
    # (Though with total_marks_end_y = img_height, these are mostly for earlier sections)
    student_id_end_y = min(student_id_end_y, img_height)
    subject_code_end_y = min(subject_code_end_y, img_height)


    # Extract Student ID Section
    if student_id_end_y > student_id_start_y:
        student_id_part = info_section_image[student_id_start_y:student_id_end_y, 0:img_width]
        cv2.imwrite("student_id_section.png", student_id_part)
        print("✅ Saved: student_id_section.png")
        cv2.imshow("Extracted Student ID Section", student_id_part)
    else:
        print("❌ Student ID section crop failed. Adjust student_id_section_height_percentage.")

    # Extract Subject Code Section
    if subject_code_end_y > subject_code_start_y:
        subject_code_part = info_section_image[subject_code_start_y:subject_code_end_y, 0:img_width]
        cv2.imwrite("subject_code_section.png", subject_code_part)
        print("✅ Saved: subject_code_section.png")
        cv2.imshow("Extracted Subject Code Section", subject_code_part)
    else:
        print("❌ Subject Code section crop failed. Adjust subject_code_section_height_percentage.")

    # Extract Total Marks Section (takes the rest of the height)
    if total_marks_end_y > total_marks_start_y:
        total_marks_part = info_section_image[total_marks_start_y:total_marks_end_y, 0:img_width]
        cv2.imwrite("total_marks_section.png", total_marks_part)
        print("✅ Saved: total_marks_section.png")
        cv2.imshow("Extracted Total Marks Section", total_marks_part)
    else:
        print("❌ Total Marks section crop failed. It might be starting beyond image height or has no remaining height.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# === Main execution block ===
if __name__ == "__main__":
    image_file = "omr4.png"  # Change this to your actual OMR image file

    print("--- Starting OMR Sheet Splitting ---")
    success_split = display_and_split_biggest_rectangle_part(image_file)

    if success_split and os.path.exists("info_section.png"):
        print("\n--- Starting Info Section Extraction (Student ID, Subject Code & Total Marks) ---")
        extract_info_sections("info_section.png")
    else:
        print("\nInfo section was not successfully generated, skipping detailed info extraction.")

    print("\n--- Processing Complete ---")