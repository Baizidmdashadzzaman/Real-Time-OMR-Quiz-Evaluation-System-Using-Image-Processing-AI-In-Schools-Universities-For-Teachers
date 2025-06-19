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

def extract_student_id_section(image_path):
    """
    Extracts the 'শিক্ষার্থী আইডি' section based on a percentage of the image height.
    This approach is simpler and relies on the student ID section being at the very top.
    """
    info_section_image = cv2.imread(image_path)

    if info_section_image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return

    img_height, img_width, _ = info_section_image.shape

    # --- ADJUST THIS PERCENTAGE CAREFULLY ---
    # This is the crucial parameter for this method.
    # The 'শিক্ষার্থী আইডি' section (label + its white input box) is typically the first major section.
    # Try values between 0.15 (15%) and 0.30 (30%).
    # Start with a value, run the script, and observe the output.
    #   - If the section is cut off at the bottom (doesn't include the whole white box),
    #     INCREASE this percentage (e.g., from 0.22 to 0.25).
    #   - If the section includes part of the NEXT section ("বিষয় কোড"),
    #     DECREASE this percentage (e.g., from 0.22 to 0.18).
    student_id_section_height_percentage = 0.22 # Example: Take the top 22% of the info_section height

    # Calculate the height for the student ID section
    crop_height = int(img_height * student_id_section_height_percentage)

    # Ensure crop_height does not exceed image height
    if crop_height > img_height:
        crop_height = img_height
    if crop_height <= 0: # Ensure at least 1 pixel height
        crop_height = 1

    # Crop the image from the top
    # The student ID section is assumed to be at y=0, spanning the full width
    student_id_part = info_section_image[0:crop_height, 0:img_width]

    if student_id_part.size > 0: # Check if the cropped part is not empty
        cv2.imwrite("student_id_section.png", student_id_part)
        print("✅ Saved: student_id_section.png")

        cv2.imshow("Extracted Student ID Section (by height %)", student_id_part)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ Cropped student ID section is empty or invalid. Adjust the height percentage.")


# === Main execution block ===
if __name__ == "__main__":
    image_file = "omr4.png"  # Change this to your actual OMR image file

    print("--- Starting OMR Sheet Splitting ---")
    success_split = display_and_split_biggest_rectangle_part(image_file)

    if success_split and os.path.exists("info_section.png"):
        print("\n--- Starting Student ID Section Extraction ---")
        extract_student_id_section("info_section.png")
    else:
        print("\nInfo section was not successfully generated, skipping student ID extraction.")

    print("\n--- Processing Complete ---")