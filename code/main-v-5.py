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
            largest_contour_info = max(contours_info, key=cv2.contourArea)
            x_info, y_info, w_info, h_info = cv2.boundingRect(largest_contour_info)
            info_section_without_outer = info_section_with_white_outer[y_info:y_info + h_info, x_info:x_info + w_info]

        # --- Process answer_section_with_white_outer to remove white outer part ---
        answer_section_without_outer = answer_section_with_white_outer # Initialize with full section as fallback
        gray_answer = cv2.cvtColor(answer_section_with_white_outer, cv2.COLOR_BGR2GRAY)
        # Adjust threshold if needed for better results. The answer section might have white background with black text.
        # So, a simple inverse threshold like this might not be ideal if the background is truly white and not pink.
        # You might need to experiment with the threshold value or method here.
        _, thresh_answer = cv2.threshold(gray_answer, 200, 255, cv2.THRESH_BINARY_INV) # Assuming a light background to dark content
        contours_answer, _ = cv2.findContours(thresh_answer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours_answer:
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
    else:
        print("❌ No significant rectangle found. Try adjusting the threshold or contour area.")


# === Main call ===
if __name__ == "__main__":
    image_file = "omr2.png"  # Change this to your actual image
    display_and_split_biggest_rectangle_part(image_file)