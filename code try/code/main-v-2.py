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

        # Save the full biggest rectangle (for reference)
        cv2.imwrite("biggest_rectangle.png", rectangular_part)

        # Now split into answer section and info section
        answer_section = rectangular_part[:, 0:int(0.75 * w)]
        info_section = rectangular_part[:, int(0.75 * w):]

        # Save the two parts
        cv2.imwrite("answer_section.png", answer_section)
        cv2.imwrite("info_section.png", info_section)

        print("✅ Saved: biggest_rectangle.png, answer_section.png, info_section.png")

        # Optional: Show the result
        cv2.imshow("Answer Section", answer_section)
        cv2.imshow("Info Section", info_section)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ No significant rectangle found. Try adjusting the threshold or contour area.")


# === Main call ===
if __name__ == "__main__":
    image_file = "omr3.png"  # Change this to your actual image
    display_and_split_biggest_rectangle_part(image_file)
