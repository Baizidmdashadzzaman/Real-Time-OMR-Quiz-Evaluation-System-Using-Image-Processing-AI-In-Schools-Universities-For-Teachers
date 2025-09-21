import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont # Using PIL for easier text drawing and display

def detect_filled_circles(image_path):
    """
    Detects filled circles in an image, useful for OMR sheet processing.
    Assigns option numbers (1, 2, 3, 4...) based on vertical position.

    Args:
        image_path (str): The path to the input image.

    Returns:
        tuple: A tuple containing:
            - PIL.Image.Image: The image with detected circles and status drawn on it.
            - list: A list of integers representing the numbers of the selected options.
    """
    try:
        # Load the image using OpenCV
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding. This helps in separating the circles
        # and filled areas more clearly, especially with varying lighting.
        # ADAPTIVE_THRESH_GAUSSIAN_C uses a weighted mean of the neighborhood.
        # THRESH_BINARY_INV means pixels brighter than the threshold become black (0),
        # and darker pixels become white (255). This is good for dark marks on light paper.
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours in the thresholded image.
        # RETR_EXTERNAL retrieves only the extreme outer contours.
        # CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments
        # and leaves only their end points.
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Prepare a copy of the original image to draw on using PIL for better text handling
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # Define font for drawing text (you might need to adjust the path or use a system font)
        # For simplicity, we'll try a common font, or default to generic if not found.
        try:
            # A common font path on Linux systems; adjust for Windows/macOS if needed
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            print("Could not load 'arial.ttf', using default PIL font.")
            font = ImageFont.load_default()

        min_circle_area = 200 # Adjust this based on your image and circle size
        max_circle_area = 2000 # Adjust this based on your image and circle size
        min_circularity = 0.5 # A value close to 1 indicates a perfect circle

        # Store potential options along with their y-coordinates to sort them later
        potential_options = []

        # Loop over the found contours
        for i, c in enumerate(contours):
            # Approximate the contour to a polygon
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)

            # Check if the contour has a reasonable number of vertices for a circle
            # (usually 8 or more for a good approximation of a circle)
            # And if its area is within expected bounds
            area = cv2.contourArea(c)

            if len(approx) > 5 and area > min_circle_area and area < max_circle_area:
                # Get bounding circle for the contour
                (x, y), radius = cv2.minEnclosingCircle(c)
                center = (int(x), int(y))
                radius = int(radius)

                # Calculate circularity: 4*pi*Area / Perimeter^2
                # A perfect circle has a circularity of 1.
                if peri > 0:
                    circularity = (4 * np.pi * area) / (peri ** 2)
                else:
                    circularity = 0

                if circularity > min_circularity:
                    # Draw the bounding circle on the OpenCV image for visual debugging (optional)
                    # cv2.circle(img_cv, center, radius, (0, 255, 0), 2)

                    # Determine if the circle is filled
                    # Create a mask for the current circle
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.circle(mask, center, radius - 2, (255), -1) # Fill inner part of circle

                    # Calculate the mean intensity of the original grayscale image within the mask
                    # A lower mean intensity (closer to 0 for black) indicates it's filled.
                    mean_intensity = cv2.mean(gray, mask=mask)[0]

                    is_filled = False
                    # Adjust this threshold based on how dark a filled circle is compared to empty
                    # For a dark mark on a light background, a lower intensity means filled.
                    if mean_intensity < 150: # Example threshold: adjust as needed
                        is_filled = True

                    potential_options.append({
                        'center': center,
                        'radius': radius,
                        'is_filled': is_filled,
                        'mean_intensity': mean_intensity # For debugging/tuning
                    })

        # Sort potential options by their y-coordinate (vertical position)
        potential_options.sort(key=lambda opt: opt['center'][1])

        selected_option_numbers = []

        # Assign option numbers and draw results
        for option_num, opt in enumerate(potential_options, 1):
            x, y = opt['center']
            radius = opt['radius']
            is_filled = opt['is_filled']

            status_text = f"Option {option_num}: "
            if is_filled:
                draw_color = (0, 255, 0) # Green for filled
                status_text += "Selected"
                selected_option_numbers.append(option_num)
            else:
                draw_color = (255, 0, 0) # Red for empty
                status_text += "Not Selected"

            # Draw circle outline on the PIL image
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         outline=draw_color, width=2)
            # Draw text on the PIL image
            draw.text((x + radius + 5, y), status_text, fill=draw_color, font=font)


        print(f"Detected {len(selected_option_numbers)} selected options: {selected_option_numbers}")
        # for i, option_center in enumerate(detected_options): # Old print statement
        #    print(f"Selected Option {i+1} at coordinates: {option_center}")

        return img_pil, selected_option_numbers

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, []

if __name__ == '__main__':
    # You will need to replace 'omr_single.png' with the actual path to your OMR image
    # For testing, save the image provided in the prompt as 'omr_single.png' in the
    # same directory as this script.
    image_path = 'answer_columns/question_23.png' # Replace with your image file name

    processed_image, selected_numbers = detect_filled_circles(image_path)

    if processed_image:
        # Save the processed image to a file
        output_path = 'omr_processed.png'
        processed_image.save(output_path)
        print(f"Processed image saved to {output_path}")
        print(f"Selected option numbers: {selected_numbers}")

        # If running in an environment that supports displaying images directly,
        # you can use processed_image.show() (requires an image viewer installed).
        # processed_image.show()
