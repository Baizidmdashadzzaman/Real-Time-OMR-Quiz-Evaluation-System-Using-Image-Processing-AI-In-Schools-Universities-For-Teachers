from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk # For treeview
import os

def analyze_image_for_selection(image_path, percentage_to_crop, dark_pixel_threshold):
    """
    Crops the top specified percentage from an image, divides it into 4 horizontal
    strips, counts "dark" pixels in each, and returns the part with the most dark pixels.

    Args:
        image_path (str): The path to the input image.
        percentage_to_crop (float): The percentage of the image height to crop
                                    from the top (e.g., 18 for 18%).
        dark_pixel_threshold (int): The threshold for considering a pixel "dark".

    Returns:
        tuple: A tuple containing (filename, selected_part_number, dark_pixel_counts_list)
               Returns (None, None, None) if an error occurs.
    """
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            original_width, original_height = img.size

            crop_pixels = int(original_height * (percentage_to_crop / 100))
            initial_crop_box = (0, crop_pixels, original_width, original_height)
            cropped_img = img.crop(initial_crop_box)

            cropped_width, cropped_height = cropped_img.size
            strip_height = cropped_height // 4

            parts_boxes = [
                (0, 0, cropped_width, strip_height),
                (0, strip_height, cropped_width, 2 * strip_height),
                (0, 2 * strip_height, cropped_width, 3 * strip_height),
                (0, 3 * strip_height, cropped_width, cropped_height)
            ]

            dark_pixel_counts = []
            for i, box in enumerate(parts_boxes):
                part_img = cropped_img.crop(box)
                dark_pixel_count = 0
                for pixel in part_img.getdata():
                    if len(pixel) >= 3:
                        r, g, b = pixel[0], pixel[1], pixel[2]
                        if (r + g + b) < dark_pixel_threshold:
                            dark_pixel_count += 1
                    elif len(pixel) == 1:
                        if pixel[0] < (dark_pixel_threshold / 3):
                            dark_pixel_count += 1
                dark_pixel_counts.append(dark_pixel_count)

            if not dark_pixel_counts:
                return os.path.basename(image_path), "No parts found", []

            max_dark_pixels = max(dark_pixel_counts)
            most_dark_parts_indices = [i for i, count in enumerate(dark_pixel_counts) if count == max_dark_pixels]

            # If there's a tie, we might need a tie-breaking rule or indicate a tie.
            # For simplicity, we'll just pick the first one if there's a tie,
            # or indicate a tie if requested. For now, we'll return the first.
            if most_dark_parts_indices:
                selected_part_number = most_dark_parts_indices[0] + 1
                return os.path.basename(image_path), f"Part {selected_part_number}", dark_pixel_counts
            else:
                return os.path.basename(image_path), "No dark pixels found", dark_pixel_counts

    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return os.path.basename(image_path), "File Not Found", []
    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")
        return os.path.basename(image_path), f"Error: {e}", []

def process_answer_columns(directory_path, percentage_to_crop, dark_pixel_threshold):
    """
    Processes all image files in a given directory to determine student selections
    based on dark pixel counts and displays a summary.

    Args:
        directory_path (str): The path to the directory containing answer images.
        percentage_to_crop (float): The percentage of the image height to crop
                                    from the top.
        dark_pixel_threshold (int): The threshold for considering a pixel "dark".
    """
    results = []
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    if not image_files:
        print(f"No image files found in '{directory_path}'")
        return

    for filename in sorted(image_files): # Sort for consistent order
        full_path = os.path.join(directory_path, filename)
        print(f"Processing: {full_path}")
        file_name, selected_option, counts = analyze_image_for_selection(full_path, percentage_to_crop, dark_pixel_threshold)
        results.append({
            'filename': file_name,
            'selected_option': selected_option,
            'dark_pixel_counts': counts
        })

    # Display results in a Tkinter window
    root = tk.Tk()
    root.title("Student Selections Summary")

    # Create a Treeview widget
    tree = ttk.Treeview(root, columns=("Filename", "Selected Option", "Dark Pixel Counts"), show="headings")
    tree.heading("Filename", text="Filename")
    tree.heading("Selected Option", text="Selected Option")
    tree.heading("Dark Pixel Counts", text="Dark Pixel Counts")

    # Set column widths
    tree.column("Filename", width=150)
    tree.column("Selected Option", width=120)
    tree.column("Dark Pixel Counts", width=200)

    for res in results:
        tree.insert("", tk.END, values=(res['filename'], res['selected_option'], str(res['dark_pixel_counts'])))

    tree.pack(expand=True, fill="both", padx=10, pady=10)

    # Add a scrollbar
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")

    root.mainloop()

# --- Usage Example ---
input_directory = 'answer_columns/'
crop_percentage = 18
DARK_PIXEL_THRESHOLD = 150 # You can adjust this threshold

# Call the function to process images and show the summary
process_answer_columns(input_directory, crop_percentage, DARK_PIXEL_THRESHOLD)
