import cv2
import numpy as np
from PIL import Image
import pytesseract
import os

# Ensure pytesseract can find the tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Change this to the path where Tesseract is installed

def save_image(image, filename):
    cv2.imwrite(filename, image)

def detect_skew(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use Canny edge detection to find edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Use Hough Line Transform to detect lines in the edges
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=200, minLineLength=100, maxLineGap=10)
    
    angles = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                angle = np.arctan2(y2 - y1, x2 - x1)
                angles.append(angle)
                
    if len(angles) > 0:
        median_angle = np.median(angles)
        angle_deg = np.degrees(median_angle)
    else:
        angle_deg = 0
        
    return angle_deg

def deskew_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return deskewed

def preprocess_image(image_path, save_intermediate=False):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    # Detect skew angle and deskew the image
    angle = detect_skew(image)
    deskewed = deskew_image(image, -angle)  # Negative angle to correct rotation
    if save_intermediate:
        save_image(deskewed, "intermediate_deskewed.png")
    
    # Convert to grayscale
    gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
    if save_intermediate:
        save_image(gray, "intermediate_gray.png")
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    if save_intermediate:
        save_image(thresh, "intermediate_thresh.png")
    
    # Remove small noise
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    if save_intermediate:
        save_image(opening, "intermediate_opening.png")
    
    # Detect vertical and horizontal lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, np.array(opening).shape[0] // 40))
    detected_vertical = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, vertical_kernel, iterations=3)
    if save_intermediate:
        save_image(detected_vertical, "intermediate_detected_vertical.png")
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (np.array(opening).shape[1] // 40, 1))
    detected_horizontal = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, horizontal_kernel, iterations=3)
    if save_intermediate:
        save_image(detected_horizontal, "intermediate_detected_horizontal.png")
    
    # Combine vertical and horizontal lines
    grid = cv2.add(detected_vertical, detected_horizontal)
    if save_intermediate:
        save_image(grid, "intermediate_grid.png")
    
    # Subtract grid from image to isolate text
    isolated_text = cv2.subtract(opening, grid)
    if save_intermediate:
        save_image(isolated_text, "intermediate_isolated_text.png")
    
    return isolated_text

def extract_cells(image_path, save_intermediate=False):
    preprocessed_image = preprocess_image(image_path, save_intermediate)
    
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cell_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 20:  # Filter out small contours that may not be cells
            cell_image = preprocessed_image[y:y+h, x:x+w]
            if save_intermediate:
                save_image(cell_image, f"intermediate_cell_{x}_{y}.png")
            cell_images.append(cell_image)
    
    return cell_images

def extract_uuids_from_cells(cell_images):
    uuids = []
    custom_config = r'--oem 3 --psm 6'
    for cell_image in cell_images:
        text = pytesseract.image_to_string(cell_image, config=custom_config)
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) == 32 and all(c in '0123456789abcdef' for c in line.lower()):
                uuids.append(line)
    return uuids

def process_images_in_directory(directory_path, save_intermediate=False):
    all_uuids = {}
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(directory_path, filename)
            cell_images = extract_cells(image_path, save_intermediate)
            uuids = extract_uuids_from_cells(cell_images)
            all_uuids[filename] = uuids
            print(f"Extracted UUIDs from {filename}: {uuids}")
    return all_uuids

# Directory containing the images
directory_path = 'data/'  # Change this to the path of your image directory

# Process all images in the directory and print the extracted UUIDs
all_extracted_uuids = process_images_in_directory(directory_path, save_intermediate=True)

# Optionally, save the extracted UUIDs to a file
output_file = 'extracted_uuids.txt'
with open(output_file, 'w') as f:
    for filename, uuids in all_extracted_uuids.items():
        f.write(f"{filename}: {', '.join(uuids)}\n")

print(f"UUIDs have been extracted and saved to {output_file}")
