import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pytesseract

# Ensure pytesseract can find the tesseract executable
# Change the path below to where Tesseract is installed on your system
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image OCR Application")

        self.label = tk.Label(root, text="Upload an image for OCR")
        self.label.pack(pady=20)

        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.text_area = tk.Text(root, wrap='word', width=60, height=20)
        self.text_area.pack(pady=20)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.perform_ocr(file_path)

    def perform_ocr(self, file_path):
        try:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)

            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.END, text)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while performing OCR: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()
