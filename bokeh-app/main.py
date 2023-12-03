import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

from cv_depth_blur import get_blur_image

# Variable to store the image paths
image_paths = []

# Function to run some code
def run_code(param1):
    plt.imshow(get_blur_image(image_paths[0], image_paths[1], int(param1)))
    plt.show()

# Function to open an image
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image_paths.append(file_path)

# Create the main window
root = tk.Tk()
root.title("Depth Based Background Blur")

root.geometry("800x600") 

# Center the window on the screen
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = 800  # Change this value to match the width of your window
window_height = 600  # Change this value to match the height of your window
x_position = (screen_width - window_width) // 2
y_position = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

# Change the background color to dark green
root.configure(bg='dark green')


# Function to run the code with parameters
def run_with_parameters():
    param1 = entry1.get()
    run_code(param1)

# Create a button to open the first image
btn_open_image1 = tk.Button(root, text="Open Image 1", command=open_image)
btn_open_image1.pack()

# Create a button to open the second image
btn_open_image2 = tk.Button(root, text="Open Image 2", command=open_image)
btn_open_image2.pack()

# Create fields to take parameters as input
entry1_label = tk.Label(root, text="Parameter: Subject Index (0 to 3):")
entry1_label.pack()
entry1 = tk.Entry(root)
entry1.pack()

# Create a button to run some code with parameters
btn_run_code = tk.Button(root, text="Run Code", command=run_with_parameters)
btn_run_code.pack()

# Run the main event loop
root.mainloop()
