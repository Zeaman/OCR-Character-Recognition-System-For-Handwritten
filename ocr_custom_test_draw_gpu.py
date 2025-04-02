import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageTk, ImageDraw, ImageOps
import tkinter as tk
from tkinter import ttk, Label, messagebox, scrolledtext, filedialog
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = r'c:\My_Files\Projects\ocr_EXE_app\models\best_ocr_model_b.pth'

class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Drawing Recognition")
        self.root.geometry("1200x800")
        
        # Main frames
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Drawing area
        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        
        # Right panel - Results
        self.right_frame = tk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20)
        
        # Drawing canvas
        self.setup_drawing_area()
        
        # Image upload button
        self.upload_btn = ttk.Button(self.controls_frame, text="Upload Image", 
                                    command=self.upload_image)
        self.upload_btn.pack(pady=10)
        
        # Transformations (EXACTLY matching training code)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(48),
            transforms.CenterCrop(37),  # Changed from RandomCrop to CenterCrop
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # Class names
        self.class_names = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]
        
        # Load model
        self.model = self.load_model(MODEL_PATH)
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
    def setup_drawing_area(self):
        """Setup canvas and controls"""
        self.canvas_frame = tk.Frame(self.left_frame)
        self.canvas_frame.pack(pady=10)
        
        # Main drawing canvas
        self.canvas = tk.Canvas(self.canvas_frame, width=300, height=300, bg="white", 
                               highlightthickness=1, highlightbackground="black")
        self.canvas.pack()
        
        # Broken line border
        border_margin = 30
        w, h = 300, 300
        self.canvas.create_rectangle(
            border_margin, border_margin, w-border_margin, h-border_margin,
            outline="red", dash=(5,5), tags="border"
        )
        self.canvas.create_text(
            w/2, h-10, text="Draw inside the dashed border",
            fill="red", font=('Helvetica', 10), tags="border"
        )
        
        # Controls
        self.controls_frame = tk.Frame(self.left_frame)
        self.controls_frame.pack(pady=10)
        
        tk.Label(self.controls_frame, text="Brush Size:").pack()
        self.brush_size = tk.Scale(self.controls_frame, from_=20, to=60, orient=tk.HORIZONTAL)
        self.brush_size.set(40)
        self.brush_size.pack()
        
        # Buttons
        self.button_frame = tk.Frame(self.left_frame)
        self.button_frame.pack(pady=20)
        
        ttk.Button(self.button_frame, text="Capture Drawing", 
                  command=self.capture_drawing, width=15).pack(side=tk.LEFT, padx=10)
        ttk.Button(self.button_frame, text="Predict", 
                  command=self.predict, width=15).pack(side=tk.LEFT, padx=10)
        ttk.Button(self.button_frame, text="Clear Canvas", 
                  command=self.clear_canvas, width=15).pack(side=tk.LEFT, padx=10)
        
        # Image display
        self.image_frame = tk.Frame(self.left_frame)
        self.image_frame.pack(pady=10)
        tk.Label(self.image_frame, text="Processed Image (37x37):").pack()
        self.image_label = Label(self.image_frame)
        self.image_label.pack()
        
        # Results display
        self.setup_results_display()
        
        # Drawing variables
        self.last_x = None
        self.last_y = None
        self.drawing = False
        self.bind_drawing_events()
    
    def setup_results_display(self):
        """Setup the prediction results area"""
        self.result_frame = tk.Frame(self.right_frame)
        self.result_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header frame with title and buttons
        header_frame = tk.Frame(self.result_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title on the left
        tk.Label(header_frame, text="Prediction Results:", 
               font=('Helvetica', 14, 'bold')).pack(side=tk.LEFT, anchor=tk.W)
        
        # Button frame on the right
        button_frame = tk.Frame(header_frame)
        button_frame.pack(side=tk.RIGHT)
        
        # Developer info button
        ttk.Button(button_frame, text="Developer Info", 
                 command=self.show_developer_info, width=15).pack(side=tk.LEFT, padx=5)
        
        # Close button
        ttk.Button(button_frame, text="Close", 
                 command=self.root.destroy, width=10).pack(side=tk.LEFT)
        
        # Container with scrollbars
        self.text_container = tk.Frame(self.result_frame, bd=2, relief=tk.SUNKEN)
        self.text_container.pack(fill=tk.BOTH, expand=True)
        
        self.text_scroll_y = tk.Scrollbar(self.text_container)
        self.text_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.prediction_text = tk.Text(self.text_container, 
                                     width=50, height=20,
                                     font=('Courier', 12), wrap=tk.WORD,
                                     yscrollcommand=self.text_scroll_y.set)
        self.prediction_text.pack(fill=tk.BOTH, expand=True)
        self.text_scroll_y.config(command=self.prediction_text.yview)
        
        self.prediction_text.insert(tk.END, "Draw or upload an image, then click Predict")
        self.prediction_text.config(state=tk.DISABLED)
    
    def show_developer_info(self):
        """Display developer information in a copyable text window"""
        dev_window = tk.Toplevel(self.root)
        dev_window.title("Developer Information")
        dev_window.geometry("400x300")
        
        # Center the window relative to the parent
        self.center_window(dev_window)
        
        # Developer information
        dev_info = """
        Developer Information:
    ================
        Name: Amanuel Mihiret
            Upwork-freelancer
        Phone: +251918724038
        Email: zeaman48@gmail.com
        """
        
        # Text widget for copyable content
        text_frame = tk.Frame(dev_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        info_text = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set,
                        font=('Arial', 11), padx=10, pady=10)
        info_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=info_text.yview)
        
        info_text.insert(tk.END, dev_info.strip())
        
        # Make text read-only but allow copying
        info_text.bind("<Key>", lambda e: "break")  # Disable all key presses
        info_text.bind("<Button-1>", lambda e: "break")  # Disable mouse clicks that might modify
        info_text.bind("<Button-3>", self.show_context_menu)  # Right-click for context menu
        
        # Close button
        close_btn = ttk.Button(dev_window, text="Close", command=dev_window.destroy)
        close_btn.pack(pady=10)

    def show_context_menu(self, event):
        """Show context menu for copying text"""
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Copy", command=lambda: self.copy_to_clipboard(event.widget))
        menu.tk_popup(event.x_root, event.y_root)

    def copy_to_clipboard(self, widget):
        """Copy selected text to clipboard"""
        try:
            text = widget.selection_get()
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
        except tk.TclError:
            pass  # No text selected
    
    def center_window(self, window):
        """Center a window relative to its parent"""
        window.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        
        # Get parent window position and dimensions
        parent_x = self.root.winfo_x()
        parent_y = self.root.winfo_y()
        parent_width = self.root.winfo_width()
        parent_height = self.root.winfo_height()
        
        # Calculate position to center over parent
        x = parent_x + (parent_width // 2) - (width // 2)
        y = parent_y + (parent_height // 2) - (height // 2)
        
        window.geometry(f'{width}x{height}+{x}+{y}')
    
    def bind_drawing_events(self):
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
    
    def start_drawing(self, event):
        border_margin = 30
        if (border_margin <= event.x <= 300-border_margin and 
            border_margin <= event.y <= 300-border_margin):
            self.drawing = True
            self.last_x, self.last_y = event.x, event.y
        else:
            self.drawing = False
            messagebox.showwarning("Warning", "Please draw inside the dashed border")
    
    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            size = self.brush_size.get()
            
            if self.last_x and self.last_y:
                self.canvas.create_line(self.last_x, self.last_y, x, y, 
                                       fill="black", width=size, capstyle=tk.ROUND, 
                                       smooth=tk.TRUE)
            
            self.canvas.create_oval(x-size/2, y-size/2, x+size/2, y+size/2, 
                                  fill="black", outline="black")
            self.last_x, self.last_y = x, y
    
    def stop_drawing(self, event):
        self.drawing = False
        self.last_x = None
        self.last_y = None
    
    def clear_canvas(self):
        self.canvas.delete("all")
        # Redraw border
        border_margin = 30
        w, h = 300, 300
        self.canvas.create_rectangle(
            border_margin, border_margin, w-border_margin, h-border_margin,
            outline="red", dash=(5,5), tags="border"
        )
        self.canvas.create_text(
            w/2, h-10, text="Draw inside the dashed border",
            fill="red", font=('Helvetica', 10), tags="border"
        )
        self.image_label.config(image='')
        self.update_prediction_text("Draw or upload an image, then click Predict")
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )
        if file_path:
            try:
                self.original_image = Image.open(file_path).convert('L')
                # Display at 37x37
                display_img = self.original_image.resize((37, 37), Image.Resampling.NEAREST)
                img_tk = ImageTk.PhotoImage(display_img)
                self.image_label.config(image=img_tk)
                self.image_label.image = img_tk
                self.update_prediction_text("Image uploaded! Click 'Predict'")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def capture_drawing(self):
        try:
            # Create blank image (white background)
            img = Image.new("L", (300, 300), 255)
            draw = ImageDraw.Draw(img)
            
            # Get only the content inside the border
            border_margin = 30
            content_area = (border_margin, border_margin, 
                          300-border_margin, 300-border_margin)
            
            # Draw all canvas items
            for item in self.canvas.find_all():
                if "border" not in self.canvas.gettags(item):
                    coords = self.canvas.coords(item)
                    item_type = self.canvas.type(item)
                    
                    if item_type == "oval":
                        x0, y0, x1, y1 = coords
                        draw.ellipse([x0, y0, x1, y1], fill="black")
                    elif item_type == "line":
                        points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                        width = self.canvas.itemcget(item, "width")
                        draw.line(points, fill="black", width=int(float(width)))
            
            # Crop to the content area
            cropped_img = img.crop(content_area)
            
            # Resize to model input size (37x37)
            processed_img = cropped_img.resize((37, 37), Image.Resampling.LANCZOS)
            self.original_image = processed_img
            
            # Display
            display_img = processed_img.resize((150, 150), Image.Resampling.NEAREST)
            img_tk = ImageTk.PhotoImage(display_img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk
            self.update_prediction_text("Image captured! Click 'Predict'")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture image: {str(e)}")
    
    def load_model(self, model_path):
        try:
            model = models.resnet18(weights=None)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()

            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(512, 36)  # 0-9, A-Z
            )

            model.load_state_dict(torch.load(model_path), strict=False)
            model.eval()
            return model
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.root.destroy()
    
    def predict(self):
        if not hasattr(self, 'original_image'):
            messagebox.showwarning("Warning", "Please draw or upload an image first!")
            return

        try:
            # Apply transformations
            img_tensor = self.transform(self.original_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                top5_prob, top5_indices = torch.topk(probabilities, 5)

            predicted_char = self.class_names[predicted.item()]
            confidence = top5_prob[0][0].item()
            
            # Format results
            result_text = f"Predicted Character: {predicted_char}\n"
            result_text += f"Confidence: {confidence:.2%}\n\n"
            result_text += "Top 5 Predictions:\n"
            result_text += "------------------\n"
            
            for i, (prob, idx) in enumerate(zip(top5_prob[0], top5_indices[0])):
                result_text += f"{i+1}. {self.class_names[idx.item()]:<3} {prob.item():.2%}\n"
            
            self.update_prediction_text(result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.update_prediction_text(f"Prediction error: {str(e)}")
    
    def update_prediction_text(self, text):
        self.prediction_text.config(state=tk.NORMAL)
        self.prediction_text.delete(1.0, tk.END)
        self.prediction_text.insert(tk.END, text)
        self.prediction_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()