import torch
import os
from PIL import Image
from torchvision.transforms import functional as F
import cv2
import torchvision.models.detection as detection_models
import tkinter as tk
from tkinter import filedialog, ttk, OptionMenu, StringVar, Label, messagebox
from screeninfo import get_monitors
import threading
import numpy as np
import time
from mss import mss

# Global flags
stop_capture = False
stop_video = False
model = None
model_selected = False  
debug = 0

# Check if CUDA is available and enable it if possible
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA is available. Using the GPU for processing.")
else:
    device = torch.device('cpu')
    print("CUDA is not available. Using the CPU for processing.")

def select_model_file():
    global model, model_selected
    model_file = filedialog.askopenfilename(filetypes=[("Model files", "*.pth")])
    if model_file:  # If a file is selected
        # Load the model file
        model = load_model(model_file)
        loaded_model_name = os.path.basename(model_file)
        messagebox.showinfo("Model Selected", f"The model {loaded_model_name} has been selected.")
        # Enable GUI elements related to capturing windows, opening videos, or exporting annotated videos
        enable_gui_elements()
        model_selected = True
        model_name.set("Model loaded: " + loaded_model_name)

def enable_gui_elements():
    # Enable GUI elements related to capturing windows, opening videos, or exporting annotated videos
    button1.config(state='normal')
    button2.config(state='normal')
    export_button.config(state='normal')

def disable_gui_elements():
    # Disable GUI elements related to capturing windows, opening videos, or exporting annotated videos
    button1.config(state='disabled')
    button2.config(state='disabled')
    export_button.config(state='disabled')

def load_model(model_file):
    start = model_file.rindex("model_") + len("model_")
    end = model_file.index("_epochs", start)
    model_name = model_file[start:end]

    model_func = getattr(detection_models, model_name)
    model = model_func(weights=None)
    checkpoint = torch.load(model_file)
    num_classes = checkpoint['model_state_dict']['roi_heads.box_predictor.cls_score.weight'].shape[0]
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = detection_models.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    model.load_state_dict(checkpoint['model_state_dict'])

    # Ensure the model is in evaluation mode and move it to the device selected
    model = model.eval().to(device)

    return model


#Used by the Capture Screen function to process the frame and draw the bounding boxes
def process_frame(frame, model):
    # Convert the frame to a PIL image
    frame_pil = Image.fromarray(frame)

    # Preprocess the image
    frame_tensor = F.to_tensor(frame_pil).unsqueeze(0).to(device)

    # Make the prediction
    prediction = model(frame_tensor)

    # Convert the tensors to numpy arrays
    prediction = {k: v.cpu().detach().numpy() for k, v in prediction[0].items()}

    
    # Get the selected confidence threshold 
    threshold = float(confidence_threshold.get().strip('%')) / 100

    # Draw the bounding boxes on the frame
    for box, score in zip(prediction['boxes'], prediction['scores']):
        if score > threshold:  # Only draw the box if the confidence score is above the selected threshold
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            cv2.putText(frame, f"{score:.2f}", (int(box[0]), int(box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return frame

#Used by the video functions to process the frames quicker
def process_batch(frames, model):
    
    # Convert frames to tensor and move to the device
    frames_tensor = torch.stack([torch.from_numpy(np.transpose(frame, (2, 0, 1))) for frame in frames]).to(device)

    frames_tensor = frames_tensor.float() / 255.0

    # Process frames
    processed_frames_dict = model(frames_tensor)

    # Get the selected confidence threshold
    threshold = float(confidence_threshold.get().strip('%')) / 100

    processed_frames = []
    for frame, frame_dict in zip(frames, processed_frames_dict):
        # Draw the bounding boxes on the frame
        for box, score in zip(frame_dict['boxes'], frame_dict['scores']):
            if score > threshold:  # Only draw the box if the confidence score is above the threshold
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                cv2.putText(frame, f"{score:.2f}", (int(box[0]), int(box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        processed_frames.append(frame)

    return processed_frames

def capture_screen(monitor):
    global stop_capture
    sct = mss()

    try:
        while True:
            if stop_capture:  # Check if the user has stopped capture by pressing the button
                break  

            # Create a dictionary that represents the monitor's screen
            monitor_screen = {"top": monitor.y, "left": monitor.x, "width": monitor.width, "height": monitor.height}

            # Capture the screen of the selected monitor
            img = sct.grab(monitor_screen)
            

            img_np = np.array(img)

            # Convert the image from BGRA to BGR
            frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)

            # Convert the image into numpy array
            img_np = np.array(frame_bgr)

            # Process the batch of frames and measure the time taken to do this
            start_time = time.time()
            frame = process_frame(img_np, model)
            if debug == 1:
                print(f"Time taken for process_frame: {time.time() - start_time} seconds")


            # Show the annotated frame
            cv2.imshow('Screen Capture', frame)

            if cv2.waitKey(1) == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")

    finally:
        stop_capture = False  
        cv2.destroyAllWindows()


    
def open_video():
    global stop_video
    filename = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
    if not filename:  
        stop_video = False  
        return  

    cap = cv2.VideoCapture(filename)

    # Get the video's frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps:
        fps = 25  # Default to 25 FPS as should be the case for most videos used
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wait_time = int(1000 / fps)
    batch_size = 3
    frames = []
    x = 0
    with torch.no_grad():
        while cap.isOpened():
            if stop_video:  # Check if the user has stopped the video
                stop_video = False  
                break

            # Read a batch of frames from the video
            ret, frame = cap.read()
            frames.append(frame)

            if len(frames) == batch_size:  # If collected enough frames for a batch
                # Process the batch of frames and measure the time taken
                start_time = time.time()
                processed_frames = process_batch(frames, model)
                if debug == 1:
                    print(f"Total Frames: {total_frames}")
                    print(f"Time taken for process_batch: {time.time() - start_time} seconds")
                frames = []  # Empty the list of frames
                
                # Display all frames currently processed
                for frame in processed_frames:
                    cv2.imshow('Video Playback', frame)
                    x += 1
                    progress_percent = (x / total_frames) * 100 
                    if progress_percent % 1 == 0:
                        print(f"Progress: {progress_percent}%")
                    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                        break

    cap.release()
    cv2.destroyAllWindows()

def export_annotated_video():
    filename = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
    if not filename:  
        return  

    cap = cv2.VideoCapture(filename)

    # Get the video's frame rate and size
    fps = cap.get(cv2.CAP_PROP_FPS) or 25  # Default to 25 FPS as most videos used will have this frame rate
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Get the base name of the file
    file_base = os.path.basename(filename)
    # Split the base name and the extension
    file_base, file_extension = os.path.splitext(file_base)
    #Split the name model variable so we can name the video appropriately
    name_of_model = model_name.get()
    name_of_model = name_of_model.split("Model loaded: ")[1]
    name_of_model = name_of_model.split(".pth")[0]

    # Create new file based on the model name and video name
    output_filename = (f"{file_base}_annotated_by_{name_of_model}{file_extension}")

    # Get the directory of the original file to add to the new filename
    file_dir = os.path.dirname(filename)

    # Combine the directory and filename to form the output path
    output_path = os.path.join(file_dir, output_filename)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    batch_size = 3
    frames = []

    with torch.no_grad():
        x = 0
        while True:
            # Read a frame from the video file
            ret, frame = cap.read()
            if not ret:  # If no frame is read, the video has finished
                break
            frames.append(frame)

            if len(frames) == batch_size:
                # Process the batch of frames and measure the time taken
                start_time = time.time()
                processed_frames = process_batch(frames, model)
                if debug == 1:
                    print(f"Total Frames: {total_frames}")
                    print(f"Time taken for process_batch: {time.time() - start_time} seconds")
                frames = []  # Empty the list of frames
                for processed_frame in processed_frames:
                    # Write the processed frames to the output file
                    out.write(processed_frame)
                    x += 1
                    # Calculate the progress to add to the bar
                    progress_percent = (x / total_frames) * 100 
                    
                    # Update the progress bar based on prior calc
                    progress['value'] = progress_percent
                    if progress_percent % 1 == 0:
                        print(f"Progress: {progress_percent}%")

                    # Update the GUI
                    root.update_idletasks()


        # If there are frames that haven't been processed yet process them now
        if frames:
            processed_frames = process_batch(frames, model)
            for processed_frame in processed_frames:
                out.write(processed_frame)

    cap.release()
    out.release()

    # Open the video for the user
    os.startfile(output_path)

    # Display a message telling the user where the file was saved to
    messagebox.showinfo("Export Complete", f"The annotated video was saved in your Downloads folder as {output_filename}")
def export_video_with_thread():
    export_video_thread = threading.Thread(target=export_annotated_video)
    export_video_thread.start()

def update_monitor_options(event=None):
    # Get the current monitors of the systen
    current_monitors = {f"Monitor {i+1}": monitor for i, monitor in enumerate(get_monitors())}

    # Update the monitor_dict
    monitor_dict.clear()
    monitor_dict.update(current_monitors)

    # Empty the current options in the dropdown menu
    dropDownMenu['menu'].delete(0, 'end')

    # Add the new options to the dropdown menu
    for monitor_name in monitor_dict.keys():
        dropDownMenu['menu'].add_command(label=monitor_name, command=tk._setit(var, monitor_name))
def update_debug():
    global debug
    debug = 1 if debug_mode.get() else 0

root = tk.Tk()
root.title("Visual Predictions Of A Model")

# Set the window size
window_width = 550
window_height = 650

# Get the screen width and height for screen cap if needed
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate the position to centre the window to display the app in the centre of the screen
position_top = int(screen_height / 2 - window_height / 2)
position_right = int(screen_width / 2 - window_width / 2)

# Set the window size and position so it appears in the centre of the screen
root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")


# Add a button to select the model file required to do anything
select_model_button = tk.Button(root, text="Select Model File", command=select_model_file, padx=10, pady=10)
select_model_button.pack(pady=20) 
# Create a StringVar to hold the model name so the user can see what model is selected after the message goes away
model_name = tk.StringVar()
model_name.set("No model loaded")  

# Create a label to display the model name selected so the user can see it whenever
model_label = tk.Label(root, textvariable=model_name)
model_label.pack()


def toggle_capture():
    global stop_capture

    if len(get_monitors()) < 2:  # If there's only one monitor do not allow screen capture
        messagebox.showerror("Error", "Screen capture requires at least two monitors.")
        return

    # Get the Monitor object for the selected monitor
    selected_monitor = monitor_dict[var.get()]

    if button1['text'] == 'Capture Window':
        button1['text'] = 'Stop Capture'
        stop_capture = False
        threading.Thread(target=capture_screen, args=(selected_monitor,)).start()
    else:
        button1['text'] = 'Capture Window'
        stop_capture = True

button1 = tk.Button(root, text="Capture Window", command=toggle_capture, padx=10, pady=10, state='disabled')
button1.pack(pady=20)   

def toggle_video():
    global stop_video
    if button2['text'] == 'Open Video File':
        button2['text'] = 'Stop Video'
        threading.Thread(target=open_video).start()
    else:
        button2['text'] = 'Open Video File'
        stop_video = True

button2 = tk.Button(root, text="Open Video File", command=toggle_video, padx=10, pady=10, state='disabled')
button2.pack(pady=20) 

export_button = tk.Button(root, text="Export Annotated Video", command=export_video_with_thread, padx=10, pady=10, state='disabled')
export_button.pack(pady=20)  

# Add a label for the progress bar so it's clear what it is
progress_label = tk.Label(root, text="Export Progress:")
progress_label.pack()

# Create a progress bar for the export video
progress = ttk.Progressbar(root, length=300, mode='determinate', maximum=100)
progress.pack()

# Create a variable to store the selected confidence threshold
confidence_threshold = StringVar(root)

# Set the default value for the threshold
confidence_threshold.set("50%")

# Create a label for the option menu
label = Label(root, text="Select the confidence threshold for the model:")
label.pack()

# Create the option menu
option_menu = OptionMenu(root, confidence_threshold, "0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%")
option_menu.pack()


# Create a dictionary mapping monitor indices to monitor objects
var = tk.StringVar(root)  

monitor_dict = {f"Monitor {i+1}": monitor for i, monitor in enumerate(get_monitors())}

# Create a function to enable the Capture Window button once a monitor has been selected
def enable_button(*args):
    button1.config(state='normal')
    button1['command'] = toggle_capture

var.trace('w', enable_button)  

# Create a label for the dropdown menu
label = tk.Label(root, text="Select Monitor:")
label.pack()
# Use the keys of the monitor_dict (the monitor names) for the OptionMenu
dropDownMenu = tk.OptionMenu(root, var, *monitor_dict.keys())
# Bind the function to the dropdown menu
dropDownMenu.bind('<Button-1>', update_monitor_options)
dropDownMenu.pack()

# Create a BooleanVar to hold the state of the checkbox
debug_mode = tk.BooleanVar()

# Create the checkbox
debug_checkbox = tk.Checkbutton(root, text="Enable Debug Mode", variable=debug_mode, command=update_debug)
debug_checkbox.pack()

# Add a message box to prompt the user to select a model file
messagebox.showinfo("Select Model", "Please select a model file to begin testing.")
update_monitor_options()
root.mainloop()