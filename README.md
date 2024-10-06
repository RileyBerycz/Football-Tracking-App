# Visualisation

A pretrained object detection model for football 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Important 
Please note that since this uses CUDA to be effective you will need a GPU to use this software

Links for the models and a test video are below.
Download the models here:
`https://drive.google.com/file/d/1AX-SN5RLKl49oKFfHasn--4vYJqPsLzs/view?usp=sharing`

If you don't have a powerful GPU I'd use the My_data model as the other two models are larger and will take more
GPU memory, all of the models performance are relatively similar.

Test Video to try:
`https://drive.google.com/file/d/1ulUs3AH1-XsuDJuqt4mdyFrVoTr3Kifl/view?usp=sharing`

Any .mp4 video will work and the models will try look for a football even if one is present
stating their confidence of the object highlighted being a football.



## Installation

Make sure you install the requirements.txt before going further

If in the local path enter the command below
`pip install -r requirements.txt`

Otherwise either navigate to the local filepath or right-click on requirements.txt and copy it's filepath then use the bellow command
`pip install -r AddYourFilePath`

The only setup needed is to make sure you have the Visualisation.py

Downloaded the .pth model files

Open the Visualisation.py

I'd recommend doing this using CMD to see any error outputs possibly being memory related errors

And running this command if in the local directory
`python "Visualisation.py"`

Otherwise double click should be fine if not like before right-click the file and copy the file path and run this command
`python "Enter your file path here"`

## Usage

Once you've launched the python file a GUI will appear.

Make sure you select the .pth model file you want to use from the file explorer then from there choose what you want the model to annotate

Make sure you select one of the avaliable model files before use!

**Capture Window** only works on multiple monitor system due to how it works.

**Open Video File** is more to sample data from a video file to see the models predictions.

**Export Annotated Video** This may take a while but it while eventually export a video with all the annotations attached.

If something seems like it isn't working if you ran the file in a terminal you will see a process execution times if debug is enabled that verify it is indeed working

Please note this was made for testing purposes for a research report and may not work as is no longer being worked on.

Enjoy!


