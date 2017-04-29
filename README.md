# README

## Python Install
    Python version >= 3.5
    
## Install dependencies
    pip install -r requirements.txt

## How to Run the script?

### model.py
     python model.py  
This script train the model using the data saved from simulator.
   
### drive.py
    python drive.py [model.json] [image_folder]
    
The script load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.
 
### video.py 
    python video.py [image_folder] --fps 48
This script create a video based on images found in the [image_folder] directory. The name of the video will be name of the directory following by '.mp4'.
The video will run at 48 FPS. The default FPS is 60.
    
## Augmentation Techniques Used

### Use left & right camera images to simulate recovery
Using left and right camera images to simulate the effect of car wandering off to the side, and recovering. We will add a small angle .25 to the left camera and subtract a small angle of 0.25 from the right camera. The main idea being the left camera has to move right to get to center, and right camera has to move left.

### Flip the images horizontally
Since the data set has a lot more images with the car turning left than right because there are more left turns in the track.
So I flip the image horizontally to simulate turing right and also reverse the corresponding steering angle.

### Brightness Adjustment
In this you adjust the brightness of the image to simulate driving in different lighting conditions

With these augmentation techniques, you can practically generate infinite unique images for training your neural network.


### Preproceesing Images
The hood of the car is visible at the bottom of the image, we can remove it during pre-processing. 
The portion of the image above the horizon (where the road ends) can be ignored. 
Only the edges of the road and its curvature are relevant for determing the steering angle.
So I cropped 55 pixels from the top and 25 pixels from the bottom


### Data Generation Techniques Used
Data is augmented and generated on the fly using python generators. So for every epoch, the optimizer practically sees a new and augmented data set.

### Model Architecture

1. **Layer 1**: Conv layer with 32 5x5 filters, followed by ELU activation
2. **Layer 2**: Conv layer with 16 3x3 filters, ELU activation, Dropout(0.4) and 2x2 max pool
3. **Layer 3**: Conv layer with 16 3x3 filters, ELU activation, Dropout(0.4)
4. **Layer 4**: Fully connected layer with 1024 neurons, Dropout(0.3) and ELU activation
5. **Layer 5**: Fully connected layer with 1024 neurons, Dropout(0.3) and ELU activation



