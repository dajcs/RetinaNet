import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import argparse
import matplotlib.pyplot as plt
import pandas as pd

from model import create_model

from config import (
    NUM_CLASSES, DEVICE, CLASSES
)

np.random.seed(42)

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', 
    help='path to input image directory',
)
parser.add_argument(
    '--imgsz', 
    default=None,
    type=int,
    help='image resize shape'
)
parser.add_argument(
    '--threshold',
    default=0.25,
    type=float,
    help='detection threshold'
)
parser.add_argument(
    '--showimg',
    default=False,
    type=bool,
    help='visualize and save image overlayed with bbox'
)
args = vars(parser.parse_args())

os.makedirs('inference_outputs/images', exist_ok=True)

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the best model and trained weights.
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
# checkpoint = torch.load('outputs/last_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

# Directory where all the images are present.
DIR_TEST = args['input']
test_images = glob.glob(f"{DIR_TEST}/*.jpg")
print(f"Test instances: {len(test_images)}")

frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second.

df = pd.DataFrame(columns=['filename','class','bbox'])

wrong_class_preds = 0
wrong_pred_imgs = []

green = np.array([150.,220.,50.])
red = np.array([40., 77., 220.])

for i in range(len(test_images)):
    # Get the image file name for saving output later on.
    image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
    filename = image_name + '.png'
    true_cid = min(int(image_name[3:5]) + 1, 11) # we have an outlier, we need the min9
    true_class = CLASSES[true_cid]

    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    if args['imgsz'] is not None:
        image = cv2.resize(image, (args['imgsz'], args['imgsz']))
    print(image.shape)
    # BGR to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # Make the pixel range between 0 and 1.
    image /= 255.0
    # Bring color channels to front (H, W, C) => (C, H, W).
    image_input = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # Convert to tensor.
    image_input = torch.tensor(image_input, dtype=torch.float).to(DEVICE)
    # Add batch dimension.
    image_input = torch.unsqueeze(image_input, 0)
    start_time = time.time()
    # Predictions
    with torch.no_grad():
        outputs = model(image_input.to(DEVICE))
    end_time = time.time()

    # Get the current fps.
    fps = 1 / (end_time - start_time)
    # Total FPS till current frame.
    total_fps += fps
    frame_count += 1

    # Load all detection to CPU for further operations.
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # Carry further only if there are detected boxes.
    if len(outputs[0]['boxes']) != 0:
        # boxes = outputs[0]['boxes'].data.numpy()
        # scores = outputs[0]['scores'].data.numpy()
        # labels = outputs[0]['labels'].data.numpy()
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        labels = outputs[0]['labels'].data.numpy()
        # Filter out boxes according to `detection_threshold`.
        # boxes = boxes[scores >= args['threshold']].astype(np.int32)
        draw_boxes = boxes.copy()
        # Get all the predicited class names.
        pred_classes = [CLASSES[i] for i in labels]
        
            # Draw the bounding boxes and write the class name on top of it.
        for j, box in enumerate(draw_boxes):  # hits already sorted by confidence in decreasing order
            class_name = pred_classes[j]
            color = COLORS[CLASSES.index(class_name)]
            # Recale boxes.
            xmin = int((box[0] / image.shape[1]) * orig_image.shape[1])
            ymin = int((box[1] / image.shape[0]) * orig_image.shape[0])
            xmax = int((box[2] / image.shape[1]) * orig_image.shape[1])
            ymax = int((box[3] / image.shape[0]) * orig_image.shape[0])
            if xmin < 8: xmin = 0
            if ymin < 8: ymin = 0
            if xmax > 1016: xmax = 1024
            if ymax > 1016: ymax = 1024
            if class_name == true_class:              # class ok, store bbox if j=0, if j>0 overwrite bbox, hopefully this is better
                bbox = str([ymin, xmin, ymax, xmax])
                cv2.rectangle(orig_image,(xmin, ymin),(xmax, ymax), # color[::-1], 
                            green, 2)
                cv2.putText(orig_image,str(j) + '-> ' + class_name, (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, # color[::-1], 
                            green, 2, lineType=cv2.LINE_AA)
                break
            elif j==0:
                wrong_class_preds += 1
                wrong_pred_imgs.append(image_name + '.jpg')
                bbox = str([ymin, xmin, ymax, xmax])  # store best score bbox
            if args['showimg']:
                cv2.rectangle(orig_image, (xmin, ymin), (xmax, ymax), # color[::-1], 
                            red, 2)
                cv2.putText(orig_image,str(j) + '-> ' +  class_name, (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, # color[::-1], 
                            red, 2, lineType=cv2.LINE_AA)
        # write out image
        cv2.imwrite(f"inference_outputs/images/{image_name}.jpg", orig_image)
        # show it if requested
        if args['showimg']:
            plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
            plt.title(f'Prediction {image_name}')
            plt.show()
                
        print('scores:', scores[:3])
        print('pred_classes:', pred_classes[:3])
        print('true_class:', true_class)

#        pred_class = pred_classes[0]
#        bbox = str([ymin, xmin, ymax, xmax])
        df.loc[len(df)] = [filename, true_class, bbox]  

    else:
        print(f'image {image_name} has no bbox predictions')
        cv2.imwrite(f"inference_outputs/images/{image_name}.jpg", orig_image)
        plt.imshow(orig_image)
        plt.title(f'Prediction {image_name} - no bbox :-(')
        plt.show()
        
    # print(f"Image {i+1} done...")
    # print('-'*50)
print('\nTEST PREDICTIONS COMPLETE')
print(f'class accuracy: {(len(test_images)-wrong_class_preds)/len(test_images)*100:.2f}')
df.to_csv('inference_outputs/submission.csv.zip', index=False)
# cv2.destroyAllWindows()
# Calculate and print the average FPS.
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")
print('\n','Nr of missed classes:',len(wrong_pred_imgs),'\n')
with open('inference_outputs/wrong_pred.txt', 'w') as f:
    for i in wrong_pred_imgs:
        f.write(i+'\n')
#print(wrong_pred_imgs)
