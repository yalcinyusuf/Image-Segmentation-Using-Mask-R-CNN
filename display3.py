import cv2
import numpy as np
import os
import sys

ROOT_DIR = os.path.abspath("C:/Users/yusuf/maskrcnn2/Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

class CustomConfig(Config):

    # Give the configuration a recognizable name
    NAME = "forklift"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 4 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 15

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 2

    DETECTION_MIN_CONFIDENCE = 0.9


class InferenceConfig(CustomConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = os.path.join(ROOT_DIR, "mask_rcnn_forklift_0004.h5")

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

class_names = ['BG', 'forklift']


# In[ ]:


def random_colors(number):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(number)]
    return colors


colors = random_colors(len(class_names))
class_dict = {name: color for name, color in zip(class_names, colors)}


# In[ ]:


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for n, c in enumerate(color):
        image[:, :, n] = np.where(mask == 1,
                                  image[:, :, n] *
                                  (1 - alpha) + alpha * c,
                                  image[:, :, n])
    return image


# In[ ]:


def display_instances(image, boxes, masks, class_ids, class_names, scores):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    for i in range(N):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = class_names[class_ids[i]]
        color = class_dict[label]
        score = scores[i] if scores is not None else None
        text = '{} {:.2f}'.format(label, score) if score else label

        mask = masks[:, :, i]

        images = apply_mask(image, mask, color)
        images = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        images = cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return image


# In[ ]:


import sys

sys.path
