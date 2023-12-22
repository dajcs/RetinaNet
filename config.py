import torch

BATCH_SIZE = 4 # Increase / decrease according to GPU memeory.
RESIZE_TO = 512 # Resize the image for training and transforms.
NUM_EPOCHS = 3 # Number of epochs to train for.
NUM_WORKERS = 0 # Number of parallel workers for data loading.

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Training images and XML files directory.
TRAIN_DIR = 'data/BCCD.v3-raw.voc/train'
# Validation images and XML files directory.
VALID_DIR = 'data/BCCD.v3-raw.voc/valid'

# Classes: 0 index is reserved for background.
CLASSES = [
    '__background__', 'RBC', 'WBC', 'Platelets'
]



# Training images and XML files directory.
TRAIN_DIR = 'data/Stream1/train'
# Validation images and XML files directory.
VALID_DIR = 'data/Stream1/val'

# Classes: 0 index is reserved for background.
CLASSES = [
    '__background__',
    'proba_2',
    'cheops',
    'debris',
    'double_star',
    'earth_observation_sat_1',
    'lisa_pathfinder',
    'proba_3_csc',
    'proba_3_ocs',
    'smart_1',
    'soho',
    'xmm_newton',
]



NUM_CLASSES = len(CLASSES)

# Whether to visualize images after creating the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = True

# Location to save model and plots.
OUT_DIR = 'outputs'