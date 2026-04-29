from .. import *

# Dataset config for CDLO RGB-D segmentation
C.project_root = "/workspace/kiat_crefle"
C.dataset_name = "CDLO"
C.dataset_path = osp.join(C.project_root, "data", "dformer_dataset")
C.rgb_root_folder = osp.join(C.dataset_path, "RGB")
C.rgb_format = ".png"
C.gt_root_folder = osp.join(C.dataset_path, "Label")
C.gt_format = ".png"
C.gt_transform = True  # Subtract 1 from labels: bg 0→255 (ignored), classes 1-5→0-4
C.x_root_folder = osp.join(C.dataset_path, "Depth")
C.x_format = ".png"
C.x_is_single_channel = True
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = True
C.num_train_imgs = 7560
C.num_eval_imgs = 1080
C.num_classes = 5  # Wire, Endpoint, Bifurcation, Connector, Noise (after gt_transform)
C.class_names = ["Wire", "Endpoint", "Bifurcation", "Connector", "Noise"]

"""Image Config"""
C.background = 255
C.image_height = 480
C.image_width = 640
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])
