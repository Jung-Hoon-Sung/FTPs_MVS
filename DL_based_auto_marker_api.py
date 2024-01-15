import logging
logging.getLogger('jaxlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# General utilities
import os
from tqdm import tqdm
from time import time
from fastprogress import progress_bar
import gc
import numpy as np
import h5py
from collections import defaultdict

from copy import deepcopy
import os, argparse, h5py, warnings
from PIL import Image, ExifTags

# CV/ML
import cv2
import torch
print(torch.cuda.is_available())
import torch.nn.functional as F
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import sys
import traceback

import sqlite3
import numpy as np
import pandas as pd
# 3D reconstruction
import pycolmap

import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.transform import Rotation as R
import heapq
from sklearn.metrics.pairwise import euclidean_distances
#####################################################################################################
from functools import partial
import random
from torchvision.transforms.functional import resize, InterpolationMode
#####################################################################################################


def main_function(src, featureout, features_resolutions, checkpoint_path, orinet_path, keynet_path, affnet_path, hardnet_path,
                  sosnet_path, disk_path, loftr_path, num_feats = 40000):
    generated_files = []
    import kornia as K
    import kornia.feature as KF
    print('Kornia version', K.__version__)
    print('Pycolmap version', pycolmap.__version__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device=torch.device("cuda")

    for key, value in features_resolutions.items():
        print("Key:", key)
        print("Value:", value[0])
        
    dataset = "auto_markers"  
    scene = f"{value[0]}_{key}"

    ls_img_name = "factory134_"
    da_img_name = "100_"
    sorted_count = "20"

    ENSEMBLE = any(len(resolutions) > 1 for resolutions in features_resolutions.values()) or len(features_resolutions) > 1
    # colmap
    MAX_IMAGE_ID = 2**31 - 1
    IS_PYTHON3 = sys.version_info[0] >= 3
    # detect_features - general
    # num_feats = 40000
    matching_alg = 'adalam' # smnn, adalam
    min_matches = 10
    # matche_features
    ransac_iters = 128
    search_expansion = 16
    # incremental_mapping
    num_threads = -1
    max_num_models = 50 # 처리할 수 있는 최대 모델 수 /defualt 50
    ba_local_num_images = 6 # local bundle adjustment 단계에서 고려할 이미지의 수 /defualt 6
    ba_global_images_freq = 500
    ba_global_points_freq  = 250000
    init_num_trials = 200 # 초기 모델을 찾는 데 사용되는 RANSAC 반복 횟수 /defualt 200
    min_num_matches = 15 # 두 이미지 간에 최소한으로 매칭되어야 하는 특징점의 개수 / defualt 15
    min_model_size = 3 # RANSAC 시 필요한 최소한의 데이터 포인트 수 /defualt 3
    ba_local_max_num_iterations = 25 # defualt 25
    max_model_overlap = 20
    ba_local_max_refinements = 2
    ba_global_max_refinements = 5
    ba_global_images_ratio = 1.1 
    ba_global_points_ratio = 1.1
    # get_image_pairs_shortlist
    sim_th = 0.6 # should be strict
    min_pairs = 20
    exhaustive_if_less = 20
    model_name = 'convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384' 
    
    def arr_to_str(a):
        return ';'.join([str(x) for x in a.reshape(-1)])

    def load_torch_image(fname, K, device=torch.device('cpu')):
        img = K.image_to_tensor(cv2.imread(fname), False).float() / 255.
        img = K.color.bgr_to_rgb(img.to(device))
        return img

    # We will use ViT global descriptor to get matching shortlists.
    def get_global_desc(fnames, model,
                        device = torch.device('cuda')):
        model = model.eval()
        model= model.to(device)
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        global_descs_convnext=[]
        for i, img_fname_full in tqdm(enumerate(fnames),total= len(fnames)):
            key = os.path.splitext(os.path.basename(img_fname_full))[0]
            img = Image.open(img_fname_full).convert('RGB')
            timg = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                desc = model.forward_features(timg.to(device)).mean(dim=(-1,2))
                desc_np = desc.detach().cpu().numpy()
                desc_norm_np = desc_np / np.linalg.norm(desc_np, axis=1, keepdims=True)
                desc_norm = torch.from_numpy(desc_norm_np)

            #print (desc_norm)
            global_descs_convnext.append(desc_norm.detach().cpu())
        global_descs_all = torch.cat(global_descs_convnext, dim=0)
        return global_descs_all

    def get_img_pairs_exhaustive(img_fnames):
        index_pairs = []
        for i in range(len(img_fnames)):
            for j in range(i+1, len(img_fnames)):
                index_pairs.append((i,j))
        return index_pairs

    def get_image_pairs_shortlist(fnames,
                                sim_th = sim_th, # should be strict
                                min_pairs = min_pairs,
                                exhaustive_if_less = exhaustive_if_less,
                                device=torch.device('cpu')):
        num_imgs = len(fnames)

        if num_imgs <= exhaustive_if_less:
            return get_img_pairs_exhaustive(fnames)

        model = timm.create_model(model_name,
                                checkpoint_path=checkpoint_path, pretrained=False)
        
        model.eval()
        descs = get_global_desc(fnames, model, device=device)
        dm = torch.cdist(descs, descs, p=2).detach().cpu().numpy()
        # removing half
        mask = dm <= sim_th
        total = 0
        matching_list = []
        ar = np.arange(num_imgs)
        already_there_set = []
        for st_idx in range(num_imgs-1):
            mask_idx = mask[st_idx]
            to_match = ar[mask_idx]
            if len(to_match) < min_pairs:
                to_match = np.argsort(dm[st_idx])[:min_pairs]  
            for idx in to_match:
                if st_idx == idx:
                    continue
                if dm[st_idx, idx] < 1000:
                    matching_list.append(tuple(sorted((st_idx, idx.item()))))
                    total+=1
        matching_list = sorted(list(set(matching_list)))
        return matching_list

    CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
        camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        model INTEGER NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        params BLOB,
        prior_focal_length INTEGER NOT NULL)"""

    CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
        image_id INTEGER PRIMARY KEY NOT NULL,
        rows INTEGER NOT NULL,
        cols INTEGER NOT NULL,
        data BLOB,
        FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

    CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
        image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        name TEXT NOT NULL UNIQUE,
        camera_id INTEGER NOT NULL,
        prior_qw REAL,
        prior_qx REAL,
        prior_qy REAL,
        prior_qz REAL,
        prior_tx REAL,
        prior_ty REAL,
        prior_tz REAL,
        CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
        FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
    """.format(MAX_IMAGE_ID)

    CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
    CREATE TABLE IF NOT EXISTS two_view_geometries (
        pair_id INTEGER PRIMARY KEY NOT NULL,
        rows INTEGER NOT NULL,
        cols INTEGER NOT NULL,
        data BLOB,
        config INTEGER NOT NULL,
        F BLOB,
        E BLOB,
        H BLOB)
    """

    CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
        image_id INTEGER PRIMARY KEY NOT NULL,
        rows INTEGER NOT NULL,
        cols INTEGER NOT NULL,
        data BLOB,
        FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
    """

    CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
        pair_id INTEGER PRIMARY KEY NOT NULL,
        rows INTEGER NOT NULL,
        cols INTEGER NOT NULL,
        data BLOB)"""

    CREATE_NAME_INDEX = \
        "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

    CREATE_ALL = "; ".join([
        CREATE_CAMERAS_TABLE,
        CREATE_IMAGES_TABLE,
        CREATE_KEYPOINTS_TABLE,
        CREATE_DESCRIPTORS_TABLE,
        CREATE_MATCHES_TABLE,
        CREATE_TWO_VIEW_GEOMETRIES_TABLE,
        CREATE_NAME_INDEX
    ])

    def image_ids_to_pair_id(image_id1, image_id2):
        if image_id1 > image_id2:
            image_id1, image_id2 = image_id2, image_id1
        return image_id1 * MAX_IMAGE_ID + image_id2


    def pair_id_to_image_ids(pair_id):
        image_id2 = pair_id % MAX_IMAGE_ID
        image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
        return image_id1, image_id2


    def array_to_blob(array):
        if IS_PYTHON3:
            return array.tostring()
        else:
            return np.getbuffer(array)


    def blob_to_array(blob, dtype, shape=(-1,)):
        if IS_PYTHON3:
            return np.fromstring(blob, dtype=dtype).reshape(*shape)
        else:
            return np.frombuffer(blob, dtype=dtype).reshape(*shape)

    class COLMAPDatabase(sqlite3.Connection):

        @staticmethod
        def connect(database_path):
            return sqlite3.connect(database_path, factory=COLMAPDatabase)


        def __init__(self, *args, **kwargs):
            super(COLMAPDatabase, self).__init__(*args, **kwargs)

            self.create_tables = lambda: self.executescript(CREATE_ALL)
            self.create_cameras_table = \
                lambda: self.executescript(CREATE_CAMERAS_TABLE)
            self.create_descriptors_table = \
                lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
            self.create_images_table = \
                lambda: self.executescript(CREATE_IMAGES_TABLE)
            self.create_two_view_geometries_table = \
                lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
            self.create_keypoints_table = \
                lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
            self.create_matches_table = \
                lambda: self.executescript(CREATE_MATCHES_TABLE)
            self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

        def add_camera(self, model, width, height, params,
                    prior_focal_length=False, camera_id=None):
            params = np.asarray(params, np.float64)
            cursor = self.execute(
                "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
                (camera_id, model, width, height, array_to_blob(params),
                prior_focal_length))
            return cursor.lastrowid

        def add_image(self, name, camera_id,
                    prior_q=np.zeros(4), prior_t=np.zeros(3), image_id=None):
            cursor = self.execute(
                "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
                prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
            return cursor.lastrowid

        def add_keypoints(self, image_id, keypoints):
            assert(len(keypoints.shape) == 2)
            assert(keypoints.shape[1] in [2, 4, 6])

            keypoints = np.asarray(keypoints, np.float32)
            self.execute(
                "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
                (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

        def add_descriptors(self, image_id, descriptors):
            descriptors = np.ascontiguousarray(descriptors, np.uint8)
            self.execute(
                "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
                (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

        def add_matches(self, image_id1, image_id2, matches):
            assert(len(matches.shape) == 2)
            assert(matches.shape[1] == 2)

            if image_id1 > image_id2:
                matches = matches[:,::-1]

            pair_id = image_ids_to_pair_id(image_id1, image_id2)
            matches = np.asarray(matches, np.uint32)
            self.execute(
                "INSERT INTO matches VALUES (?, ?, ?, ?)",
                (pair_id,) + matches.shape + (array_to_blob(matches),))

        def add_two_view_geometry(self, image_id1, image_id2, matches,
                                F=np.eye(3), E=np.eye(3), H=np.eye(3), config=2):
            assert(len(matches.shape) == 2)
            assert(matches.shape[1] == 2)

            if image_id1 > image_id2:
                matches = matches[:,::-1]

            pair_id = image_ids_to_pair_id(image_id1, image_id2)
            matches = np.asarray(matches, np.uint32)
            F = np.asarray(F, dtype=np.float64)
            E = np.asarray(E, dtype=np.float64)
            H = np.asarray(H, dtype=np.float64)
            self.execute(
                "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (pair_id,) + matches.shape + (array_to_blob(matches), config,
                array_to_blob(F), array_to_blob(E), array_to_blob(H)))

    def get_focal(image_path, err_on_default=False):
        image         = Image.open(image_path)
        max_size      = max(image.size)

        exif = image.getexif()
        focal = None
        if exif is not None:
            focal_35mm = None
            for tag, value in exif.items():
                focal_35mm = None
                if ExifTags.TAGS.get(tag, None) == 'FocalLengthIn35mmFilm':
                    focal_35mm = float(value)
                    break

            if focal_35mm is not None:
                focal = focal_35mm / 35. * max_size
        
        if focal is None:
            if err_on_default:
                raise RuntimeError("Failed to find focal length")

            # failed to find it in exif, use prior
            FOCAL_PRIOR = 1.2
            focal = FOCAL_PRIOR * max_size

        return focal

    def create_camera(db, image_path, camera_model):
        image         = Image.open(image_path)
        width, height = image.size

        focal = get_focal(image_path)

        if camera_model == 'simple-pinhole':
            model = 0 # simple pinhole
            param_arr = np.array([focal, width / 2, height / 2])
        if camera_model == 'pinhole':
            model = 1 # pinhole
            param_arr = np.array([focal, focal, width / 2, height / 2])
        elif camera_model == 'simple-radial':
            model = 2 # simple radial
            param_arr = np.array([focal, width / 2, height / 2, 0.1])
        elif camera_model == 'opencv':
            model = 4 # opencv
            param_arr = np.array([focal, focal, width / 2, height / 2, 0., 0., 0., 0.])
            
        return db.add_camera(model, width, height, param_arr)

    def add_keypoints(db, h5_path, image_path, img_ext, camera_model, single_camera = True, keypoints_name = 'ensemble_keypoints.h5'):
        keypoint_f = h5py.File(os.path.join(h5_path, keypoints_name), 'r')

        camera_id = None
        fname_to_id = {}
        for filename in tqdm(list(keypoint_f.keys())):
            keypoints = keypoint_f[filename][()]

            fname_with_ext = filename# + img_ext
            path = os.path.join(image_path, fname_with_ext)
            if not os.path.isfile(path):
                raise IOError(f'Invalid image path {path}')

            if camera_id is None or not single_camera:
                camera_id = create_camera(db, path, camera_model)
            image_id = db.add_image(fname_with_ext, camera_id)
            fname_to_id[filename] = image_id

            db.add_keypoints(image_id, keypoints)

        return fname_to_id

    def add_matches(db, h5_path, fname_to_id, matches_name = 'ensemble_matches.h5'):
        match_file = h5py.File(os.path.join(h5_path, matches_name), 'r')
        
        added = set()
        n_keys = len(match_file.keys())
        n_total = (n_keys * (n_keys - 1)) // 2

        with tqdm(total=n_total) as pbar:
            for key_1 in match_file.keys():
                group = match_file[key_1]
                for key_2 in group.keys():
                    id_1 = fname_to_id[key_1]
                    id_2 = fname_to_id[key_2]

                    pair_id = image_ids_to_pair_id(id_1, id_2)
                    if pair_id in added:
                        warnings.warn(f'Pair {pair_id} ({id_1}, {id_2}) already added!')
                        continue
                
                    matches = group[key_2][()]
                    db.add_matches(id_1, id_2, matches)

                    added.add(pair_id)

                    pbar.update(1)

    def get_opencv_major_version(lib=None):
        # if the supplied library is None, import OpenCV
        if lib is None:
            import cv2 as lib

        # return the major version number
        return int(lib.__version__.split(".")[0])

    def is_cv2(or_better=False):
        # grab the OpenCV major version number
        major = get_opencv_major_version()

        # check to see if we are using *at least* OpenCV 2
        if or_better:
            return major >= 2

        # otherwise we want to check for *strictly* OpenCV 2
        return major == 2

    class KeyNetAffNetHardNet(KF.LocalFeature):

        def __init__(
            self,
            num_features: int = 5000,
            upright: bool = False,
            device = torch.device('cpu'),
            scale_laf: float = 1.0,
        ):
            ori_module = KF.PassLAF() if upright else KF.LAFOrienter(angle_detector=KF.OriNet(False)).eval()
            if not upright:
                weights = torch.load(orinet_path)['state_dict']
                ori_module.angle_detector.load_state_dict(weights)
            detector = KF.KeyNetDetector(False, num_features=num_features, ori_module=ori_module, aff_module=KF.LAFAffNetShapeEstimator(False).eval()).to(device)
            
            kn_weights = torch.load(keynet_path)['state_dict']
            detector.model.load_state_dict(kn_weights)
            affnet_weights = torch.load(affnet_path)['state_dict']
            detector.aff.load_state_dict(affnet_weights)
            
            """descriptors"""
            hardnet = KF.HardNet(False).eval()
            hn_weights = torch.load(hardnet_path)['state_dict']
            hardnet.load_state_dict(hn_weights)
            descriptor = KF.LAFDescriptor(hardnet, patch_size=32, grayscale_descriptor=True).to(device)
            super().__init__(detector, descriptor, scale_laf)

            """##############"""

    class KeyNetAffNetSoSNet(KF.LocalFeature):

        def __init__(
            self,
            num_features: int = 5000,
            upright: bool = False,
            device = torch.device('cpu'),
            scale_laf: float = 1.0,
        ):
            ori_module = KF.PassLAF() if upright else KF.LAFOrienter(angle_detector=KF.OriNet(False)).eval()
            if not upright:
                weights = torch.load(orinet_path)['state_dict']
                ori_module.angle_detector.load_state_dict(weights)
            detector = KF.KeyNetDetector(False, num_features=num_features, ori_module=ori_module, aff_module=KF.LAFAffNetShapeEstimator(False).eval()).to(device)
            
            kn_weights = torch.load(keynet_path)['state_dict']
            detector.model.load_state_dict(kn_weights)
            affnet_weights = torch.load(affnet_path)['state_dict']
            detector.aff.load_state_dict(affnet_weights)
            
            """descriptors"""
            sosnet = KF.SOSNet(False).eval()
            sos_weights = torch.load(sosnet_path)
            sosnet.load_state_dict(sos_weights)
            descriptor = KF.LAFDescriptor(sosnet, patch_size=32, grayscale_descriptor=True).to(device)
            super().__init__(detector, descriptor, scale_laf)
            
    def calculate_new_size(size, max_length):
        height, width = size
        aspect_ratio = float(width) / float(height)

        if height > width:
            new_height = max_length
            new_width = int(max_length * aspect_ratio)
        else:
            new_height = int(max_length / aspect_ratio)
            new_width = max_length

        return new_height, new_width

    def rotate_image(img, angle):
        height, width = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2 - 0.5, height / 2 - 0.5), angle, 1.0)
        matrix = np.vstack((matrix, [[0, 0, 1]]))
        return cv2.warpPerspective(img, matrix, (width, height)), matrix

    def rotate_coordinates(img, angle, keypoints):
        if keypoints is None or keypoints.shape[0] == 0 or keypoints.shape[1] != 2:
            raise ValueError("Invalid keypoints input")

        height, width = img.shape[:2]
        inv_matrix = cv2.getRotationMatrix2D((width / 2 - 0.5, height / 2 - 0.5), angle, 1.0)
        inv_matrix = np.vstack((inv_matrix, [[0, 0, 1]]))
        return cv2.perspectiveTransform(keypoints[None, :, :], inv_matrix)[0]

    def detect_features(img_fnames, num_feats=40000, 
                        upright=False, 
                        device=torch.device('cpu'), 
                        feature_dir='.featureout', 
                        local_feature='KeyNetAffNetHardNet', 
                        resolution=1200, 
                        PS=41,
                        n_features= 40000):
                    
        if local_feature == 'DISK':
            # Load DISK from Kaggle models so it can run when the notebook is offline.
            disk = KF.DISK().to(device)
            pretrained_dict = torch.load(disk_path, map_location=device)
            disk.load_state_dict(pretrained_dict['extractor'])
            # disk = KF.DISK.from_pretrained('depth').to(device)
            disk.eval()
                   
        elif local_feature == 'KeyNetAffNetHardNet':
            feature = KeyNetAffNetHardNet(num_feats, upright, device).to(device).eval()
        
        elif local_feature == 'KeyNetAffNetSoSNet':
            feature = KeyNetAffNetSoSNet(num_feats, upright, device).to(device).eval()
        
        elif local_feature == 'SIFT':
            feature = KF.SIFTFeature(num_features=n_features, upright=upright, device=device).to(device).eval()
            # sift = cv2.SIFT_create()
            # sift = cv2.xfeatures2d.SIFT_create()
            # sift_descriptor = KF.SIFTDescriptor(patch_size=PS, num_ang_bins=8, num_spatial_bins=4, rootsift=True, clipval=0.2).to(device).eval()
            
        else:
            raise NotImplementedError
        
        if not os.path.isdir(feature_dir):
            os.makedirs(feature_dir)    
        
        keypoints_name = f'{resolution}_{local_feature}_keypoints.h5'
        with h5py.File(f'{feature_dir}/{resolution}_{local_feature}_lafs.h5', mode='w') as f_laf, \
            h5py.File(f'{feature_dir}/{keypoints_name}', mode='w') as f_kp, \
            h5py.File(f'{feature_dir}/{resolution}_{local_feature}_descriptors.h5', mode='w') as f_desc:
            
            for img_path in progress_bar(img_fnames):
                img_fname = img_path.split('/')[-1]
                key = img_fname
                with torch.inference_mode():
                    # Load and rotate the image
                    timg = load_torch_image(img_path, K, device=device)
                    timg_permuted = timg.permute(0,2,3,1).cpu().numpy()
                    
                    if local_feature == 'DISK':
                        rotated_img, matrix = rotate_image(timg_permuted[0].astype(np.float32), 0)
                    else:
                        rotated_img, matrix = rotate_image(timg_permuted[0].astype(np.float32), 180)
                        
                    timg = torch.from_numpy(rotated_img).permute(2,0,1).unsqueeze(0).to(device)
                    H, W = timg.shape[2:]
                    resize_to = calculate_new_size((H, W), resolution)
                    timg_resized = K.geometry.resize(timg, resize_to, antialias=True)
                    print(f'Resized {timg.shape} to {timg_resized.shape} {resolution}')
                    h, w = timg_resized.shape[2:]
                    gc.collect()
                            
                    if local_feature == 'DISK':
                        features = disk(timg_resized, num_feats, pad_if_not_divisible=True)[0]
                        kps1, descs = features.keypoints, features.descriptors  
                        lafs = KF.laf_from_center_scale_ori(kps1[None], 96 * torch.ones(1, len(kps1), 1, 1, device=device))
                        
                    elif local_feature == 'KeyNetAffNetHardNet' or local_feature == 'KeyNetAffNetSoSNet':
                        lafs, resps, descs = feature(K.color.rgb_to_grayscale(timg_resized))
                        # Rotate keypoints
                        kpts = KF.get_laf_center(lafs).reshape(-1, 2).detach().cpu().numpy()
                        kpts = rotate_coordinates(timg_resized[0].permute(1,2,0).cpu().numpy(), 180, kpts)
                        
                        lafs[:,:,0,2] = torch.from_numpy(kpts[:, 0]).to(device)
                        lafs[:,:,1,2] = torch.from_numpy(kpts[:, 1]).to(device)
                        
                    elif local_feature == 'SIFT':
                        # OpenCV의 SIFT 키포인트 검출기를 사용하여 키포인트 검출
                        gray_img = cv2.cvtColor(timg_resized.cpu().numpy()[0].transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
                        lafs, resps, descs = feature(K.color.rgb_to_grayscale(timg_resized))
                        # Rotate keypoints
                        kpts = KF.get_laf_center(lafs).reshape(-1, 2).detach().cpu().numpy()
                        kpts = rotate_coordinates(timg_resized[0].permute(1,2,0).cpu().numpy(), 180, kpts)
                        
                        lafs[:,:,0,2] = torch.from_numpy(kpts[:, 0]).to(device)
                        lafs[:,:,1,2] = torch.from_numpy(kpts[:, 1]).to(device)
                        
                    
                    lafs[:,:,0,:] *= float(W) / float(w)
                    lafs[:,:,1,:] *= float(H) / float(h)
                    desc_dim = descs.shape[-1]
                    kpts = KF.get_laf_center(lafs).reshape(-1, 2).detach().cpu().numpy()
                    
                    descs = descs.reshape(-1, desc_dim).detach().cpu().numpy()
                    f_laf[key] = lafs.detach().cpu().numpy()
                    f_kp[key] = kpts
                    f_desc[key] = descs
                
                    # Free some memory
                    del lafs, descs, kpts, timg, timg_resized
                    torch.cuda.empty_cache()
                    
                    # else:
                    #     # 결과 저장
                    #     f_kp[key] = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints]) if keypoints else np.array([])
                    #     f_desc[key] = descs.cpu().numpy() if descs.nelement() != 0 else np.array([])

                    #     del keypoints, descs, patches, timg, timg_resized
                    #     torch.cuda.empty_cache()
                            
        return keypoints_name

                    
    def get_unique_idxs(A, dim=0):
        # https://stackoverflow.com/questions/72001505/how-to-get-unique-elements-and-their-firstly-appeared-indices-of-a-pytorch-tenso
        unique, idx, counts = torch.unique(A, dim=dim, sorted=True, return_inverse=True, return_counts=True)
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0],device=cum_sum.device), cum_sum[:-1]))
        first_indices = ind_sorted[cum_sum]
        return first_indices

    def match_features(img_fnames,
                    index_pairs,
                    feature_dir=".featureout",
                    device=torch.device("cpu"),
                    min_matches=10, 
                    force_mutual = True,
                    matching_alg="adalam",
                    local_feature = "KeyNetAffNetHardNet",
                    resolution=1200
                    ):
        assert matching_alg in ['smnn', 'adalam']
        matches_name = f'{resolution}_{local_feature}_matches.h5'
        with h5py.File(f'{feature_dir}/{resolution}_{local_feature}_lafs.h5', mode='r') as f_laf, \
            h5py.File(f'{feature_dir}/{resolution}_{local_feature}_descriptors.h5', mode='r') as f_desc, \
            h5py.File(f'{feature_dir}/{resolution}_{local_feature}_matches.h5', mode='w') as f_match:

            for pair_idx in progress_bar(index_pairs):
                gc.collect()
                idx1, idx2 = pair_idx
                fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
                key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
                lafs1 = torch.from_numpy(f_laf[key1][...]).to(device)
                lafs2 = torch.from_numpy(f_laf[key2][...]).to(device)
                desc1 = torch.from_numpy(f_desc[key1][...]).to(device)
                desc2 = torch.from_numpy(f_desc[key2][...]).to(device)
                if matching_alg == 'adalam':
                    img1, img2 = cv2.imread(fname1), cv2.imread(fname2)
                    hw1, hw2 = img1.shape[:2], img2.shape[:2]
                    adalam_config = KF.adalam.get_adalam_default_config()
                    adalam_config['force_seed_mnn']= False
                    adalam_config['search_expansion'] = search_expansion
                    adalam_config['ransac_iters'] = ransac_iters
                    adalam_config['device'] = device
                    dists, idxs = KF.match_adalam(desc1, desc2,
                                                lafs1, lafs2, # Adalam takes into account also geometric information
                                                hw1=hw1, hw2=hw2,
                                                config=adalam_config) # Adalam also benefits from knowing image size
                else:   
                    dists, idxs = KF.match_smnn(desc1, desc2, 0.98)
                if len(idxs)  == 0:
                    continue
                
                # Force mutual nearest neighbors
                if force_mutual:
                    first_indices = get_unique_idxs(idxs[:,1])
                    idxs = idxs[first_indices]
                    dists = dists[first_indices]
                n_matches = len(idxs)
                if False:
                    print (f'{key1}-{key2}: {n_matches} matches')
                group  = f_match.require_group(key1)
                if n_matches >= min_matches:
                    group.create_dataset(key2, data=idxs.detach().cpu().numpy().reshape(-1, 2))
                    
        return matches_name

    def get_top_k(keypoints, descriptors, k):
        positions, scores = keypoints[:,:2], keypoints[:,2]

        # top-k selection
        idxs = scores.argsort()[-k:]

        return positions[idxs], descriptors[idxs] / 1.41, scores[idxs]

    def inlier_matches(inlier_mask, matches):
        if inlier_mask is not None:
            matches_after_ransac = np.array([match for match, is_inlier in zip(matches, inlier_mask) if is_inlier])
        else:
            matches_after_ransac = np.array([])
            
        return matches_after_ransac

    def match_features2(img_fnames,
                        index_pairs,
                        feature_dir=".featureout",
                        device=torch.device('cpu'),
                        local_feature="LoFTR",
                        min_matches=15,
                        resolution=1200
                        ):
        gc.collect()

        if local_feature == "LoFTR":
            matcher = KF.LoFTR(pretrained=None)
            matcher.load_state_dict(torch.load(loftr_path)['state_dict'])
            matcher = matcher.to(device).eval()
            # matcher = KF.LoFTR(pretrained='outdoor').eval().to(device)  # online
            
        else:
            print("No model!!")

        # First we do pairwise matching, and then extract "keypoints" from loftr matches.
        matches_name = f'{resolution}_{local_feature}_matches.h5'
        keypoints_name = f'{resolution}_{local_feature}_keypoints.h5'
        with h5py.File(f'{feature_dir}/{resolution}_{local_feature}_matches.h5', mode='w') as f_match:
            for pair_idx in progress_bar(index_pairs):
                idx1, idx2 = pair_idx
                fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
                key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]

                if local_feature == "LoFTR":
                    # Load img1
                    timg1 = K.color.rgb_to_grayscale(load_torch_image(fname1, K, device=device))
                    H1, W1 = timg1.shape[2:]
                    resize_to1 = calculate_new_size((H1, W1), resolution)
                    timg_resized1 = K.geometry.resize(timg1, resize_to1, antialias=True)
                    h1, w1 = timg_resized1.shape[2:]

                    # Load img2
                    timg2 = K.color.rgb_to_grayscale(load_torch_image(fname2, K, device=device))
                    H2, W2 = timg2.shape[2:]
                    resize_to2 = calculate_new_size((H2, W2), resolution)
                    timg_resized2 = K.geometry.resize(timg2, resize_to2, antialias=True)
                    h2, w2 = timg_resized2.shape[2:]

                    with torch.inference_mode():
                        input_dict = {"image0": timg_resized1,"image1": timg_resized2}
                        correspondences = matcher(input_dict)
                    mkpts0 = correspondences['keypoints0'].cpu().numpy()
                    mkpts1 = correspondences['keypoints1'].cpu().numpy()
                    confidence = correspondences['confidence'].cpu().numpy()
        
                    mkpts0[:,0] *= float(W1) / float(w1)
                    mkpts0[:,1] *= float(H1) / float(h1)

                    mkpts1[:,0] *= float(W2) / float(w2)
                    mkpts1[:,1] *= float(H2) / float(h2)
                else:
                    print("No model!!")

                n_matches = len(mkpts1)
                group  = f_match.require_group(key1)
                if n_matches >= min_matches:
                    group.create_dataset(key2, data=np.concatenate([mkpts0, mkpts1], axis=1))

        # Let's find unique loftr pixels and group them together.
        kpts = defaultdict(list)
        match_indexes = defaultdict(dict)
        total_kpts=defaultdict(int)
        with h5py.File(f'{feature_dir}/{resolution}_{local_feature}_matches.h5', mode='r') as f_match:
            for k1 in f_match.keys():
                group  = f_match[k1]
                for k2 in group.keys():
                    matches = group[k2][...]
                    total_kpts[k1]
                    kpts[k1].append(matches[:, :2])
                    kpts[k2].append(matches[:, 2:])
                    current_match = torch.arange(len(matches)).reshape(-1, 1).repeat(1, 2)
                    current_match[:, 0]+=total_kpts[k1]
                    current_match[:, 1]+=total_kpts[k2]
                    total_kpts[k1]+=len(matches)
                    total_kpts[k2]+=len(matches)
                    match_indexes[k1][k2]=current_match

        for k in kpts.keys():
            kpts[k] = np.round(np.concatenate(kpts[k], axis=0))
        unique_kpts = {}
        unique_match_idxs = {}
        out_match = defaultdict(dict)
        for k in kpts.keys():
            uniq_kps, uniq_reverse_idxs = torch.unique(torch.from_numpy(kpts[k]),dim=0, return_inverse=True)
            unique_match_idxs[k] = uniq_reverse_idxs
            unique_kpts[k] = uniq_kps.numpy()
        for k1, group in match_indexes.items():
            for k2, m in group.items():
                m2 = deepcopy(m)
                m2[:,0] = unique_match_idxs[k1][m2[:,0]]
                m2[:,1] = unique_match_idxs[k2][m2[:,1]]
                mkpts = np.concatenate([unique_kpts[k1][ m2[:,0]],
                                        unique_kpts[k2][  m2[:,1]],
                                    ],
                                    axis=1)
                unique_idxs_current = get_unique_idxs(torch.from_numpy(mkpts), dim=0)
                m2_semiclean = m2[unique_idxs_current]
                unique_idxs_current1 = get_unique_idxs(m2_semiclean[:, 0], dim=0)
                m2_semiclean = m2_semiclean[unique_idxs_current1]
                unique_idxs_current2 = get_unique_idxs(m2_semiclean[:, 1], dim=0)
                m2_semiclean2 = m2_semiclean[unique_idxs_current2]
                out_match[k1][k2] = m2_semiclean2.numpy()
        with h5py.File(f'{feature_dir}/{resolution}_{local_feature}_keypoints.h5', mode='w') as f_kp:
            for k, kpts1 in unique_kpts.items():
                f_kp[k] = kpts1
                
        with h5py.File(f'{feature_dir}/{resolution}_{local_feature}_matches.h5', mode='w') as f_match:
            for k1, gr in out_match.items():
                group  = f_match.require_group(k1)
                for k2, match in gr.items():
                    group[k2] = match
                        
        return keypoints_name, matches_name

    def ensemble_keypoint_matches(keypoints, matches, ensemble_keypoints, ensemble_matches, ensemble_output_dir):
        bias_table = {}

        merged_keypoints = {}
        keypoints.sort()
        for keypoint in keypoints:
            splitext = keypoint.split('_')
            resolution = splitext[0]
            local_feature = splitext[1]

            with h5py.File(f'{feature_dir}/{keypoint}', 'r') as data:
                for key, value in data.items():
                    if key not in merged_keypoints:
                        merged_keypoints[key] = np.array(value)  # directly save as numpy array
                    else:
                        merged_keypoints[key] = np.concatenate([merged_keypoints[key], np.array(value)])  # stack the arrays vertically
                    
                    # resolution
                    bias_table.setdefault(resolution, {})
                    # resolution - local_feature
                    bias_table[resolution].setdefault(local_feature, {})
                    
                    # resolution - local_feature - index
                    bias_table[resolution][local_feature].setdefault("index", 0)
                    idx = keypoints.index(keypoint)
                    bias_table[resolution][local_feature]["index"] = idx
                    
                    # resolution - local_feature - key
                    bias_table[resolution][local_feature].setdefault(key, 0)
                    bias_table[resolution][local_feature][key] = np.array(value).shape[0]
        # print(merged_keypoints)  # print merged_keypoints for checking
        
        merged_matches = {}
        matches.sort()
        for match in matches:
            splitext = match.split('_')
            resolution = splitext[0]
            local_feature = splitext[1]
            with h5py.File(f'{feature_dir}/{match}', 'r') as data:  
                for key, value in data.items():
                    merged_matches.setdefault(key, {})                
                    for key2, value2 in value.items():                    
                        np_val2 = np.array(value2)
                        merged_matches[key].setdefault(key2, [])
                        
                        idx = bias_table[resolution][local_feature]["index"]
                        if idx == 0:
                            adjusted_val2 = np_val2                        
                            merged_matches[key][key2].append(adjusted_val2)
                        else:
                            bias1, bias2 = 0, 0
                            for i in range(idx):
                                resol = keypoints[i].split('_')[0]
                                loc_feat = keypoints[i].split('_')[1]
                                bias1 += bias_table[resol][loc_feat][key]
                                bias2 += bias_table[resol][loc_feat][key2]
                            
                            n = np_val2.shape[0]
                            adjusted_val2 = np.concatenate(((np_val2[:,0] + bias1).reshape(n, -1), (np_val2[:,1] + bias2).reshape(n, -1)), axis=1)
                            
                            if len(merged_matches[key][key2]) == 0:
                                merged_matches[key][key2].append(adjusted_val2)
                            else:
                                merged_matches[key][key2][0] = np.concatenate([merged_matches[key][key2][0], adjusted_val2])
        # print(merged_matches)  # print merged_matches for checking
                
        # Save ensemble results
        with h5py.File(os.path.join(ensemble_output_dir, ensemble_keypoints), 'w') as keypoints_out:
            for fname, data in merged_keypoints.items():
                keypoints_out.create_dataset(fname, data=data)  # use data directly

        with h5py.File(os.path.join(ensemble_output_dir, ensemble_matches), 'w') as matches_out:
            for fname1, data1 in merged_matches.items():
                group = matches_out.create_group(fname1)
                for fname2, data2 in data1.items():
                    if isinstance(data2[0], list):
                        group.create_dataset(fname2, data=np.concatenate(data2))  # concatenate the numpy arrays before saving
                    else:
                        group.create_dataset(fname2, data=data2[0])

    def get_max_resolution(images):
        max_length = 0
        for img_path in images:
            img = cv2.imread(img_path)
            max_length = max(max_length, img.shape[0], img.shape[1])
        return max_length

    def import_into_colmap(img_dir,
                        feature_dir ='.featureout',
                        database_path = 'colmap.db',
                        img_ext='.jpg',
                        keypoints_name="ensemble_keypoints.h5",
                        matches_name="ensemble_matches.h5"):
        db = COLMAPDatabase.connect(database_path)
        db.create_tables()
        single_camera = False
        fname_to_id = add_keypoints(db, feature_dir, img_dir, img_ext, 'simple-radial', single_camera, keypoints_name)
        add_matches(db, feature_dir, fname_to_id, matches_name)

        db.commit()
        return

    def filter_matches(df_errors_filtered, matches_h5_path, output_h5_path):
        # Load existing matches
        with h5py.File(matches_h5_path, 'r') as f:
            matches = {k: {kk: vv[:] for kk, vv in v.items()} for k, v in f.items()}

        # Create a new h5 file for filtered matches
        with h5py.File(output_h5_path, 'w') as f:
            for _, row in df_errors_filtered.iterrows():
                image_name = row['image_name']
                if image_name not in matches:
                    continue
                group = f.require_group(image_name)
                for match_image_name, match_indices in row['matches']:
                    if match_image_name in matches[image_name]:
                        group.create_dataset(match_image_name, data=matches[image_name][match_image_name][match_indices])
        return output_h5_path
    
    def create_custom_csv(df_with_reset_index, keypoints, reconstruction, output_csv):
        csv_data = []

        for _, row in df_with_reset_index.iterrows():
            point_id = row['Point3DId']
            for img_name in row['Images']:
                # 이미지 이름으로부터 이미지 ID를 얻음
                img_id = image_name_to_id[img_name]
                img_name_no_ext = os.path.splitext(img_name)[0]

                for i, point2D in enumerate(reconstruction.images[img_id].points2D):
                    if point2D.point3D_id == point_id:
                        x, y = keypoints[img_name][i]
                        csv_data.append({
                            "img_id": img_name_no_ext,
                            "marker": f"point {point_id}",
                            "x": x,
                            "y": y
                        })

        # DataFrame 생성 및 CSV로 저장
        df_csv = pd.DataFrame(csv_data)
        df_csv.to_csv(output_csv, index=False)
    
    # Get data from csv.
    data_dict = {}
    # Replace this with your image directory
    img_dir = f'{src}'
    # List of valid image file extensions
    img_exts = ['.JPG', '.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    # Get all files in the directory with valid image extensions
    img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) and os.path.splitext(f)[1].lower() in img_exts]
    data_dict[dataset] = {}
    data_dict[dataset][scene] = img_files

    total_max_resolution_time = 0
    total_shortlisting_time = 0
    total_feature_detection_time = 0
    total_feature_matching_time = 0
    total_bundle_adjustment_time = 0

    for dataset in data_dict:
        for scene in data_dict[dataset]:
            print(f'{dataset} / {scene} -> {len(data_dict[dataset][scene])} images')
    out_results = {}
    timings = {"shortlisting":[],
            "feature_detection": [],
            "feature_matching":[],
            "RANSAC": [],
            "Reconstruction": []}
    gc.collect()
    datasets = []
    for dataset in data_dict:
        datasets.append(dataset)
    start_time = time()
    for dataset in datasets:
        print(dataset)
        if dataset not in out_results:
            out_results[dataset] = {}
        for scene in data_dict[dataset]:
            print(scene)
            img_dir = f'{src}'
            if not os.path.exists(img_dir):
                continue
            try:
                out_results[dataset][scene] = {}
                img_fnames = [f'{src}/{x}' for x in data_dict[dataset][scene]]
                print (f"Got {len(img_fnames)} images")
                # feature_dir = f'{featureout}/{dataset}_{scene}'
                feature_dir = f'{featureout}/{dataset}/{scene}'
                # generated_files.append(feature_dir)
                if not os.path.isdir(feature_dir):
                    os.makedirs(feature_dir, exist_ok=True)
                t=time()
                
                #Calculate the maximum resolution
                max_resolution_start_time = time()
                max_length = get_max_resolution(img_fnames)
                print(f"Max image length: {max_length}")
                max_resolution_end_time = time()
                max_resolution_time = max_resolution_end_time - max_resolution_start_time
                total_max_resolution_time += max_resolution_time
                
                shortlisting_start_time = time()
                
                index_pairs = get_image_pairs_shortlist(img_fnames,
                                    sim_th = sim_th, # should be strict
                                    min_pairs = min_pairs, # we select at least min_pairs PER IMAGE with biggest similarity
                                    exhaustive_if_less = exhaustive_if_less,
                                    device=device)
                # Print the index pairs and the corresponding image file names
                for pair in index_pairs:
                    img1 = img_fnames[pair[0]]
                    img2 = img_fnames[pair[1]]
                    print(f"Index pair: {pair}, Image pair: ({img1}, {img2})")
                t=time() -t 
                timings['shortlisting'].append(t)
                
                shortlisting_end_time = time()
                shortlisting_time = shortlisting_end_time - shortlisting_start_time
                total_shortlisting_time += shortlisting_time
                print(f"{len(index_pairs)}, pairs to match, {shortlisting_time:.4f} sec")

                print (f'{len(index_pairs)}, pairs to match, {t:.4f} sec')
                gc.collect()
                feature_detection_start_time = time()
                t=time()            
                for local_feature in features_resolutions:
                    for resolution in features_resolutions[local_feature]:
                        print(f"Got {local_feature} & {resolution}")                       
                        if local_feature == 'KeyNetAffNetHardNet' or local_feature == 'KeyNetAffNetSoSNet' or local_feature == 'DISK':
                            keypoints_name = detect_features(img_fnames,
                                num_feats = num_feats,
                                upright = False,
                                device=device,
                                feature_dir=feature_dir,
                                local_feature=local_feature,
                                resolution=resolution 
                                )                       
                            gc.collect()
                            t=time() -t 
                            timings['feature_detection'].append(t)
                            print(f'Features detected in  {t:.4f} sec')
                            t=time()
                            matches_name = match_features(img_fnames, index_pairs, feature_dir=feature_dir, device=device,
                                                        min_matches=min_matches, matching_alg=matching_alg,
                                                        local_feature=local_feature, resolution=resolution)               
                        elif local_feature == 'LoFTR':
                            feature_matching_start_time = time()
                            keypoints_name, matches_name = match_features2(img_fnames, index_pairs,
                                                                        feature_dir=feature_dir, device=device,
                                                                        local_feature=local_feature,
                                                                        min_matches=min_matches,
                                                                        resolution=resolution)       
                        else:
                            print('Redefine the model!!!')
                ENSEMBLE = any(len(resolutions) > 1 for resolutions in features_resolutions.values()) or len(features_resolutions) > 1           
                if ENSEMBLE == True:
                    keypoints, matches = [], []
                    for (root, dirs, files) in os.walk(feature_dir):
                        print("# root : " + root)
                        if len(dirs) > 0:
                            for dir_name in dirs:
                                print("dir: " + dir_name)
                        if len(files) > 0:
                            for file_name in files:
                                file = os.path.splitext(file_name)[0]
                                extension = os.path.splitext(file_name)[1]
                                print("file: " + file_name)                 
                                if extension == ".h5":
                                    types = file.split("_")[-1]
                                    if types == "keypoints":
                                        keypoints.append(file_name)
                                    elif types == "matches":
                                        matches.append(file_name)
                                    else:
                                        print("byebye")
                                else:
                                    print("Bye")                    
                    keypoints_name = 'ensemble_keypoints.h5'
                    matches_name = 'ensemble_matches.h5'
                    ensemble_keypoint_matches(keypoints, matches, ensemble_keypoints=keypoints_name, ensemble_matches=matches_name, 
                                    ensemble_output_dir=feature_dir)   
                t=time() -t 
                timings['feature_matching'].append(t)
                feature_matching_end_time = time()
                feature_matching_time = feature_matching_end_time - feature_matching_start_time
                total_feature_matching_time += feature_matching_time
                print(f"Features matched in {feature_matching_time:.4f} sec")
                
                print(f'Features matched in  {t:.4f} sec')
                           
                database_path = f'{feature_dir}/colmap.db'
                if os.path.isfile(database_path):
                    os.remove(database_path)
                    
                bundle_adjustment_start_time = time()
                gc.collect()
                import_into_colmap(img_dir, 
                                feature_dir=feature_dir,
                                database_path=database_path, 
                                keypoints_name=keypoints_name, 
                                matches_name=matches_name)
                output_path = f'{feature_dir}/colmap_rec'
                t=time()
                pycolmap.match_exhaustive(database_path)
                t=time() - t 
                timings['RANSAC'].append(t)
                print(f'RANSAC in  {t:.4f} sec')
                t=time()  
                
                # By default colmap does not generate a reconstruction if less than 10 images are registered. Lower it to 3.
                mapper_options = pycolmap.IncrementalMapperOptions()
                mapper_options.num_threads = num_threads                             
                mapper_options.max_num_models = max_num_models                        
                mapper_options.ba_local_num_images = ba_local_num_images              
                mapper_options.init_num_trials = init_num_trials                     
                mapper_options.min_num_matches = min_num_matches                     
                mapper_options.min_model_size = min_model_size                  
                mapper_options.ba_global_images_freq = ba_global_images_freq           
                mapper_options.ba_global_points_freq = ba_global_points_freq           
                mapper_options.ba_global_images_ratio = ba_global_images_ratio 
                mapper_options.ba_global_points_ratio = ba_global_points_ratio 
                os.makedirs(output_path, exist_ok=True)
                maps = pycolmap.incremental_mapping(database_path=database_path, image_path=img_dir, output_path=output_path, options=mapper_options)
                print(maps)
                
                # # Update output_path to the correct directory
                output_path = f'{feature_dir}/colmap_rec/0'
                
                # # Increase column width to avoid cutting off the logs
                # pd.set_option('display.max_colwidth', None)
                
                # Initialize the Reconstruction object
                reconstruction = pycolmap.Reconstruction()
                reconstruction.read(output_path)
                
                # Initialize error container and a dictionary for images corresponding to each Point3DId
                point_errors = defaultdict(list)
                point_images = defaultdict(set)  # we use a set to avoid duplicate image names
                
                for image_id, image in reconstruction.images.items():
                    camera = reconstruction.cameras[image.camera_id]
                    params = camera.params
                    K = np.array([[params[0], 0, params[2]], [0, params[0], params[3]], [0, 0, 1]])
                    
                    # Convert quaternion to rotation matrix
                    r = R.from_quat([image.qvec[1], image.qvec[2], image.qvec[3], image.qvec[0]])
                    rotation_matrix = r.as_matrix()
                    for point2D_id, point2D in enumerate(image.points2D):
                        if point2D.point3D_id != -1 and point2D.point3D_id in reconstruction.points3D:
                            point3D = reconstruction.points3D[point2D.point3D_id]         
                             
                            # Transform 3D point from world coordinates to camera coordinates
                            point3D_cam = np.dot(np.linalg.inv(rotation_matrix), point3D.xyz - image.tvec) 
                                   
                            # Check if the point is in front of the camera
                            if point3D_cam[2] <= 0:
                                continue        
                            
                            # Project the 3D point onto the image plane
                            point2D_proj = np.dot(K, point3D_cam)
                            point2D_proj = point2D_proj / point2D_proj[2]      
                                     
                            # Compute the error
                            error = np.linalg.norm(point2D_proj[0:2] - point2D.xy)        
                            
                            # Save the error for each 3D point and the corresponding image name
                            point_errors[point2D.point3D_id].append(error)
                            point_images[point2D.point3D_id].add(image.name)  # add the image name to the set of images for this Point3DId
                            
                squared_errors = {point_id: np.sum(np.square(errs)) for point_id, errs in point_errors.items()}
                rmse = {point_id: np.sqrt(sum_err / len(point_errors[point_id])) for point_id, sum_err in squared_errors.items()}

                df_rmse = pd.DataFrame([
                    {
                        'Point3DId': point_id,
                        'rpj_RMSE': rmse_value,
                        'Images': list(point_images[point_id]),
                        'ImageCount': len(point_images[point_id])
                    }
                    for point_id, rmse_value in rmse.items()
                ])

                # print(df_rmse)

                # Apply filters on the DataFrame
                if ls_img_name and da_img_name:
                    # Apply filters on the DataFrame if ls_img_name and da_img_name are present
                    df_errors_filtered =df_rmse[df_rmse['Images'].apply(
                        lambda images: sum(ls_img_name in image for image in images) >= 1 and sum(da_img_name in image for image in images) >= 1)
                    ]
                else:
                    # Skip filtering if ls_img_name and da_img_name are missing, and use the original DataFrame
                    df_errors_filtered = df_rmse.copy()
                    
                df_errors_filtered_sorted = df_errors_filtered.sort_values(by='rpj_RMSE')
                
                # print(df_errors_filtered_sorted)
                
                # Filter only those with ImageCount >= 3
                df_errors_filtered_sorted_count = df_errors_filtered_sorted[df_errors_filtered_sorted['ImageCount'] >= 7]
                
                # Take the top N
                df_errors_filtered_sorted_top_N = df_errors_filtered_sorted_count.head(int(sorted_count))
                # print(df_errors_filtered_sorted_top_N)
                # print(len(df_errors_filtered_sorted_top_N))
                
                df_with_reset_index = df_errors_filtered_sorted_top_N.reset_index()
                new_index = ['point_' + str(i) for i in range(len(df_with_reset_index))]
                df_with_reset_index.index = new_index
                df_with_reset_index.drop('index', axis=1, inplace=True)
                print(df_with_reset_index)
                print(len(df_with_reset_index))
            
                
                if ENSEMBLE != True:
                    keypoint_h5_dir = f'{feature_dir}/{value[0]}_{key}_keypoints.h5'
                    matches_h5_dir = f'{feature_dir}/{value[0]}_{key}_matches.h5'
                    # keypoint_h5_dir = '/data/output_sp_sg_retri/keypoints.h5'
                    # matches_h5_dir = '/data/output_sp_sg_retri/matches.h5'
                else:
                    keypoint_h5_dir = f'{feature_dir}/ensemble_keypoints.h5'
                    matches_h5_dir = f'{feature_dir}/ensemble_matches.h5'
                    
                generated_files.append(keypoint_h5_dir)
                # generated_files.append(matches_h5_dir) 
                
                # Load existing keypoints and matches
                with h5py.File(keypoint_h5_dir, 'r') as f:
                    keypoints = {k: v[:] for k, v in f.items()}
                print("Loaded keypoints: ", len(keypoints))
                
                with h5py.File(matches_h5_dir, 'r') as f:
                    matches = {k: {kk: vv[:] for kk, vv in v.items()} for k, v in f.items()}
                print("Loaded matches: ", len(matches))
                
                # Initialize a new dictionary to store the matches with low errors
                matches_filtered = {}
                # Get the 2D indices of the selected 3D points in each image
                selected_indices = defaultdict(list)
                for point_id in df_with_reset_index['Point3DId']:
                    for image_id, image in reconstruction.images.items():
                        for i, point2D in enumerate(image.points2D):
                            if point2D.point3D_id == point_id:
                                selected_indices[image.name].append(i)
                
                for _, marker in tqdm(df_with_reset_index.iterrows(), total=df_with_reset_index.shape[0]):
                    # Get the images for the current marker
                    images = marker['Images']
                    # Iterate through the images for the current marker
                    for img1 in images:
                        # If the current image has matches
                        if img1 in matches:
                            # Initialize a new dictionary to store the matches for the current image
                            matches_filtered[img1] = {}
                            # Iterate through the matches for the current image
                            for img2, match in matches[img1].items():
                                # If the other image is also in the images for the current marker
                                if img2 in images:
                                    # Filter the match to only include the selected 2D points
                                    matches_filtered[img1][img2] = match[np.isin(match[:, 0], selected_indices[img1]) & 
                                                                        np.isin(match[:, 1], selected_indices[img2])]
                    
                # Save the matches_filtered dictionary to a new .h5 file
                with h5py.File(f'{feature_dir}/{value[0]}_{key}_markers_n5.h5', 'w') as f:
                    generated_files.append(f'{feature_dir}/{value[0]}_{key}_markers_n5.h5')
                    for img1, img1_matches in matches_filtered.items():
                        group = f.create_group(img1)
                        for img2, match in img1_matches.items():
                            group.create_dataset(img2, data=match)
                
                image_name_to_id = {image.name: image_id for image_id, image in reconstruction.images.items()}
                create_custom_csv(df_with_reset_index, keypoints, reconstruction, '/data/rc_format_markers/factory134_1280_LoFTR_markers_n5.csv')
                
                gc.collect()
            except Exception as e:
                print(f"An error occurred during the second loop: {e}")
                traceback.print_exc()
    return generated_files, df_with_reset_index, keypoints         
                
if __name__ == "__main__":
    src = '/data/refined_images'
    featureout = '/data/featureout'
    features_resolutions = {
    'LoFTR': [1280]
    }
    checkpoint_path='/data/weights/convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384.pth'
    orinet_path = '/data/weights/OriNet.pth'
    keynet_path = '/data/weights/keynet_pytorch.pth'
    affnet_path = '/data/weights/AffNet.pth'
    hardnet_path = '/data/weights/HardNet.pth'
    sosnet_path = '/data/weights/sosnet_32x32_liberty.pth'
    disk_path = '/data/weights/epipolar-save.pth'
    loftr_path = '/data/weights/outdoor_ds.ckpt'
       
    num_feats = 40000
    os.makedirs(featureout, exist_ok=True)
    main_function(src, featureout, features_resolutions, checkpoint_path, orinet_path, keynet_path, affnet_path, hardnet_path,
                  sosnet_path, disk_path, loftr_path, num_feats = 8000)

################################################################################################################################################  
#     data_dict = {}
#     # Replace this with your image directory
#     img_dir = f'{src}'
#     # List of valid image file extensions
#     img_exts = ['.JPG', '.jpg', '.jpeg', '.png', '.bmp', '.tiff']
#     # Get all files in the directory with valid image extensions
#     img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) and os.path.splitext(f)[1].lower() in img_exts]
#     data_dict[dataset] = {}
#     data_dict[dataset][scene] = img_files
#     for dataset in data_dict:
#         for scene in data_dict[dataset]:
#             print(f'{dataset} / {scene} -> {len(data_dict[dataset][scene])} images')
#     out_results = {}
#     timings = {"shortlisting":[],
#             "feature_detection": [],
#             "feature_matching":[],
#             "RANSAC": [],
#             "Reconstruction": []}
#     gc.collect()
#     datasets = []
#     for dataset in data_dict:
#         datasets.append(dataset)
#     start_time = time()
#     for dataset in datasets:
#         print(dataset)
#         if dataset not in out_results:
#             out_results[dataset] = {}
#         for scene in data_dict[dataset]:
#             print(scene)
#             img_dir = f'{src}'
#             if not os.path.exists(img_dir):
#                 continue
#             try:
#                 out_results[dataset][scene] = {}
#                 img_fnames = [f'{src}/{x}' for x in data_dict[dataset][scene]]
#                 print (f"Got {len(img_fnames)} images")

#                 """only extracted markers.h5"""
#                 feature_dir = "/data/featureout/auto_markers/1280_LoFTR"
#                 # output_path = f'{feature_dir}/models/0'
#                 # # Update output_path to the correct directory
#                 output_path = f'{feature_dir}/colmap_rec/0'
                
#                 # # Increase column width to avoid cutting off the logs
#                 # pd.set_option('display.max_colwidth', None)
                
#                 # Initialize the Reconstruction object
#                 reconstruction = pycolmap.Reconstruction()
#                 reconstruction.read(output_path)
                
#                 # Initialize error container and a dictionary for images corresponding to each Point3DId
#                 point_errors = defaultdict(list)
#                 point_images = defaultdict(set)  # we use a set to avoid duplicate image names
#                 # point_image_errors = defaultdict(dict)
                
#                 for image_id, image in reconstruction.images.items():
#                     camera = reconstruction.cameras[image.camera_id]
#                     params = camera.params
#                     # K = np.array([[params[0], 0, params[2]], [0, params[0], params[3]], [0, 0, 1]])
#                     K = np.array([[params[0], 0, params[1]], [0, params[0], params[2]], [0, 0, 1]])
                    
#                     # Convert quaternion to rotation matrix
#                     r = R.from_quat([image.qvec[1], image.qvec[2], image.qvec[3], image.qvec[0]])
#                     # r = R.from_quat([image.qvec[0], image.qvec[1], image.qvec[2], image.qvec[3]])
#                     rotation_matrix = r.as_matrix()
                    
#                     for point2D_id, point2D in enumerate(image.points2D):
#                         if point2D.point3D_id != -1 and point2D.point3D_id in reconstruction.points3D:
#                             point3D = reconstruction.points3D[point2D.point3D_id]         
                             
#                             # Transform 3D point from world coordinates to camera coordinates
#                             # point3D_cam = np.dot(np.linalg.inv(rotation_matrix), point3D.xyz - image.tvec) 
#                             point3D_cam = np.dot(rotation_matrix, point3D.xyz) + image.tvec
                                   
#                             # Check if the point is in front of the camera
#                             if point3D_cam[2] <= 0:
#                                 continue        
                            
#                             # Project the 3D point onto the image plane
#                             point2D_proj = np.dot(K, point3D_cam)
#                             point2D_proj = point2D_proj / point2D_proj[2]      
                                     
#                             # Compute the error and update point_errors and point_images
#                             error = np.linalg.norm(point2D_proj[0:2] - point2D.xy)        
#                             point_errors[point2D.point3D_id].append(error)
#                             point_images[point2D.point3D_id].add(image.name)

#                             # Additionally, save the error for each 3D point and corresponding image
#                             # point_image_errors[point2D.point3D_id][image.name] = error * 0.0001
                            
#                 squared_errors = {point_id: np.sum(np.square(errs)) for point_id, errs in point_errors.items()}
#                 rmse = {point_id: np.sqrt(sum_err / len(point_errors[point_id])) for point_id, sum_err in squared_errors.items()}

#                 df_rmse = pd.DataFrame([
#                     {
#                         'Point3DId': point_id,
#                         'rpj_RMSE': rmse_value,
#                         'Images': list(point_images[point_id]),
#                         'ImageCount': len(point_images[point_id])
#                     }
#                     for point_id, rmse_value in rmse.items()
#                 ])

#                 # print(df_rmse)

#                 # Apply filters on the DataFrame
#                 if ls_img_name and da_img_name:
#                     # Apply filters on the DataFrame if ls_img_name and da_img_name are present
#                     df_errors_filtered =df_rmse[df_rmse['Images'].apply(
#                         lambda images: sum(ls_img_name in image for image in images) >= 1 and sum(da_img_name in image for image in images) >= 1)
#                     ]
#                 else:
#                     # Skip filtering if ls_img_name and da_img_name are missing, and use the original DataFrame
#                     df_errors_filtered = df_rmse.copy()
                    
#                 df_errors_filtered_sorted = df_errors_filtered.sort_values(by='rpj_RMSE')
                
#                 # print(df_errors_filtered_sorted)
                
#                 # Filter only those with ImageCount >= 3
#                 df_errors_filtered_sorted_count = df_errors_filtered_sorted[df_errors_filtered_sorted['ImageCount'] >= 5]
                
#                 # Take the top N
#                 df_errors_filtered_sorted_top_N = df_errors_filtered_sorted_count.head(int(sorted_count))
#                 # print(df_errors_filtered_sorted_top_N)
#                 # print(len(df_errors_filtered_sorted_top_N))
                
#                 df_with_reset_index = df_errors_filtered_sorted_top_N.reset_index()
#                 new_index = ['point_' + str(i) for i in range(len(df_with_reset_index))]
#                 df_with_reset_index.index = new_index
#                 df_with_reset_index.drop('index', axis=1, inplace=True)
#                 print(df_with_reset_index)
#                 print(len(df_with_reset_index))
            
                
#                 if ENSEMBLE != True:
#                     keypoint_h5_dir = f'{feature_dir}/{value[0]}_{key}_keypoints.h5'
#                     matches_h5_dir = f'{feature_dir}/{value[0]}_{key}_matches.h5'
#                     # keypoint_h5_dir = '/data/output_sp_sg_retri/keypoints.h5'
#                     # matches_h5_dir = '/data/output_sp_sg_retri/matches.h5'
#                 else:
#                     keypoint_h5_dir = f'{feature_dir}/ensemble_keypoints.h5'
#                     matches_h5_dir = f'{feature_dir}/ensemble_matches.h5'
                    
#                 generated_files.append(keypoint_h5_dir)
#                 # generated_files.append(matches_h5_dir) 
                
#                 # Load existing keypoints and matches
#                 with h5py.File(keypoint_h5_dir, 'r') as f:
#                     keypoints = {k: v[:] for k, v in f.items()}
#                 print("Loaded keypoints: ", len(keypoints))
                
#                 with h5py.File(matches_h5_dir, 'r') as f:
#                     matches = {k: {kk: vv[:] for kk, vv in v.items()} for k, v in f.items()}
#                 print("Loaded matches: ", len(matches))
                
#                 # Initialize a new dictionary to store the matches with low errors
#                 matches_filtered = {}
#                 # Get the 2D indices of the selected 3D points in each image
#                 selected_indices = defaultdict(list)
#                 for point_id in df_with_reset_index['Point3DId']:
#                     for image_id, image in reconstruction.images.items():
#                         for i, point2D in enumerate(image.points2D):
#                             if point2D.point3D_id == point_id:
#                                 selected_indices[image.name].append(i)
                
#                 for _, marker in tqdm(df_with_reset_index.iterrows(), total=df_with_reset_index.shape[0]):
#                     # Get the images for the current marker
#                     images = marker['Images']
#                     # Iterate through the images for the current marker
#                     for img1 in images:
#                         # If the current image has matches
#                         if img1 in matches:
#                             # Initialize a new dictionary to store the matches for the current image
#                             matches_filtered[img1] = {}
#                             # Iterate through the matches for the current image
#                             for img2, match in matches[img1].items():
#                                 # If the other image is also in the images for the current marker
#                                 if img2 in images:
#                                     # Filter the match to only include the selected 2D points
#                                     matches_filtered[img1][img2] = match[np.isin(match[:, 0], selected_indices[img1]) & 
#                                                                         np.isin(match[:, 1], selected_indices[img2])]
                    
#                 # Save the matches_filtered dictionary to a new .h5 file
#                 with h5py.File(f'{feature_dir}/{value[0]}_{key}_markers_n5.h5', 'w') as f:
#                     generated_files.append(f'{feature_dir}/{value[0]}_{key}_markers_n5.h5')
#                     for img1, img1_matches in matches_filtered.items():
#                         group = f.create_group(img1)
#                         for img2, match in img1_matches.items():
#                             group.create_dataset(img2, data=match)
                
#                 image_name_to_id = {image.name: image_id for image_id, image in reconstruction.images.items()}
#                 create_custom_csv(df_with_reset_index, keypoints, reconstruction, '/data/rc_format_markers/factory134_1280_LoFTR_markers_n5.csv')
                
#                 gc.collect()
#             except Exception as e:
#                 print(f"An error occurred during the second loop: {e}")
#                 traceback.print_exc()
#     return generated_files, df_with_reset_index, keypoints         
                
# if __name__ == "__main__":
#     src = '/data/refined_images'
#     featureout = '/data/featureout'
#     features_resolutions = {
#     'LoFTR': [1280]
#     }
#     checkpoint_path='/data/weights/convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384.pth'
#     orinet_path = '/data/weights/OriNet.pth'
#     keynet_path = '/data/weights/keynet_pytorch.pth'
#     affnet_path = '/data/weights/AffNet.pth'
#     hardnet_path = '/data/weights/HardNet.pth'
#     sosnet_path = '/data/weights/sosnet_32x32_liberty.pth'
#     disk_path = '/data/weights/epipolar-save.pth'
#     loftr_path = '/data/weights/outdoor_ds.ckpt'
       
#     num_feats = 40000
#     os.makedirs(featureout, exist_ok=True)
#     main_function(src, featureout, features_resolutions, checkpoint_path, orinet_path, keynet_path, affnet_path, hardnet_path,
#                   sosnet_path, disk_path, loftr_path, num_feats = 8000)
    
    # create_custom_csv(df_with_reset_index, keypoints, '/data/rc_format_markers/factory134_1280_DISK_markers_n2.csv')
    

