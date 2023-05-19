# General utilities
import os
from tqdm import tqdm
from time import time
from fastprogress import progress_bar
import gc
import numpy as np
import h5py
from IPython.display import clear_output
from collections import defaultdict
from copy import deepcopy

# CV/ML
import cv2
import torch
import torch.nn.functional as F
import kornia as K
import kornia.feature as KF
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import config
import traceback

import sqlite3
import numpy as np

# 3D reconstruction
import pycolmap

print('Kornia version', K.__version__)
print('Pycolmap version', pycolmap.__version__)

###################################################
# Config

device=torch.device(config.DEVICE)
local_features = config.local_features

if len(local_features) > 1:
    ENSEMBLE = True
    keypoints_h5 = "ensemble_keypoint.h5"
else:
    ENSEMBLE = False
    if local_features[0] == "DISK":
        keypoints_h5 = "disk_keypoints.h5"
    elif local_features[0] == "KeyNetAffNetHardNet":
        keypoints_h5 = "keynet_keypoints.h5"
    elif local_features[0] == "LoFTR":
        keypoints_h5 = "loftr_keypoints.h5"
    else:
        raise print("I don't know!!!")

sim_th = config.sim_th
min_pairs = config.min_pairs
exhaustive_if_less = config.exhaustive_if_less

src = config.src
src2 = config.src2
src3 = config.src3
featureout = config.featureout

feature_dir = config.feature_dir

num_feats = config.num_feats
max_length = config.max_length
matching_alg = config.matching_alg
min_matches = config.min_matches

database_path = config.database_path

table_submission = config.table_submission
output_submission = config.output_submission

MAX_IMAGE_ID = config.MAX_IMAGE_ID
IS_PYTHON3 = config.IS_PYTHON3

# keypoints_h5 = config.keypoints_h5
# matches_h5 = config.matches_h5

OriNet = config.OriNet
KeyNet = config.KeyNet
AffNet = config.AffNet
HardNet = config.HardNet

DISK = config.DISK

DISK_lafs = config.DISK_lafs
DISK_keypoints = config.DISK_keypoints
DISK_descriptors = config.DISK_descriptors
DISK_matches = config.DISK_matches

KeyNet_lafs = config.KeyNet_lafs
KeyNet_keypoints = config.KeyNet_keypoints
KeyNet_descriptors = config.KeyNet_descriptors
KeyNet_matches = config.KeyNet_matches


LoFTR = config.LoFTR
LoFTR_matches = config.LoFTR_matches
LoFTR_keypoints = config.LoFTR_keypoints

colmap_db = config.colmap_db
###################################################

# Can be LoFTR, KeyNetAffNetHardNet, or DISK

def arr_to_str(a):
    return ';'.join([str(x) for x in a.reshape(-1)])


def load_torch_image(fname, device=torch.device('cpu')):
    img = K.image_to_tensor(cv2.imread(fname), False).float() / 255.
    img = K.color.bgr_to_rgb(img.to(device))
    return img

# We will use ViT global descriptor to get matching shortlists.
def get_global_desc(fnames, model,
                    device =  torch.device('cuda')):
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

    # model = timm.create_model('tf_efficientnet_b7',
    #                           config.checkpoint_path)
    model = timm.create_model('tf_efficientnet_b7', pretrained=False)
    model.load_state_dict(torch.load(config.checkpoint_path))
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

# Code to manipulate a colmap database.
# Forked from https://github.com/colmap/colmap/blob/dev/scripts/python/database.py

# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

# This script is based on an original implementation by True Price.

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
        
# Code to interface DISK with Colmap.
# Forked from https://github.com/cvlab-epfl/disk/blob/37f1f7e971cea3055bb5ccfc4cf28bfd643fa339/colmap/h5_to_db.py

#  Copyright [2020] [Michał Tyszkiewicz, Pascal Fua, Eduard Trulls]
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import os, argparse, h5py, warnings
import numpy as np
from tqdm import tqdm
from PIL import Image, ExifTags


def get_focal(image_path, err_on_default=False):
    image         = Image.open(image_path)
    max_size      = max(image.size)

    exif = image.getexif()
    focal = None
    if exif is not None:
        focal_35mm = None
        # https://github.com/colmap/colmap/blob/d3a29e203ab69e91eda938d6e56e1c7339d62a99/src/util/bitmap.cc#L299
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


def add_keypoints(db, h5_path, keypoints_h5, image_path, img_ext, camera_model, single_camera = True):
    keypoint_f = h5py.File(os.path.join(h5_path, keypoints_h5), 'r')

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

def add_matches(db, h5_path, fname_to_id):
    match_file = h5py.File(os.path.join(h5_path, matches_h5), 'r')
    
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
                
# Making kornia local features loading w/o internet
class KeyNetAffNetHardNet(KF.LocalFeature):
    """Convenience module, which implements KeyNet detector + AffNet + HardNet descriptor.

    .. image:: _static/img/keynet_affnet.jpg
    """

    def __init__(
        self,
        num_features: int = 5000,
        upright: bool = False,
        device = torch.device('cpu'),
        scale_laf: float = 1.0,
    ):
        ori_module = KF.PassLAF() if upright else KF.LAFOrienter(angle_detector=KF.OriNet(False)).eval()
        if not upright:
            weights = torch.load(OriNet)['state_dict']
            ori_module.angle_detector.load_state_dict(weights)
        detector = KF.KeyNetDetector(
            False, num_features=num_features, ori_module=ori_module, aff_module=KF.LAFAffNetShapeEstimator(False).eval()
        ).to(device)
        kn_weights = torch.load(KeyNet)['state_dict']
        detector.model.load_state_dict(kn_weights)
        affnet_weights = torch.load(AffNet)['state_dict']
        detector.aff.load_state_dict(affnet_weights)
        
        hardnet = KF.HardNet(False).eval()
        hn_weights = torch.load(HardNet)['state_dict']
        hardnet.load_state_dict(hn_weights)
        descriptor = KF.LAFDescriptor(hardnet, patch_size=32, grayscale_descriptor=True).to(device)
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
        
# def detect_features(img_fnames,
#                     num_feats = 8192,
#                     upright = False,
#                     device=torch.device('cpu'),
#                     feature_dir = '.featureout',
#                     max_length = 840,
#                     local_features= ['DISK']):
#     for local_feature in local_features:    
#         if local_feature == 'DISK':
#             # Load DISK from Kaggle models so it can run when the notebook is offline.
#             disk = KF.DISK().to(device)
#             pretrained_dict = torch.load(DISK, map_location=device)
#             disk.load_state_dict(pretrained_dict['extractor'])
#             # disk = KF.DISK.from_pretrained('depth').to(device)
#             disk.eval()
#         elif local_feature == 'KeyNetAffNetHardNet':
#             feature = KeyNetAffNetHardNet(num_feats, upright, device).to(device).eval()
#         else:
#             return
#         if not os.path.isdir(feature_dir):
#             os.makedirs(feature_dir)
#         with h5py.File(f'{feature_dir}/{config.lafs}', mode='w') as f_laf, \
#             h5py.File(f'{feature_dir}/{config.keypoints}', mode='w') as f_kp, \
#             h5py.File(f'{feature_dir}/{config.descriptors}', mode='w') as f_desc:
#             for img_path in progress_bar(img_fnames):
#                 img_fname = img_path.split('/')[-1]
#                 key = img_fname
#                 with torch.inference_mode():
#                     timg = load_torch_image(img_path, device=device)
#                     H, W = timg.shape[2:]
#                     resize_to = calculate_new_size((W, H), max_length)
#                     timg_resized = K.geometry.resize(timg, resize_to, antialias=True)
#                     print(f'Resized {timg.shape} to {timg_resized.shape} (max_length={max_length})')
#                     h, w = timg_resized.shape[2:]
#                     if local_feature == 'DISK':
#                         features = disk(timg_resized, num_feats, pad_if_not_divisible=True)[0]
#                         kps1, descs = features.keypoints, features.descriptors
                        
#                         lafs = KF.laf_from_center_scale_ori(kps1[None], torch.ones(1, len(kps1), 1, 1, device=device))
#                     if local_feature == 'KeyNetAffNetHardNet':
#                         lafs, resps, descs = feature(K.color.rgb_to_grayscale(timg_resized))
#                     lafs[:,:,0,:] *= float(W) / float(w)
#                     lafs[:,:,1,:] *= float(H) / float(h)
#                     desc_dim = descs.shape[-1]
#                     kpts = KF.get_laf_center(lafs).reshape(-1, 2).detach().cpu().numpy()
#                     descs = descs.reshape(-1, desc_dim).detach().cpu().numpy()
#                     f_laf[key] = lafs.detach().cpu().numpy()
#                     f_kp[key] = kpts
#                     f_desc[key] = descs
#     return

def detect_features(img_fnames,
                    num_feats = 8192,
                    upright = False,
                    device=torch.device('cpu'),
                    feature_dir = '.featureout',
                    max_length = 840,
                    local_features= ['LoFTR']):
    for local_feature in local_features:    
        if local_feature == 'DISK':
            # Load DISK from Kaggle models so it can run when the notebook is offline.
            disk = KF.DISK().to(device)
            pretrained_dict = torch.load(DISK, map_location=device)
            disk.load_state_dict(pretrained_dict['extractor'])
            # disk = KF.DISK.from_pretrained('depth').to(device)
            disk.eval()
        elif local_feature == 'KeyNetAffNetHardNet':
            feature = KeyNetAffNetHardNet(num_feats, upright, device).to(device).eval()
        else:
            return
        
        if not os.path.isdir(feature_dir):
            os.makedirs(feature_dir)
        
        if local_feature == 'DISK':
            with h5py.File(f'{feature_dir}/{DISK_lafs}', mode='w') as f_laf, \
                h5py.File(f'{feature_dir}/{DISK_keypoints}', mode='w') as f_kp, \
                h5py.File(f'{feature_dir}/{DISK_descriptors}', mode='w') as f_desc:
                for img_path in progress_bar(img_fnames):
                    img_fname = img_path.split('/')[-1]
                    key = img_fname
                    with torch.inference_mode():
                        timg = load_torch_image(img_path, device=device)
                        H, W = timg.shape[2:]
                        resize_to = calculate_new_size((H, W), max_length)
                        timg_resized = K.geometry.resize(timg, resize_to, antialias=True)
                        print(f'Resized {timg.shape} to {timg_resized.shape} (max_length={max_length})')
                        h, w = timg_resized.shape[2:]
                        
                        features = disk(timg_resized, num_feats, pad_if_not_divisible=True)[0]
                        kps1, descs = features.keypoints, features.descriptors
                        
                        lafs = KF.laf_from_center_scale_ori(kps1[None], torch.ones(1, len(kps1), 1, 1, device=device))
                        lafs[:,:,0,:] *= float(W) / float(w)
                        lafs[:,:,1,:] *= float(H) / float(h)
                        desc_dim = descs.shape[-1]
                        kpts = KF.get_laf_center(lafs).reshape(-1, 2).detach().cpu().numpy()
                        descs = descs.reshape(-1, desc_dim).detach().cpu().numpy()
                        f_laf[key] = lafs.detach().cpu().numpy()
                        f_kp[key] = kpts
                        f_desc[key] = descs
                        
        elif local_feature == 'KeyNetAffNetHardNet':
            with h5py.File(f'{feature_dir}/keynet_lafs.h5', mode='w') as f_laf, \
                h5py.File(f'{feature_dir}/{KeyNet_keypoints}', mode='w') as f_kp, \
                h5py.File(f'{feature_dir}/{KeyNet_descriptors}', mode='w') as f_desc:
                for img_path in progress_bar(img_fnames):
                    img_fname = img_path.split('/')[-1]
                    key = img_fname
                    with torch.inference_mode():
                        timg = load_torch_image(img_path, device=device)
                        H, W = timg.shape[2:]
                        resize_to = calculate_new_size((H, W), max_length)
                        timg_resized = K.geometry.resize(timg, resize_to, antialias=True)
                        print(f'Resized {timg.shape} to {timg_resized.shape} (max_length={max_length})')
                        h, w = timg_resized.shape[2:]
                        
                        lafs, resps, descs = feature(K.color.rgb_to_grayscale(timg_resized))
                        
                        lafs[:,:,0,:] *= float(W) / float(w)
                        lafs[:,:,1,:] *= float(H) / float(h)
                        desc_dim = descs.shape[-1]
                        kpts = KF.get_laf_center(lafs).reshape(-1, 2).detach().cpu().numpy()
                        descs = descs.reshape(-1, desc_dim).detach().cpu().numpy()
                        f_laf[key] = lafs.detach().cpu().numpy()
                        f_kp[key] = kpts
                        f_desc[key] = descs
        else:
            print("select model!!")
    return

def detect_features_ensemble(img_fnames,
                    num_feats = config.num_feats,
                    upright = False,
                    device=torch.device(config.DEVICE),
                    feature_dir = config.feature_dir,
                    max_length = config.max_length):
    if config.LOCAL_FEATURE == ['LoFTR, DISK']:
        # Load DISK from Kaggle models so it can run when the notebook is offline.
        disk = KF.DISK().to(device)
        pretrained_dict = torch.load(config.DISK, map_location=device)
        disk.load_state_dict(pretrained_dict['extractor'])
        # disk = KF.DISK.from_pretrained('depth').to(device)
        disk.eval()
    if config.LOCAL_FEATURE == ['LoFTR','KeyNetAffNetHardNet']:
        feature = KeyNetAffNetHardNet(num_feats, upright, device).to(device).eval()
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)
    with h5py.File(f'{feature_dir}/{config.lafs}', mode='w') as f_laf, \
         h5py.File(f'{feature_dir}/{config.keypoints}', mode='w') as f_kp, \
         h5py.File(f'{feature_dir}/{config.descriptors}', mode='w') as f_desc:
        for img_path in progress_bar(img_fnames):
            img_fname = img_path.split('/')[-1]
            key = img_fname
            with torch.inference_mode():
                timg = load_torch_image(img_path, device=device)
                H, W = timg.shape[2:]
                resize_to = calculate_new_size((H, W), max_length)
                timg_resized = K.geometry.resize(timg, resize_to, antialias=True)
                print(f'Resized {timg.shape} to {timg_resized.shape} (max_length={max_length})')
                h, w = timg_resized.shape[2:]
                if config.LOCAL_FEATURE == ['LoFTR, DISK']:
                    features = disk(timg_resized, num_feats, pad_if_not_divisible=True)[0]
                    kps1, descs = features.keypoints, features.descriptors
                    
                    lafs = KF.laf_from_center_scale_ori(kps1[None], torch.ones(1, len(kps1), 1, 1, device=device))
                if config.LOCAL_FEATURE == ['LoFTR','KeyNetAffNetHardNet']:
                    lafs, resps, descs = feature(K.color.rgb_to_grayscale(timg_resized))
                lafs[:,:,0,:] *= float(W) / float(w)
                lafs[:,:,1,:] *= float(H) / float(h)
                desc_dim = descs.shape[-1]
                kpts = KF.get_laf_center(lafs).reshape(-1, 2).detach().cpu().numpy()
                descs = descs.reshape(-1, desc_dim).detach().cpu().numpy()
                f_laf[key] = lafs.detach().cpu().numpy()
                f_kp[key] = kpts
                f_desc[key] = descs
    return

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
                   min_matches=15, 
                   force_mutual = True,
                   matching_alg="smnn",
                   local_feature='DISK'
                  ):
    if local_feature == 'DISK':
        assert matching_alg in ['smnn', 'adalam']
        with h5py.File(f'{feature_dir}/{DISK_lafs}', mode='r') as f_laf, \
            h5py.File(f'{feature_dir}/{DISK_descriptors}', mode='r') as f_desc, \
            h5py.File(f'{feature_dir}/{DISK_matches}', mode='w') as f_match:

            for pair_idx in progress_bar(index_pairs):
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
                    #adalam_config['orientation_difference_threshold'] = None
                    #adalam_config['scale_rate_threshold'] = None
                    adalam_config['force_seed_mnn']= False
                    adalam_config['search_expansion'] = 16
                    adalam_config['ransac_iters'] = 128
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
                            
    elif local_feature == 'KeyNetAffNetHardNet':
        assert matching_alg in ['smnn', 'adalam']
        with h5py.File(f'{feature_dir}/{KeyNet_lafs}', mode='r') as f_laf, \
            h5py.File(f'{feature_dir}/{KeyNet_descriptors}', mode='r') as f_desc, \
            h5py.File(f'{feature_dir}/{KeyNet_matches}', mode='w') as f_match:

            for pair_idx in progress_bar(index_pairs):
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
                    #adalam_config['orientation_difference_threshold'] = None
                    #adalam_config['scale_rate_threshold'] = None
                    adalam_config['force_seed_mnn']= False
                    adalam_config['search_expansion'] = 16
                    adalam_config['ransac_iters'] = 128
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
                                
        return

def match_loftr(img_fnames,
                index_pairs,
                feature_dir=".featureout",
                device=torch.device('cpu'),
                min_matches=15, max_length=1200):
    matcher = KF.LoFTR(pretrained=None)
    matcher.load_state_dict(torch.load(LoFTR)['state_dict'])
    matcher = matcher.to(device).eval()

    # First we do pairwise matching, and then extract "keypoints" from loftr matches.
    with h5py.File(f'{feature_dir}/{LoFTR_matches}', mode='w') as f_match:
        for pair_idx in progress_bar(index_pairs):
            idx1, idx2 = pair_idx
            fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
            key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
            
            # Load img1
            timg1 = K.color.rgb_to_grayscale(load_torch_image(fname1, device=device))
            H1, W1 = timg1.shape[2:]
            resize_to1 = calculate_new_size((H1, W1), max_length)
            timg_resized1 = K.geometry.resize(timg1, resize_to1, antialias=True)
            h1, w1 = timg_resized1.shape[2:]
            # Load img2
            timg2 = K.color.rgb_to_grayscale(load_torch_image(fname2, device=device))
            H2, W2 = timg2.shape[2:]
            resize_to2 = calculate_new_size((H2, W2), max_length)
            timg_resized2 = K.geometry.resize(timg2, resize_to2, antialias=True)
            h2, w2 = timg_resized2.shape[2:]
            with torch.inference_mode():
                input_dict = {"image0": timg_resized1,"image1": timg_resized2}
                correspondences = matcher(input_dict)
            mkpts0 = correspondences['keypoints0'].cpu().numpy()
            mkpts1 = correspondences['keypoints1'].cpu().numpy()
            
            # Apply MAGSAC
            Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 1, 0.999, 100000)
            inliers = inliers.ravel() > 0
            mkpts0 = mkpts0[inliers]
            mkpts1 = mkpts1[inliers]

            mkpts0[:,0] *= float(W1) / float(w1)
            mkpts0[:,1] *= float(H1) / float(h1)

            mkpts1[:,0] *= float(W2) / float(w2)
            mkpts1[:,1] *= float(H2) / float(h2)

            n_matches = len(mkpts1)
            group  = f_match.require_group(key1)
            if n_matches >= min_matches:
                 group.create_dataset(key2, data=np.concatenate([mkpts0, mkpts1], axis=1))

    # Let's find unique loftr pixels and group them together.
    kpts = defaultdict(list)
    match_indexes = defaultdict(dict)
    total_kpts=defaultdict(int)
    with h5py.File(f'{feature_dir}/{config.LoFTR_matches}', mode='r') as f_match:
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
    with h5py.File(f'{feature_dir}/{LoFTR_keypoints}', mode='w') as f_kp:
        for k, kpts1 in unique_kpts.items():
            f_kp[k] = kpts1
    
    with h5py.File(f'{feature_dir}/{LoFTR_matches}', mode='w') as f_match:
        for k1, gr in out_match.items():
            group  = f_match.require_group(k1)
            for k2, match in gr.items():
                group[k2] = match
    return

# # def ensemble_matches(loftr_keypoints_file, loftr_matches_file, keypoints_file, matches_file, ensemble_output_dir):
# def ensemble_matches(keypoints, matches, ensemble_keypoints='ensemble_keypoints.h5', ensemble_matches='ensemble_matches.h5', ensemble_output_dir):
#     loftr_keypoints = h5py.File(loftr_keypoints_file, 'r')
#     loftr_matches = h5py.File(loftr_matches_file, 'r')
#     keypoints = h5py.File(keypoints_file, 'r')
#     matches = h5py.File(matches_file, 'r')
    
#     # Ensemble keypoints
#     ensemble_keypoints = {}
#     for fname in loftr_keypoints.keys():
#         loftr_kp = np.array(loftr_keypoints[fname])
#         keynet_or_disk_kp = np.array(keypoints[fname])
        
#         print(f"Image file name: {fname}")
#         print(f"LoFTR keypoints shape: {loftr_kp.shape}")
#         print(f"KeyNet or DISK keypoints shape: {keynet_or_disk_kp.shape}")
#         mkpts = [loftr_kp, keynet_or_disk_kp]
#         ensemble_kp = np.concatenate(mkpts, axis=0)
#         ensemble_keypoints[fname] = ensemble_kp
    
#     # Ensemble matches
#     ensemble_matches = {}
#     for fname1 in loftr_matches.keys():
#         ensemble_matches[fname1] = {}
#         for fname2 in loftr_matches[fname1].keys():
#             loftr_match = np.array(loftr_matches[fname1][fname2])
#             keynet_or_disk_match = np.array(matches[fname1][fname2])
#             print(f"Image pair: {fname1}, {fname2}")
#             print(f"LoFTR match shape: {loftr_match.shape}")
#             print(f"KeyNet or DISK match shape: {keynet_or_disk_match.shape}")
            
#             mkpts = [loftr_match, keynet_or_disk_match]
#             ensemble_match = np.concatenate(mkpts, axis=0)
#             ensemble_matches[fname1][fname2] = ensemble_match
#             print(f"Image pair: {fname1}, {fname2}")
#             print(f"Ensemble match shape: {ensemble_match.shape}")
            
#     # Save ensemble results
#     with h5py.File(os.path.join(ensemble_output_dir, ensemble_keypoints), 'w') as keypoints_out:
#         for fname, data in ensemble_keypoints.items():
#             keypoints_out.create_dataset(fname, data=data)

#     with h5py.File(os.path.join(ensemble_output_dir, ensemble_keypoints), 'w') as matches_out:
#         for fname1, data1 in ensemble_matches.items():
#             group = matches_out.create_group(fname1)
#             for fname2, data2 in data1.items():
#                 group.create_dataset(fname2, data=data2)

#     loftr_keypoints.close()
#     loftr_matches.close()
#     keypoints.close()
#     matches.close()


def ensemble_matches(keypoints, matches, ensemble_keypoints='ensemble_keypoints.h5', ensemble_matches='ensemble_matches.h5', 
                     ensemble_output_dir='featureout'):
    merged_keypoints = {}
    for keypoint in keypoints:
        with open(keypoint, 'r') as f:
            data = h5py.File(f, 'r')
            for key, value in data.items():
                if key not in merged_keypoints:
                    # 키가 merged_dict에 없으면 새로운 항목으로 추가
                    merged_keypoints[key] = value
                else:
                    # 키가 이미 있다면, 값을 리스트로 만들어서 관리
                    if not isinstance(merged_keypoints[key], np.array):
                        # 기존 값이 리스트가 아니라면 리스트로 변경
                        merged_keypoints[key] = [merged_keypoints[key]]
                    # 새로운 값을 리스트에 추가
                    merged_keypoints[key].append(value)

    # 합쳐진 dictionary 확인
    print(merged_keypoints)
    
    merged_matches = {}
    for match in matches:
        with open(match, 'r') as f:
            data = h5py.File(f, 'r')
            for key, value in data.items():
                if key not in merged_matches:
                    # 키가 merged_dict에 없으면 새로운 항목으로 추가
                    merged_matches[key] = value
                else:
                    # 키가 이미 있다면, 값을 리스트로 만들어서 관리
                    if not isinstance(merged_matches[key], np.array):
                        # 기존 값이 리스트가 아니라면 리스트로 변경
                        merged_matches[key] = [merged_matches[key]]
                    # 새로운 값을 리스트에 추가
                    merged_matches[key].append(value)

    # 합쳐진 dictionary 확인
    print(merged_matches)
            
    # Save ensemble results
    with h5py.File(os.path.join(ensemble_output_dir, ensemble_keypoints), 'w') as keypoints_out:
        for fname, data in merged_keypoints.items():
            keypoints_out.create_dataset(fname, data=data)

    with h5py.File(os.path.join(ensemble_output_dir, ensemble_matches), 'w') as matches_out:
        for fname1, data1 in merged_matches.items():
            group = matches_out.create_group(fname1)
            for fname2, data2 in data1.items():
                group.create_dataset(fname2, data=data2)


def import_into_colmap(img_dir,
                       feature_dir ='.featureout',
                       database_path = 'colmap.db',
                       img_ext='.jpg'):
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(db, feature_dir, keypoints_h5, img_dir, img_ext, 'simple-radial', single_camera)
    add_matches(
        db,
        feature_dir,
        fname_to_id,
    )

    db.commit()
    return

# Get data from csv.

data_dict = {}
with open(f'{src}/{table_submission}', 'r') as f:
    for i, l in enumerate(f):
        # Skip header.
        if l and i > 0:
            image, dataset, scene, _, _ = l.strip().split(',')
            if dataset not in data_dict:
                data_dict[dataset] = {}
            if scene not in data_dict[dataset]:
                data_dict[dataset][scene] = []
            data_dict[dataset][scene].append(image)
            
for dataset in data_dict:
    for scene in data_dict[dataset]:
        print(f'{dataset} / {scene} -> {len(data_dict[dataset][scene])} images')
        
out_results = {}
timings = {"shortlisting":[],
           "feature_detection": [],
           "feature_matching":[],
           "RANSAC": [],
           "Reconstruction": []}

# Function to create a submission file.
def create_submission(out_results, data_dict, output_submission):    
    with open(output_submission, 'w') as f:
        f.write('image_path,dataset,scene,rotation_matrix,translation_vector\n')
        for dataset in data_dict:
            if dataset in out_results:
                res = out_results[dataset]
            else:
                res = {}
            for scene in data_dict[dataset]:
                if scene in res:
                    scene_res = res[scene]
                else:
                    scene_res = {"R":{}, "t":{}}
                for image in data_dict[dataset][scene]:
                    if image in scene_res:
                        print (image)
                        R = scene_res[image]['R'].reshape(-1)
                        T = scene_res[image]['t'].reshape(-1)
                    else:
                        R = np.eye(3).reshape(-1)
                        T = np.zeros((3))
                    f.write(f'{image},{dataset},{scene},{arr_to_str(R)},{arr_to_str(T)}\n')
                
gc.collect()
datasets = []
for dataset in data_dict:
    datasets.append(dataset)

for dataset in datasets:
    print(dataset)
    if dataset not in out_results:
        out_results[dataset] = {}
    for scene in data_dict[dataset]:
        print(scene)
        # Fail gently if the notebook has not been submitted and the test data is not populated.
        # You may want to run this on the training data in that case?
        img_dir = f'{src}/{src2}/{dataset}/{scene}/{src3}'
        if not os.path.exists(img_dir):
            continue
        # Wrap the meaty part in a try-except block.
        try:
            out_results[dataset][scene] = {}
            img_fnames = [f'{src}/{src2}/{x}' for x in data_dict[dataset][scene]]
            print (f"Got {len(img_fnames)} images")
            feature_dir = f'{featureout}/{dataset}_{scene}'
            if not os.path.isdir(feature_dir):
                os.makedirs(feature_dir, exist_ok=True)
            t=time()
            index_pairs = get_image_pairs_shortlist(img_fnames,
                                  sim_th = sim_th, # should be strict
                                  min_pairs = min_pairs, # we select at least min_pairs PER IMAGE with biggest similarity
                                  exhaustive_if_less = exhaustive_if_less,
                                  device=device)
            t=time() -t 
            timings['shortlisting'].append(t)
            print (f'{len(index_pairs)}, pairs to match, {t:.4f} sec')
            gc.collect()
            t=time()
            
            for local_feature in local_features:
                if local_feature != 'LoFTR':
                    detect_features(img_fnames,
                        num_feats = num_feats,
                        upright = False,
                        device=device,
                        feature_dir = feature_dir,
                        max_length = max_length,
                        local_features=local_feature)
                    gc.collect()
                    t=time() -t 
                    timings['feature_detection'].append(t)
                    print(f'Features detected in  {t:.4f} sec')
                    t=time()
                    match_features(img_fnames, index_pairs, feature_dir=feature_dir, local_feature=local_feature, device=device)                
                elif local_features == 'LoFTR': 
                    match_loftr(img_fnames, index_pairs, feature_dir=feature_dir, device=device, max_length=1200)            
                else:
                    print('Redefine the model!!!')
            
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
                                else:
                                    matches.append(file_name)
                            else:
                                print("Bye")
                
                # ensemble_matches(keypoints, matches, ensemble_output_dir=feature_dir)
                ensemble_matches(keypoints, matches, ensemble_keypoints='ensemble_keypoints.h5', ensemble_matches='ensemble_matches.h5', ensemble_output_dir=feature_dir)   
            
            
            t=time() -t 
            timings['feature_matching'].append(t)
            print(f'Features matched in  {t:.4f} sec')
            
            database_path = f'{feature_dir}/{colmap_db}'
            # database_path = f'{feature_dir}/colmap.db'
            print(database_path)
            if os.path.isfile(database_path):
                os.remove(database_path)
            gc.collect()
            import_into_colmap(img_dir, feature_dir=feature_dir,database_path=database_path)
            output_path = f'{feature_dir}/colmap_rec_{local_features}'

            t=time()
            pycolmap.match_exhaustive(database_path)
            t=time() - t 
            timings['RANSAC'].append(t)
            print(f'RANSAC in  {t:.4f} sec')

            t=time()
            # By default colmap does not generate a reconstruction if less than 10 images are registered. Lower it to 3.
            mapper_options = pycolmap.IncrementalMapperOptions()
            mapper_options.min_model_size = 3
            os.makedirs(output_path, exist_ok=True)
            maps = pycolmap.incremental_mapping(database_path=database_path, image_path=img_dir, output_path=output_path, options=mapper_options)
            print(maps)
            #clear_output(wait=False)
            t=time() - t
            timings['Reconstruction'].append(t)
            print(f'Reconstruction done in  {t:.4f} sec')
            imgs_registered  = 0
            best_idx = None
            print ("Looking for the best reconstruction")
            if isinstance(maps, dict):
                for idx1, rec in maps.items():
                    print (idx1, rec.summary())
                    if len(rec.images) > imgs_registered:
                        imgs_registered = len(rec.images)
                        best_idx = idx1
            if best_idx is not None:
                print (maps[best_idx].summary())
                for k, im in maps[best_idx].images.items():
                    key1 = f'{dataset}/{scene}/{src3}/{im.name}'
                    out_results[dataset][scene][key1] = {}
                    out_results[dataset][scene][key1]["R"] = deepcopy(im.rotmat())
                    out_results[dataset][scene][key1]["t"] = deepcopy(np.array(im.tvec))
            print(f'Registered: {dataset} / {scene} -> {len(out_results[dataset][scene])} images')
            print(f'Total: {dataset} / {scene} -> {len(data_dict[dataset][scene])} images')
            create_submission(out_results, data_dict)
            gc.collect()
            
        # except:
        #     pass
            
        except Exception as e:
            print(f"An error occurred during the second loop: {e}")
            traceback.print_exc()
        
create_submission(out_results, data_dict, output_submission=output_submission)