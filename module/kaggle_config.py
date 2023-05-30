import sys

IS_PYTHON3 = sys.version_info[0] >= 3 # python version
MAX_IMAGE_ID = 2**31 - 1 
DEVICE = 'cuda'  # 변경 가능: 'cuda', 'cpu' 

"""""""--------------------------------Required--------------------------------"""""""""
"""Model select"""
local_features = ['KeyNetAffNetHardNet']  # 변경 가능: ['LoFTR'], ['DISK'], ['LoFTR', 'DISK'], ['LoFTR', 'KeyNetAffNetHardNet'],['KeyNetAffNetHardNet'], ['DISK', 'KeyNetAffNetHardNet']

"""Ratio_resize"""
resolutions = [1536, 1696] # resolutions = [600, 840, 1024], [1500, 1700], [1700]

"""get image pairs shortlist"""
#checkpoint_path='/kaggle/input/efficientnetweightfile/tf_efficientnet_b7_ra-6c08e654.pth' # 모델은 이미지에서 특징(feature)을 추출하고, 이 특징들을 비교함으로써 서로 다른 이미지들 사이의 유사도를 판단하는 데 사용
checkpoint_path='/kaggle/input/efficientnetbl2/tf_efficientnet_l2_ns_475-bebbd00a.pth'
# checkpoint_path='/home/jhun/IMC_innopam/weights/tf_efficientnet_b8_ra-572d5dd9.pth'
sim_th = 0.6 # 두 이미지의 유사도가 sim_th보다 높아야만 유사하다고 간주하며, 그렇지 않으면 유사하지 않다고 판정
min_pairs = 20 # 각 이미지에 대해, min_pairs 수만큼의 최소한의 이미지 쌍을 선택, 만약 특정 이미지와 유사도가 sim_th 이하인 이미지가 min_pairs보다 적게 있다면, 해당 이미지와 가장 가까운 이미지들을 min_pairs 수만큼 선택

"""detect_features"""
num_feats = 50000 # max keypoints: 고려할 keypoint 수

"""match_features"""
matching_alg = 'adalam' # option smnn, adalam
ransac_iters = 128 # RANSAC 반복 횟수

"""match_features, match_loftr"""
min_matches = 10 # 두 이미지 간에 최소한으로 필요한 match되는 특징점들의 개수

"""mapper_options"""
min_num_matches = 30 # 3D 복구 과정 중 증분 매핑에서 사용되는 파라미터로, 두 이미지 간에 최소한으로 매칭되어야 하는 특징점의 개수
min_model_size = 3 # RANSAC (RANdom SAmple Consensus) 알고리즘을 적용할 때 필요한 최소한의 데이터 포인트 수
init_num_trials = 50 #초기 모델을 찾는 데 사용되는 RANSAC 반복 횟수 /defualt 200
ba_local_num_images = 10 # local bundle adjustment 단계에서 고려할 이미지의 수 /defualt 6
ba_global_max_num_iterations = 100 # # Set the maximum number of global bundle adjustment iterations /defualt 50
"""""""--------------------------------Optional--------------------------------"""""""""
"""get image pairs shortlist"""

# 전체적인 이미지 쌍을 생성하는 방식을 결정하는 파라미터, 만약 전체 이미지의 수가 이 값보다 적다면, 모든 가능한 이미지 쌍을 생성, 그렇지 않다면, sim_th와 min_pairs에 의해 결정된 방식에 따라 이미지 쌍을 생성
exhaustive_if_less = 20

"""detect_features"""
num_features = 2048 # OriNet을 통해서 keypoints에 회전 불변성을 부여할 개수
scale_laf = 1.0 # Local Affine Frame 비율 조정 (이미지의 회전, 크기 조절, 그리고 이동 변환을 포함)

"""match_features"""
search_expansion = 16 #매칭 검색을 확장하는 정도 (adalam options)

"""mapper_options"""
num_threads = -1 # 병렬처리를 위한 스레드 수를 설정 /defualt -1 (-1은 사용 가능한 모든 코어를 사용하도록 지시)
max_num_models = 50 # 처리할 수 있는 최대 모델 수를 설정 /defualt 50
ba_global_images_freq = 100 # 몇 개의 이미지가 처리된 후에 global bundle adjustment을 수행할 것인지를 결정 /defualt 500
ba_global_points_freq = 100000 # 몇 개의 3D 점이 처리된 후에 global bundle adjustment을 수행할 것인지를 결정 /defualt 250000

"""""""--------------------------------image and weights path--------------------------------"""""""""
"""img dir"""
src = '/kaggle/input/image-matching-challenge-2023'
src2 = 'test' #src 디렉토리에서 어떤 폴더를 선택할지 정함 option : 'train', 'test' (user가 경로 내에 폴더를 생성하고 변경 가능)
src3 = 'images' # 디렉토리에서 어떤 폴더를 선택할지 정함 option : 'images', '...' (user가 경로 내에 폴더를 생성하고 변경 가능)
featureout = 'featureout' #  featureout path와 동일하게 변수명 맞추기 dependency함

"""weight file path"""
# KeyNetAffNetHardNet file path
OriNet = '/kaggle/input/kornia-local-feature-weights/OriNet.pth'
KeyNet = '/kaggle/input/kornia-local-feature-weights/keynet_pytorch.pth'
AffNet = '/kaggle/input/kornia-local-feature-weights/AffNet.pth'
HardNet = '/kaggle/input/kornia-local-feature-weights/HardNetLib.pth'
# HardNet8 = '/kaggle/input/kornia-local-feature-weights/hardnet8v2.pt'
# DISK weight file path
Disk = '/kaggle/input/diskweights/epipolar-save.pth'
# LoFTR weight file path
LoFTR = '/kaggle/input/loftrweights/weights/outdoor_ds.ckpt'

# featureout path
feature_dir = '.featureout'
# import_into_colmap
database_path = 'colmap.db'

"""""""--------------------------------Create CSV--------------------------------"""""""""
# image list filename[csv format]
table_submission = 'sample_submission.csv'

# output submission [csv format]
output_submission = 'submission.csv'