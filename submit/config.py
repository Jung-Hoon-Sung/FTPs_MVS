import sys

# Model 설정
local_features = ['LoFTR']  # 변경 가능: ['LoFTR'], ['DISK'], ['LoFTR', 'DISK'], ['LoFTR', 'KeyNetAffNetHardNet'],['KeyNetAffNetHardNet'], [DISK, 'KeyNetAffNetHardNet']

# 디바이스 설정 이건 아직 안바꿈!!! 좀 봐야겠음
DEVICE = 'cuda'  # 변경 가능: 'cuda', 'cpu' 

#Colmap에 사용되는 parameter [defult]
MAX_IMAGE_ID = 2**31 - 1
IS_PYTHON3 = sys.version_info[0] >= 3

# get image pairs shortlist
sim_th = 0.6 # should be strict
min_pairs = 20
exhaustive_if_less = 20

# image dir path
"""img_dir"""
src = '/home/jhun/IMC2023_innopam/data/image-matching-challenge-2023'
src2 = 'train' #src 디렉토리에서 어떤 폴더를 선택할지 정함 option : 'train', 'test' (user가 경로 내에 폴더를 생성하고 변경 가능)
src3 = 'images' # 디렉토리에서 어떤 폴더를 선택할지 정함 option : 'images', '...' (user가 경로 내에 폴더를 생성하고 변경 가능)
featureout = 'featureout' #  featureout path와 동일하게 변수명 맞추기 dependency함

# get_image_pairs_shortlist
checkpoint_path='/home/jhun/IMC2023_innopam/weights/tf_efficientnet_b7_ra-6c08e654.pth'

# KeyNetAffNetHardNet weight file path
OriNet = '/home/jhun/IMC2023_innopam/weights/OriNet.pth'
KeyNet = '/home/jhun/IMC2023_innopam/weights/keynet_pytorch.pth'
AffNet = '/home/jhun/IMC2023_innopam/weights/AffNet.pth'
HardNet = '/home/jhun/IMC2023_innopam/weights/HardNetLib.pth'

# DISK weight file path
DISK = '/home/jhun/IMC2023_innopam/weights/epipolar-save.pth'

# LoFTR weight file path
LoFTR = '/home/jhun/IMC2023_innopam/weights/outdoor_ds.ckpt'

# featureout path
# feature_dir = '.featureout'
feature_dir = '.featureout'

# detect feature and match feature for KeyNetAffNetHardNet or DISK 
"""parameter"""
num_feats = 8192
max_length = 1200
matching_alg = 'smnn' # option smnn, adalam
min_matches = 2

"""path or file"""
DISK_lafs = 'disk_lafs.h5'
DISK_keypoints = 'disk_keypoints.h5'
DISK_descriptors = 'disk_descriptors.h5'
DISK_matches = 'disk_matches.h5'

KeyNet_lafs = 'keynet_lafs.h5'
KeyNet_keypoints = 'keynet_keypoints.h5'
KeyNet_descriptors = 'keynet_descriptors.h5'
KeyNet_matches = 'keynet_matches.h5'

# match feature for LoFTR
"""parameter"""
LoFTR_max_length = 1200
LoFTR_min_matches = 15
"""path or file"""
LoFTR_matches = 'LoFTR_matches.h5'
LoFTR_keypoints = 'LoFTR_keypoints.h5'

# ensemble filename
"""filename"""
ensemble_keypoints = 'ensemble_keypoints.h5'
ensemble_matches = 'ensemble_matches.h5'

# matching이 끝나고 최종 import colmap할 때, 필요한 .h5 filename
"""path or file"""
keypoints_h5 = 'LoFTR_keypoints.h5' # option =>  KeyNetAffNetHardNet or DISK : keypoints.h5, loFTR : LoFTR_keypoints.h5, ensemble : ensemble_keypoints.h5
matches_h5 = 'LoFTR_matches.h5' # option =>  KeyNetAffNetHardNet or DISK : matches.h5, loFTR : LoFTR_matches.h5, ensemble : ensemble_matches.h5

""".h5는 pycolmap에 입력으로 들어가기 위해 생성하는 것""" 

# import_into_colmap
database_path = 'colmap.db'
img_ext = '.jpg' # 사실상 사용하지 않음, 단 USER가 사용할 수 있음 현재는 jpg, png, JEPG, etc... 처리됨

colmap_db = 'colmap.db'
# image list filename[csv format]
table_submission = 'all_class_sample_train_labels.csv'

# output submission [csv format]
output_submission = 'submission.csv'