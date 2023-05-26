import sys

# Model 설정
local_features = ['KeyNetAffNetHardNet']  # 변경 가능: ['LoFTR'], ['DISK'], ['LoFTR', 'DISK'], ['LoFTR', 'KeyNetAffNetHardNet'],['KeyNetAffNetHardNet'], [DISK, 'KeyNetAffNetHardNet']
resolutions = [1700]

# resolutions = [600, 840, 1024]
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
src = '/home/jhun/IMC_innopam/data/image-matching-challenge-2023'
src2 = 'train' #src 디렉토리에서 어떤 폴더를 선택할지 정함 option : 'train', 'test' (user가 경로 내에 폴더를 생성하고 변경 가능)
src3 = 'images' # 디렉토리에서 어떤 폴더를 선택할지 정함 option : 'images', '...' (user가 경로 내에 폴더를 생성하고 변경 가능)
featureout = 'featureout' #  featureout path와 동일하게 변수명 맞추기 dependency함

# get_image_pairs_shortlist
checkpoint_path='/home/jhun/IMC_innopam/weights/tf_efficientnet_b7_ra-6c08e654.pth'

# KeyNetAffNetHardNet weight file path
OriNet = '/home/jhun/IMC_innopam/weights/OriNet.pth'
KeyNet = '/home/jhun/IMC_innopam/weights/keynet_pytorch.pth'
AffNet = '/home/jhun/IMC_innopam/weights/AffNet.pth'
HardNet = '/home/jhun/IMC_innopam/weights/HardNetLib.pth'
HardNet8 = '/home/jhun/IMC_innopam/weights/hardnet8v2.pt'

# DISK weight file path
DISK = '/home/jhun/IMC_innopam/weights/epipolar-save.pth'

# LoFTR weight file path
LoFTR = '/home/jhun/IMC_innopam/weights/outdoor_ds.ckpt'

# featureout path
# feature_dir = '.featureout'
feature_dir = '.featureout'

# detect feature and match feature for KeyNetAffNetHardNet or DISK 
"""parameter"""
num_feats = 40000
matching_alg = 'adalam' # option smnn, adalam
min_matches = 2

# import_into_colmap
database_path = 'colmap.db'
colmap_db = 'colmap.db'
# image list filename[csv format]
table_submission = 'all_class_sample_train_labels.csv'

# output submission [csv format]
output_submission = 'submission.csv'