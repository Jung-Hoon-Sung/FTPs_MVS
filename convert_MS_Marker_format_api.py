import time
import h5py
import os
import glob
import pandas as pd

def convert_RC_CP_format(keypoint_h5_dir, marker_h5_dir, image_dir, output_csv):
    start_time = time.time()

    # 키포인트와 매치 정보 로드
    with h5py.File(keypoint_h5_dir, 'r') as f:
        keypoints = {k: v[:] for k, v in f.items()}
    print("Loaded keypoints: ", len(keypoints))

    with h5py.File(marker_h5_dir, 'r') as f:
        matches = {k: {kk: vv[:] for kk, vv in v.items()} for k, v in f.items()}
    print("Loaded matches: ", len(matches))

    # 특징점을 그룹화하기 위한 딕셔너리
    feature_groups = {}
    group_id = 0

    # 각 이미지에 대한 매칭을 처리하여 그룹화
    for img_id, match_data in matches.items():
        for match_id, match_indices in match_data.items():
            for index_pair in match_indices:
                key1 = (img_id, index_pair[0])
                key2 = (match_id, index_pair[1])

                if key1 not in feature_groups and key2 not in feature_groups:
                    # 두 특징점 모두 새로운 그룹에 속함
                    feature_groups[key1] = feature_groups[key2] = group_id
                    group_id += 1
                elif key1 in feature_groups and key2 not in feature_groups:
                    # 첫 번째 특징점이 기존 그룹에 속함
                    feature_groups[key2] = feature_groups[key1]
                elif key2 in feature_groups and key1 not in feature_groups:
                    # 두 번째 특징점이 기존 그룹에 속함
                    feature_groups[key1] = feature_groups[key2]
                # 이미 두 특징점 모두 그룹에 속해있는 경우 추가 작업은 필요 없음

    # DataFrame 생성
    data = []
    for (img_id, index), group_id in feature_groups.items():
        x, y = keypoints[img_id][index]
        # img_id_no_ext = img_id.rstrip(".jpg")
        data.append({
            "img_id": img_id,
            "marker": f"point {group_id}",
            "x": x,
            "y": y
        })

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

    end_time = time.time()
    print("Execution time: ", end_time - start_time)
    
if __name__ == "__main__":
    keypoint_h5_dir = '/data/featureout/auto_markers/1280_DISK/1280_DISK_keypoints.h5'
    marker_h5_dir = '/data/featureout/auto_markers/1280_DISK/1280_DISK_markers_n2.h5'
    image_dir = '/data/refined_images'
    output_csv = '/data/rc_format_markers/factory134_1280_DISK_markers_n2.csv'
    convert_RC_CP_format(keypoint_h5_dir, marker_h5_dir, image_dir, output_csv)
