# import time
# import h5py
# import os
# import glob
# import pandas as pd

# def convert_RC_CP_format(keypoint_h5_dir, marker_h5_dir, image_dir, output_csv):
#     class UnionFind:
#         def __init__(self):
#             self.weights = {}
#             self.parents = {}

#         def __getitem__(self, object):
#             if object not in self.parents:
#                 self.parents[object] = object
#                 self.weights[object] = 1
#                 return object
#             path = [object]
#             root = self.parents[object]
#             while root != path[-1]:
#                 path.append(root)
#                 root = self.parents[root]
#             for ancestor in path:
#                 self.parents[ancestor] = root
#             return root

#         def __iter__(self):
#             return iter(self.parents)

#         def union(self, *objects):
#             roots = [self[x] for x in objects]
#             heaviest = max((self.weights[r], r) for r in roots)[1]
#             for r in roots:
#                 if r != heaviest:
#                     self.weights[heaviest] += self.weights[r]
#                     self.parents[r] = heaviest
                    
#         def sets(self):
#             parents = {}
#             for object in self:
#                 parent = self[object]
#                 if parent not in parents:
#                     parents[parent] = {object}
#                 else:
#                     parents[parent].add(object)
#             return list(parents.values())

#     start_time = time.time()

#     with h5py.File(keypoint_h5_dir, 'r') as f:
#         keypoints = {}
#         for k, v in f.items():
#             img_id = k
#             keypoints[img_id] = v[:]
#     print("Loaded keypoints: ", len(keypoints))

#     with h5py.File(marker_h5_dir, 'r') as f:
#         matches = {}
#         for k, v in f.items():
#             img_id = k
#             matches[img_id] = {}
#             for kk, vv in v.items():
#                 match_id = kk
#                 matches[img_id][match_id] = vv[:]
#     print("Loaded matches: ", len(matches))

#     uf = UnionFind()    

#     for img_id in keypoints.keys():
#         match_data = matches.get(img_id, {})
#         for match_id, match_indices in match_data.items():
#             if match_id not in keypoints:
#                 continue
#             for i in range(match_indices.shape[0]):
#                 tiepoint1_index = match_indices[i, 0]
#                 tiepoint2_index = match_indices[i, 1]
#                 uf.union((img_id, tiepoint1_index), (match_id, tiepoint2_index))

#     components = uf.sets()
#     print(f"Total connected components (keypoint groups): {len(components)}")

#     df = pd.DataFrame(columns=["img_id", "marker", "x", "y"])
#     marker_count = 0

#     data = []
#     for component in components:
#         marker_label = f"point {marker_count}"
#         for node in component:
#             img_id, tiepoint_index = node
#             x, y = tuple(keypoints[img_id][tiepoint_index])
#             img_id_no_ext = img_id.rstrip(".jpg")
#             data.append({"img_id": img_id_no_ext, "marker": marker_label, "x": x, "y": y})
#         marker_count += 1

#     df = pd.DataFrame(data)
#     df.to_csv(output_csv, index=False)

#     end_time = time.time()
#     print("Execution time: ", end_time - start_time)
    
# if __name__ == "__main__":
#     keypoint_h5_dir = '/data/featureout/auto_markers/1280_DISK/1280_DISK_keypoints.h5'
#     marker_h5_dir = '/data/featureout/auto_markers/1280_DISK/1280_DISK_markers_n2.h5'
#     image_dir = '/data/refined_images'
#     output_csv = '/data/rc_format_markers/factory134_1280_DISK_markers_n2.csv'
#     convert_RC_CP_format(keypoint_h5_dir, marker_h5_dir, image_dir, output_csv)

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
        img_id_no_ext = img_id.rstrip(".jpg")
        data.append({
            "img_id": img_id_no_ext,
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
