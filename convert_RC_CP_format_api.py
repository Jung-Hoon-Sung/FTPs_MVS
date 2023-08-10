import time
import h5py
import os
import glob
import pandas as pd
from collections import Counter

def convert_RC_CP_format(keypoint_h5_dir, marker_h5_dir, image_dir, output_csv, limit_pair_markerlist):
    class UnionFind:
        def __init__(self):
            """Create a new empty union-find structure."""
            self.weights = {}
            self.parents = {}

        def __getitem__(self, object):
            """Find and return the name of the set containing the object."""
            if object not in self.parents:
                self.parents[object] = object
                self.weights[object] = 1
                return object
            path = [object]
            root = self.parents[object]
            while root != path[-1]:
                path.append(root)
                root = self.parents[root]
            for ancestor in path:
                self.parents[ancestor] = root
            return root

        def __iter__(self):
            """Iterate through all items ever found or unioned by this structure."""
            return iter(self.parents)

        def union(self, *objects):
            """Find the sets containing the objects and merge them all."""
            roots = [self[x] for x in objects]
            heaviest = max((self.weights[r], r) for r in roots)[1]
            for r in roots:
                if r != heaviest:
                    self.weights[heaviest] += self.weights[r]
                    self.parents[r] = heaviest
                    
        def sets(self):
            """Get the sets of the union-find data structure."""
            parents = {}
            for object in self:
                parent = self[object]
                if parent not in parents:
                    parents[parent] = {object}
                else:
                    parents[parent].add(object)
            return list(parents.values())
                    
    start_time = time.time()

    with h5py.File(keypoint_h5_dir, 'r') as f:
        keypoints = {}
        for k, v in f.items():
            img_id = k
            keypoints[img_id] = v[:]
    print("Loaded keypoints: ", len(keypoints))

    with h5py.File(marker_h5_dir, 'r') as f:
        matches = {}
        for k, v in f.items():
            img_id = k
            matches[img_id] = {}
            for kk, vv in v.items():
                match_id = kk
                matches[img_id][match_id] = vv[:]
    print("Loaded matches: ", len(matches))

    # img_list creation and transformation into a dictionary
    img_list = glob.glob(os.path.join(image_dir, "*.jpg")) 
    # img_dict = {os.path.basename(img_path): img_path for img_path in img_list}
    img_dict = {os.path.basename(img_path).rstrip(".jpg"): img_path for img_path in img_list}

    uf = UnionFind()    

    for img_id in keypoints.keys():
        match_data = matches.get(img_id, {})
        for match_id, match_indices in match_data.items():
            if match_id not in keypoints:
                continue
            for i in range(match_indices.shape[0]):
                tiepoint1_index = match_indices[i, 0]
                tiepoint2_index = match_indices[i, 1]
                uf.union((img_id, tiepoint1_index), (match_id, tiepoint2_index))

    components = uf.sets()
    print(f"Total connected components (keypoint groups): {len(components)}")

    df = pd.DataFrame(columns=["img_id", "marker", "x", "y"])
    marker_count = 0
    total_components = len(components)

    data = []
    # Inside the loop where you create the data list:
    for i, component in enumerate(components):
        component_img_ids = set(img_id for img_id, _ in component)
        if len(component_img_ids) >= limit_pair_markerlist:
            for node in component:
                img_id, tiepoint_index = node
                marker_label = f"point {marker_count}" 
                x, y = tuple(keypoints[img_id][tiepoint_index])
                img_path = img_dict.get(img_id, "Path not found")  # Getting the image path from the dictionary
                img_id_no_ext = img_id.rstrip(".jpg")  # Remove the '.jpg' extension from img_id
                data.append({"img_id": img_id_no_ext, "marker": marker_label, "x": x, "y": y})
                # data.append({"img_id": f"{img_id}", "marker": marker_label, "x": x, "y": y})
                # data.append({"img_id": img_path, "marker": marker_label, "x": x, "y": y})
            marker_count += 1
            print(f"Processing component {i+1} of {total_components}")
            print(f"Created marker {marker_label} for component {i+1} of {total_components} with {len(component_img_ids)} images and {len(component)} nodes")

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

    end_time = time.time()
    print("Execution time: ", end_time - start_time)
    
if __name__ == "__main__":
    keypoint_h5_dir = '/data/featureout/factory134/1920_KeyNetAffNetHardNet/1920_KeyNetAffNetHardNet_keypoints.h5'
    marker_h5_dir = '/data/featureout/factory134/1920_KeyNetAffNetHardNet/1920_KeyNetAffNetHardNet_markers.h5'
    image_dir = '/data/refined_datasets/1920_KeyNetAffNetHardNet/images'
    output_csv = '/data/factory134_markers.csv'
    limit_pair_markerlist = 3
    convert_RC_CP_format(keypoint_h5_dir, marker_h5_dir, image_dir, output_csv, limit_pair_markerlist)