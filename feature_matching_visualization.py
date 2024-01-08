import os
import cv2
import numpy as np
import h5py
import matplotlib.pyplot as plt
from glob import glob
import shutil
import argparse

# visualize_results
base_images_path = '/data/refined_images'
base_keypoints_path = "data/output_sp_sg_retri"
base_matches_path = "data/output_sp_sg_retri"
base_output_folder = "/data/visualization_output_sp_sg_retri"


def visualize_keypoints(img_path, keypoints_file, img_key, output_path=None):
    img = cv2.imread(img_path)
    
    # img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

    with h5py.File(keypoints_file, mode='r') as f_kp:
        try:
            kpts = f_kp[img_key][...]
        except KeyError:
            print(f"KeyError: No keypoints found for {img_key} in {keypoints_file}. Skipping...")
            return

    kpts_cv = [cv2.KeyPoint(pt[0], pt[1], 1) for pt in kpts]
    img_kpts = cv2.drawKeypoints(img, kpts_cv, None)

    if output_path is not None:
        img_kpts_resized = cv2.resize(img_kpts, (img_kpts.shape[1] // 2, img_kpts.shape[0] // 2))
        cv2.imwrite(output_path, img_kpts_resized)
    else:
        plt.figure(figsize=(15, 15))
        plt.imshow(cv2.cvtColor(img_kpts, cv2.COLOR_BGR2RGB))
        plt.show()

# def visualize_matches(img1_path, img2_path, keypoints_file, matches_file, img1_key, img2_key, output_path=None):
#     img1 = cv2.imread(img1_path)
#     img2 = cv2.imread(img2_path)

#     # with h5py.File(keypoints_file, mode='r') as f_kp:
#     #     print(list(file.keys()))
    
#     with h5py.File(keypoints_file, 'r') as file:
#         print(list(file.keys()))
        
#         kpts1 = f_kp[img1_key][...]
#         kpts2 = f_kp[img2_key][...]

#     with h5py.File(matches_file, mode='r') as f_match:
#         matches = f_match[img1_key][img2_key][...]

#     # 여기서 matches 리스트가 비어 있는지 확인
#     if len(matches) == 0:
#         print(f"No matches found between {img1_key} and {img2_key}. Skipping image creation...")
#         return

def visualize_matches(img1_path, img2_path, keypoints_file, matches_file, img1_key, img2_key, output_path=None):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    with h5py.File(keypoints_file, 'r') as f_kp:
        print("visualize_matches")
        print(list(f_kp.keys()))
        
        kpts1 = f_kp[img1_key][...]
        kpts2 = f_kp[img2_key][...]
        
    with h5py.File(matches_file, 'r') as f_match:

        matches = f_match[img1_key][img2_key][...]

    if len(matches) == 0:
        print(f"No matches found between {img1_key} and {img2_key}. Skipping image creation...")
        return
    
    kpts1_cv = [cv2.KeyPoint(pt[0], pt[1], 1) for pt in kpts1]
    kpts2_cv = [cv2.KeyPoint(pt[0], pt[1], 1) for pt in kpts2]
    matches_cv = [cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=0) for i, j in matches]

    img_matches = cv2.drawMatches(img1, kpts1_cv, img2, kpts2_cv, matches_cv, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    for match in matches_cv:
        pt1 = tuple(map(int, kpts1_cv[match.queryIdx].pt))
        pt2 = tuple(map(int, (kpts2_cv[match.trainIdx].pt[0] + img1.shape[1], kpts2_cv[match.trainIdx].pt[1])))
        
        color = tuple(np.random.randint(0, 256, 3).tolist())
        img_matches = cv2.line(img_matches, pt1, pt2, color, thickness=4)
    
    if output_path is not None:
        img_matches_resized = cv2.resize(img_matches, (img_matches.shape[1] // 2, img_matches.shape[0] // 2))
        cv2.imwrite(output_path, img_matches_resized)
    else:
        plt.figure(figsize=(15, 15))
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.show()


def visualize_all_keypoints(images_path, keypoints_file, output_folder, method_name):
    image_files = sorted(glob(os.path.join(images_path, "*.*")))

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for i, img_path in enumerate(image_files):
        img_key = os.path.basename(img_path)
        output_path = os.path.join(output_folder, f"{method_name}_{img_key}_keypoints.jpg")
        print(f"Processing image: {img_key}")
        visualize_keypoints(img_path, keypoints_file, img_key, output_path)
        
def visualize_all_matches(images_path, keypoints_file, matches_file, output_folder, method_name):
    image_files = sorted(glob(os.path.join(images_path, "*.*")))

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    with h5py.File(matches_file, mode='r') as f_match:
        for i, img1_path in enumerate(image_files):
            img1_key = os.path.basename(img1_path)
            if img1_key not in f_match:
                continue

            for img2_key in f_match[img1_key].keys():
                try:
                    img2_path = os.path.join(images_path, img2_key)

                    output_path = os.path.join(output_folder, f"{method_name}_{img1_key}_and_{img2_key}_markers.jpg")
                    print(f"Processing image pair: {img1_key} and {img2_key}")
                    visualize_matches(img1_path, img2_path, keypoints_file, matches_file, img1_key, img2_key, output_path)
                except KeyError:
                    print(f"KeyError: No markers found between {img1_key} and {img2_key}. Skipping...")

def visualize_all_methods(images_path, 
                          silk_keypoints_file, silk_matches_file, 
                          keynetaffnethardnet_keypoints_file, keynetaffnethardnet_matches_file,
                          keynetaffnetsosnet_keypoints_file, keynetaffnetsosnet_matches_file, 
                          loftr_keypoints_file, loftr_matches_file, 
                          ensemble_keypoints_file, ensemble_matches_file, 
                          disk_keypoints_file, disk_matches_file,
                          sift_keypoints_file, sift_matches_file, 
                          rootsift_keypoints_file, rootsift_matches_file,
                          superpoint_keypoints_file, superpoint_matches_file,
                          disk_lightglue_keypoints_file, disk_lightglue_matches_file,
                          sp_sg_keypoints_file, sp_sg_matches_file, 
                          output_folder,
                          only_matches=False, only_keypoints=False, visualize_methods=None):
    methods = [
        ('KeyNetAffNetHardNet', keynetaffnethardnet_keypoints_file, keynetaffnethardnet_matches_file),
        ('KeyNetAffNetSoSNet', keynetaffnetsosnet_keypoints_file, keynetaffnetsosnet_matches_file),
        ('LoFTR', loftr_keypoints_file, loftr_matches_file),
        ('DISK', disk_keypoints_file, disk_matches_file),
        ('SIFT', sift_keypoints_file, sift_matches_file),
        ('RootSIFT', rootsift_keypoints_file, rootsift_matches_file),
        ('Silk', silk_keypoints_file, silk_matches_file),
        ('SuperPoint', superpoint_keypoints_file, superpoint_matches_file),
        ('DISK_LightGlue', disk_lightglue_keypoints_file, disk_lightglue_matches_file),
        ('SP_SG', sp_sg_keypoints_file, sp_sg_matches_file),
        ('Ensemble', ensemble_keypoints_file, ensemble_matches_file),
    ]
    
    for method_name, keypoints_file, matches_file in methods:
        # Check if files exist for the method
        if keypoints_file is not None and not os.path.isfile(keypoints_file):
            print(f"No keypoints file found for {method_name}. Skipping...")
            continue
        if matches_file is not None and not os.path.isfile(matches_file):
            print(f"No markers file found for {method_name}. Skipping...")
            continue

        if visualize_methods is not None and method_name not in visualize_methods:
            continue
        print(f"Visualizing {method_name} results")
        method_output_folder = os.path.join(output_folder, method_name)
        keypoints_folder = os.path.join(method_output_folder, "keypoints")
        matches_folder = os.path.join(method_output_folder, "markers")
        if not only_matches:  # Only visualize keypoints if `only_matches` is False
            visualize_all_keypoints(images_path, keypoints_file, keypoints_folder, method_name)
        if not only_keypoints:  # Only visualize matches if `only_keypoints` is False
            visualize_all_matches(images_path, keypoints_file, matches_file, matches_folder, method_name)

    # Handle Ensemble separately as it does not use resolutions
    if visualize_methods is None or 'Ensemble' in visualize_methods:
        # Check if files exist for the Ensemble method
        if ensemble_keypoints_file is not None and not os.path.isfile(ensemble_keypoints_file):
            print(f"No keypoints file found for Ensemble. Skipping...")
        elif ensemble_matches_file is not None and not os.path.isfile(ensemble_matches_file):
            print(f"No matches file found for Ensemble. Skipping...")
        else:
            method_name = 'Ensemble'
            keypoints_file = ensemble_keypoints_file
            matches_file = ensemble_matches_file
            print(f"Visualizing {method_name} results")
            method_output_folder = os.path.join(output_folder, method_name)
            keypoints_folder = os.path.join(method_output_folder, "keypoints")
            matches_folder = os.path.join(method_output_folder, "markers")
            if not only_matches:  # Only visualize keypoints if `only_matches` is False
                visualize_all_keypoints(images_path, keypoints_file, keypoints_folder, method_name)
            if not only_keypoints:  # Only visualize matches if `only_keypoints` is False
                visualize_all_matches(images_path, keypoints_file, matches_file, matches_folder, method_name)

def process_all_subfolders(base_images_path, base_keypoints_path, base_matches_path, base_output_folder, 
                           only_matches=False, only_keypoints=False, visualize_methods=None, resolutions=['2088']):
    
    print(f"Processing images in path: {base_images_path}")

    output_folder = base_output_folder

    if 'Ensemble' in visualize_methods:
        ensemble_keypoints_file = os.path.join(base_keypoints_path, "ensemble_keypoints.h5")
        ensemble_matches_file = os.path.join(base_matches_path, "ensemble_matches.h5")
        visualize_all_methods(base_images_path, None, None, None, None, ensemble_keypoints_file, ensemble_matches_file, None, None, None, None, 
                              output_folder, only_matches=only_matches, only_keypoints=only_keypoints, visualize_methods=['Ensemble'])

    for res in resolutions:
        keynetaffnethardnet_keypoints_file = os.path.join(base_keypoints_path, f"{res}_KeyNetAffNetHardNet_keypoints.h5")
        keynetaffnethardnet_matches_file = os.path.join(base_matches_path, f"{res}_KeyNetAffNetHardNet_markers.h5")
        
        keynetaffnetsosnet_keypoints_file = os.path.join(base_keypoints_path, f"{res}_KeyNetAffNetSoSNet_keypoints.h5")
        keynetaffnetsosnet_matches_file = os.path.join(base_matches_path, f"{res}_KeyNetAffNetSoSNet_markers.h5")

        disk_keypoints_file = os.path.join(base_keypoints_path, f"{res}_DISK_keypoints.h5")
        disk_matches_file = os.path.join(base_matches_path, f"{res}_DISK_markers.h5")
        
        
        sift_keypoints_file = os.path.join(base_keypoints_path, f"{res}_SIFT_keypoints.h5")
        sift_matches_file = os.path.join(base_matches_path, f"{res}_SIFT_markers.h5")

        loftr_keypoints_file = os.path.join(base_keypoints_path, f"{res}_LoFTR_keypoints.h5")
        loftr_matches_file = os.path.join(base_matches_path, f"{res}_LoFTR_markers.h5")

        rootsift_keypoints_file = os.path.join(base_keypoints_path, f"{res}_RootSIFT_keypoints.h5")
        rootsift_matches_file = os.path.join(base_matches_path, f"{res}_RootSIFT_markers.h5")

        silk_keypoints_file = os.path.join(base_keypoints_path, f"{res}_Silk_keypoints.h5")
        silk_matches_file = os.path.join(base_matches_path, f"{res}_Silk_markers.h5")
        
        superpoint_keypoints_file = os.path.join(base_keypoints_path, f"{res}_SuperPoint_keypoints.h5")
        superpoint_matches_file = os.path.join(base_matches_path, f"{res}_SuperPoint_matches.h5")
        
        disk_lightglue_keypoints_file = os.path.join(base_keypoints_path, "keypoints.h5")
        disk_lightglue_matches_file = os.path.join(base_matches_path, "markers.h5")
        
        sp_sg_keypoints_file = os.path.join(base_keypoints_path, "keypoints.h5")
        sp_sg_matches_file = os.path.join(base_matches_path, "matches.h5")
        
        visualize_all_methods(base_images_path, 
                              silk_keypoints_file, silk_matches_file, 
                              keynetaffnethardnet_keypoints_file, keynetaffnethardnet_matches_file,
                              keynetaffnetsosnet_keypoints_file, keynetaffnetsosnet_matches_file,
                              loftr_keypoints_file, loftr_matches_file, 
                              None, None, 
                              disk_keypoints_file, disk_matches_file,
                              sift_keypoints_file, sift_matches_file,
                              rootsift_keypoints_file, rootsift_matches_file,
                              superpoint_keypoints_file, superpoint_matches_file,
                              disk_lightglue_keypoints_file, disk_lightglue_matches_file,
                              sp_sg_keypoints_file, sp_sg_matches_file,
                              output_folder, only_matches=only_matches, only_keypoints=only_keypoints, 
                              visualize_methods=['KeyNetAffNetHardNet', 'KeyNetAffNetSoSNet', 'LoFTR', 'DISK', 'RootSIFT', 'Silk', 'SIFT', 'SuperPoint', 'DISK_LightGlue', 'SP_SG'])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage various functionalities.")
    parser.add_argument("--visualize_results", action="store_true", default=True, help="Run image matching visualization.")
    parser.add_argument("--only_markers", action="store_true", default=True, help="Only visualize matches, skipping keypoints.")
    parser.add_argument("--only_keypoints", action="store_true", default=False, help="Only visualize keypoints, skipping matches.")
    parser.add_argument("--methods", nargs='+', default=['SP_SG'], help="Specify methods to visualize.")
    parser.add_argument("--resolutions", nargs='+', default=['2000'], help="Specify resolutions to process.")
    args = parser.parse_args()

    if args.visualize_results:
        # Adjust the base output folder and process
        if args.only_markers:
            sub_output_folder = os.path.join(base_output_folder, "only_mathes")
            if not os.path.exists(sub_output_folder):
                os.makedirs(sub_output_folder)
            process_all_subfolders(base_images_path, base_keypoints_path, base_matches_path, sub_output_folder, 
                                   only_matches=True, only_keypoints=False, 
                                   visualize_methods=args.methods, resolutions=args.resolutions)

        if args.only_keypoints:
            sub_output_folder = os.path.join(base_output_folder, "only_keypoints")
            if not os.path.exists(sub_output_folder):
                os.makedirs(sub_output_folder)
            process_all_subfolders(base_images_path, base_keypoints_path, base_matches_path, sub_output_folder, 
                                   only_matches=False, only_keypoints=True, 
                                   visualize_methods=args.methods, resolutions=args.resolutions)

        # If none of the flags is set, process for both markers and keypoints
        if not args.only_markers and not args.only_keypoints:
            process_all_subfolders(base_images_path, base_keypoints_path, base_matches_path, base_output_folder, 
                                   only_matches=False, only_keypoints=False, 
                                   visualize_methods=args.methods, resolutions=args.resolutions)

