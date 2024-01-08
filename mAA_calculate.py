import numpy as np
import pandas as pd
import math
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
from sklearn.metrics import mean_squared_error
from prettytable import PrettyTable
# matplotlib.use('TkAgg')

@dataclass
class Camera:
    rotmat: np.array
    tvec: np.array

def rotation_matrix_to_quaternion(R):
    """Converts a rotation matrix to quaternion."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w
 
def quaternion_from_4x4_matrix(matrix): # 회전 매트릭스를 쿼터니언 형태로 변환, 쿼터니언은 3D 회전을 나타내는 수학적 표현 방법
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]

    # Symmetric matrix K.
    K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                  [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                  [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                  [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
    K /= 3.0

    # Quaternion is eigenvector of K that corresponds to largest eigenvalue.
    w, V = np.linalg.eigh(K) # np.linalg.eigh = 행렬의 고유값과 고유벡터를 계산하는 함수
    q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0.0: # 부호의 중복성을 없애기 위해 항상 양수가 되도록 하기 위함 
        np.negative(q, q)
    return q

def quaternion_from_3x3_matrix(matrix):
    """
    Convert a 3x3 rotation matrix to a quaternion.
    The quaternion is returned as [w, x, y, z].
    """
    m = np.array(matrix, dtype=np.float64, copy=False)
    m00, m01, m02 = m[0, 0], m[0, 1], m[0, 2]
    m10, m11, m12 = m[1, 0], m[1, 1], m[1, 2]
    m20, m21, m22 = m[2, 0], m[2, 1], m[2, 2]

    # Compute quaternion elements
    w = np.sqrt(max(0, 1 + m00 + m11 + m22)) / 2
    x = np.sqrt(max(0, 1 + m00 - m11 - m22)) / 2
    y = np.sqrt(max(0, 1 - m00 + m11 - m22)) / 2
    z = np.sqrt(max(0, 1 - m00 - m11 + m22)) / 2

    x = np.copysign(x, m21 - m12)
    y = np.copysign(y, m02 - m20)
    z = np.copysign(z, m10 - m01)

    return np.array([w, x, y, z])

def compute_mAA(err_q, err_t, ths_q, ths_t): # 평균 정확도 계산: 주어진 임계값에 대하여 회전 및 변환의 오차를 평가하고, 각 임계값에 대한 정확도를 계산
    '''Compute the mean average accuracy over a set of thresholds. Additionally returns the metric only over rotation and translation.'''
    err_q_array = np.array(err_q)
    err_t_array = np.array(err_t)
    acc, acc_q, acc_t = [], [], []
    for th_q, th_t in zip(ths_q, ths_t):
        cur_acc_q = (err_q_array <= th_q)
        cur_acc_t = (err_t_array <= th_t)
        cur_acc = cur_acc_q & cur_acc_t
        
        acc.append(cur_acc.astype(np.float32).mean())
        acc_q.append(cur_acc_q.astype(np.float32).mean())
        acc_t.append(cur_acc_t.astype(np.float32).mean())
    return np.array(acc), np.array(acc_q), np.array(acc_t)

def euler_to_rotation_matrix(heading, pitch, roll):
    """Converts heading, pitch, roll (in degrees) to a rotation matrix."""
    heading = math.radians(heading)
    pitch = math.radians(pitch)
    roll = math.radians(roll)
    
    R_x = np.array([[1,          0,                 0],
                    [0, math.cos(roll), -math.sin(roll)],
                    [0, math.sin(roll), math.cos(roll)]])

    R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                    [0,                1,                0],
                    [-math.sin(pitch), 0, math.cos(pitch)]])

    R_z = np.array([[math.cos(heading), -math.sin(heading), 0],
                    [math.sin(heading), math.cos(heading), 0],
                    [0,                 0,                 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def dict_from_eo_csv(csv_path):
    """Parse EO data from CSV into dictionary."""
    data = pd.read_csv(csv_path, skiprows=1, header=0)
    eo_dict = {}
    for idx, row in data.iterrows():
        R = euler_to_rotation_matrix(row['omega'], row['phi'], row['kappa'])
        T = np.array([row['x'], row['y'], row['z']])
        eo_dict[row['#name']] = Camera(rotmat=R, tvec=T)
    return eo_dict

def evaluate_R_t(R_gt, t_gt, R, t, eps=1e-15):
    t = t.flatten()
    t_gt = t_gt.flatten()

    # q_gt = rotation_matrix_to_quaternion(R_gt)
    # q = rotation_matrix_to_quaternion(R)
    q_gt = quaternion_from_4x4_matrix(R_gt)
    q = quaternion_from_4x4_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    GT_SCALE = np.linalg.norm(t_gt)
    t = GT_SCALE * (t / (np.linalg.norm(t) + eps))
    err_t = min(np.linalg.norm(t_gt - t), np.linalg.norm(t_gt + t))
    
    return np.degrees(err_q), err_t

def eval_eo_data(pred_csv_path, gt_csv_path, rotation_thresholds_degrees, translation_thresholds_meters, verbose=False):
    # EO 데이터의 정확도를 평가하고 평균 정확도를 반환: 주어진 예측 EO 데이터와 실제 EO 데이터 사이의 회전 및 변환 오차를 계산하고 평균 정확도를 출력
    pred_dict = dict_from_eo_csv(pred_csv_path)
    gt_dict = dict_from_eo_csv(gt_csv_path)

    err_q_all, err_t_all = [], []
    for image in gt_dict:
        R_gt, T_gt = gt_dict[image].rotmat, gt_dict[image].tvec
        R_pred, T_pred = pred_dict[image].rotmat, pred_dict[image].tvec
        err_q, err_t = evaluate_R_t(R_gt, T_gt, R_pred, T_pred)
        err_q_all.append(err_q)
        err_t_all.append(err_t)

    mAA, mAA_q, mAA_t = compute_mAA(err_q=err_q_all,
                                   err_t=err_t_all,
                                   ths_q=rotation_thresholds_degrees,
                                   ths_t=translation_thresholds_meters)
    
    if verbose:
        print(f'mAA={np.mean(mAA):.06f}, mAA_q={np.mean(mAA_q):.06f}, mAA_t={np.mean(mAA_t):.06f}')

    return np.mean(mAA)

def save_matrices_to_csv(data_dict, csv_path):
    """Save rotation and translation matrices to CSV."""
    rows = []
    for image, camera in data_dict.items():
        rotmat_flat = camera.rotmat.flatten()
        tvec = camera.tvec
        row = [image] + list(rotmat_flat) + list(tvec)
        rows.append(row)

    # Construct column names
    column_names = ["#name"]
    for i in range(3):
        for j in range(3):
            column_names.append(f"Rotation_matrix")
    column_names.extend(["translation_vector_x", "translation_vector_y", "translation_vector_z"])

    # Convert rows to DataFrame and save to CSV
    df = pd.DataFrame(rows, columns=column_names)
    df.to_csv(csv_path, index=False)

def save_all_matrices_to_csv(pred_dict, gt_dict, pred_csv_path, gt_csv_path):
    """Save rotation and translation matrices for both GT and predictions to CSV."""
    save_matrices_to_csv(pred_dict, pred_csv_path)
    save_matrices_to_csv(gt_dict, gt_csv_path)

# def quaternion_to_euler(q):
#     """
#     Convert a quaternion into euler angles (roll, pitch, yaw)
#     roll is rotation around x in radians (counterclockwise)
#     pitch is rotation around y in radians (counterclockwise)
#     yaw is rotation around z in radians (counterclockwise)
#     """
#     x, y, z, w = q
#     t0 = +2.0 * (w * x + y * z)
#     t1 = +1.0 - 2.0 * (x * x + y * y)
#     roll_x = math.atan2(t0, t1)
    
#     t2 = +2.0 * (w * y - z * x)
#     t2 = +1.0 if t2 > +1.0 else t2
#     t2 = -1.0 if t2 < -1.0 else t2
#     pitch_y = math.asin(t2)
    
#     t3 = +2.0 * (w * z + x * y)
#     t4 = +1.0 - 2.0 * (y * y + z * z)
#     yaw_z = math.atan2(t3, t4)
    
#     return roll_x, pitch_y, yaw_z  # in radians

def quaternion_to_opk(q):
    """
    Convert a quaternion into omega, phi, kappa angles (roll, pitch, yaw)
    omega is rotation around x in radians (counterclockwise)
    phi is rotation around y in radians (counterclockwise)
    kappa is rotation around z in radians (counterclockwise)
    """
    x, y, z, w = q
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    omega = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    phi = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    kappa = math.atan2(t3, t4)
    
    return omega, phi, kappa  # in radians

def generate_corrected_eo(data_dict):
    corrected_eo = {}
    for image, camera in data_dict.items():
        R = camera.rotmat
        # q = rotation_matrix_to_quaternion(R)
        q = quaternion_from_4x4_matrix(R)
        omega, phi, kappa = quaternion_to_opk(q)
        corrected_eo[image] = {
            'x': camera.tvec[0],
            'y': camera.tvec[1],
            'z': camera.tvec[2],
            'omega': omega,    
            'phi': phi,    
            'kappa': kappa      
        }
    return corrected_eo

def save_corrected_eo_to_csv(eo_dict, path):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['#camera'])  # 이 행을 추가
        writer.writerow(['#name', 'X', 'Y', 'Z', 'omega', 'Phi', 'kappa'])
        for image, values in eo_dict.items():
            writer.writerow([image, values['x'], values['y'], values['z'], values['omega'], values['phi'], values['kappa']])
    return path

    
def plot_camera(ax, R, T, color='r', label=None, length=0.5, image_id=None):
    """
    Plot a camera in 3D axes given rotation and translation.
    """
    # Origin of the camera
    O = T
    R = R.T

    # Compute camera orientation lines
    OX = O + length * R[:, 0]
    OY = O + length * R[:, 1]
    OZ = O + length * R[:, 2]

    # Plot camera center
    ax.scatter(*O, color=color, s=200, label=label)

    # Plot camera orientation lines
    ax.quiver(*O, *(OX - O), color='r', arrow_length_ratio=0.1)
    ax.quiver(*O, *(OY - O), color='g', arrow_length_ratio=0.1)
    ax.quiver(*O, *(OZ - O), color='b', arrow_length_ratio=0.1)

    # Add image ID as text next to the camera center
    if image_id is not None:
        ax.text(O[0], O[1], O[2], image_id, fontsize=10, color=color)

def visualize_camera_poses(pred_dict, gt_dict, image_name):
    """
    Visualize the GT and predicted camera poses for a given image.
    """
    # Create a new figure and 3D axis
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.grid(True)

    # Plot the GT camera pose
    R_gt, T_gt = gt_dict[image_name].rotmat, gt_dict[image_name].tvec
    plot_camera(ax, R_gt, T_gt, color='g', label='Ground Truth', length=0.005)

    # Plot the predicted camera pose
    R_pred, T_pred = pred_dict[image_name].rotmat, pred_dict[image_name].tvec
    plot_camera(ax, R_pred, T_pred, color='r', label='Prediction', length=0.005)

    # Set axis limits based on GT and Prediction
    max_range = np.array([T_gt - T_pred, T_pred - T_gt]).ptp(axis=1).max() / 2.0
    mid_x = (T_gt[0] + T_pred[0]) * 0.5
    mid_y = (T_gt[1] + T_pred[1]) * 0.5
    mid_z = (T_gt[2] + T_pred[2]) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Title with image name
    ax.set_title(f"Camera Poses for {image_name}")

    # Display distance between GT and prediction
    distance = np.linalg.norm(T_gt - T_pred)
    ax.text2D(0.05, 0.95, f"Distance: {distance:.2f}m", transform=ax.transAxes)

    ax.legend()
    plt.show()

def create_dicts_from_csv(filename):
    df = pd.read_csv(filename)
    data_dict = {}
    for _, row in df.iterrows():
        image_name = row['#name']
        rotmat = np.array(row[1:10]).reshape(3,3)
        tvec = np.array(row[10:13])
        data_dict[image_name] = {'rotmat': rotmat, 'tvec': tvec}
    return data_dict

def compute_errors(gt, pred, columns):
    errors = gt[columns].subtract(pred[columns])
    sq_errors = errors ** 2

    stats = {
        'mean': errors.mean(),
        'std': errors.std(),
        'min': errors.min(),
        'max': errors.max()
    }

    rmse = np.sqrt(sq_errors.mean())

    return rmse, stats

def display_stats(stats, columns):
    for col in columns:
        min_value = round(stats['min'][col], 4)
        max_value = round(stats['max'][col], 4)
        mean_value = round(stats['mean'][col], 4)
        std_value = round(stats['std'][col], 4)
        print(f"{col:<10}: Min: {min_value:>8}, Max: {max_value:>8}, Mean: {mean_value:>8}, Std: {std_value:>8}")

def multiply_xyz_by_100(values):
    return [value * 100 if i < 3 and not np.isnan(value) else value for i, value in enumerate(values)]

def radians_to_degrees(radians):
    # 라디안 값을 도로 변환
    return radians * (180 / np.pi)

def convert_angles(stats):
    # 각도를 라디안에서 도로 변환
    for key in stats:
        if key in ["Min", "Max", "Mean", "Std", "RMSE"]:
            # Omega, Phi, Kappa 변환 (인덱스 3, 4, 5)
            stats[key][3:] = [radians_to_degrees(value) if not np.isnan(value) else value for value in stats[key][3:]]
        elif key == "Total RMSE":
            # Total RMSE의 경우 Omega, Phi, Kappa만 변환 (인덱스 3, 4, 5)
            stats[key][3] = radians_to_degrees(stats[key][3]) if not np.isnan(stats[key][3]) else np.nan
    return stats
   
def main(pred_csv_path, gt_csv_path, save_base_path):
    # rotation_thresholds_degrees = np.linspace(0.2, 10, 10)
    # translation_thresholds_meters = np.geomspace(0.05, 1, 10)
    
    pred_dict = dict_from_eo_csv(pred_csv_path)
    gt_dict = dict_from_eo_csv(gt_csv_path)
    
    # for image in gt_dict:
    #     R_gt, T_gt = gt_dict[image].rotmat, gt_dict[image].tvec
    #     if image not in pred_dict:
    #         print(f"Warning: No prediction for image {image}. Skipping.")
    #         continue

    #     R_pred, T_pred = pred_dict[image].rotmat, pred_dict[image].tvec
    #     if R_pred is None or T_pred is None:
    #         print(f"Warning: No prediction for image {image}. Skipping.")
    #         continue

    #     err_q, err_t = evaluate_R_t(R_gt, T_gt, R_pred, T_pred)

    #     print(f"Image: {image}")
    #     print(f"    Rotation Error: {err_q:.2f}°, Translation Error: {err_t:.2f}m")

    #     mAA, mAA_q, mAA_t = compute_mAA(err_q=[err_q],
    #                                    err_t=[err_t],
    #                                    ths_q=rotation_thresholds_degrees,
    #                                    ths_t=translation_thresholds_meters)

    #     for i, (th_q, th_t) in enumerate(zip(rotation_thresholds_degrees, translation_thresholds_meters)):
    #         print(f"    Rotation Threshold: {th_q:.2f}°, Translation Threshold: {th_t:.2f}m")
    #         print(f"        Accuracy: {mAA[i]:.2f}, Rotation Accuracy: {mAA_q[i]:.2f}, Translation Accuracy: {mAA_t[i]:.2f}")
    
    # print("="*80)
    # mAA = eval_eo_data(
    #     pred_csv_path=pred_csv_path,
    #     gt_csv_path=gt_csv_path,
    #     rotation_thresholds_degrees=rotation_thresholds_degrees,
    #     translation_thresholds_meters=translation_thresholds_meters,
    #     verbose=True
    # )
    # print("Final mAA:", mAA)

###################################################################################################################
    # pred_save_path = save_base_path + "/trans_matrices/pred_rot_trans_matrices.csv"
    # gt_save_path = save_base_path + "/trans_matrices/gt_rot_trans_matrices.csv"

    # save_all_matrices_to_csv(
    #     pred_dict,
    #     gt_dict,
    #     pred_save_path,
    #     gt_save_path
    # )
    
    # gt_corrected_eo = generate_corrected_eo(gt_dict)
    # pred_corrected_eo = generate_corrected_eo(pred_dict)

    # gt_eo = save_corrected_eo_to_csv(gt_corrected_eo, "/data/EO_opk/corrected/corrected_gt_eo.csv")
    # pred_eo =save_corrected_eo_to_csv(pred_corrected_eo, "/data/EO_opk/corrected/corrected_pred_eo.csv")

###################rotation matrix visualization###################################################################################################################
    # image_name = "100_0208_0016.jpg"
    # 100_0208_0016.jpg
    # factory134_merge_901.lsp
    # visualize_camera_poses(pred_dict, gt_dict, image_name)

    # CSV에서 데이터 읽기
    # gt_dict = create_dicts_from_csv('/data/EO_opk/trans_matrices/gt_rot_trans_matrices.csv')
    # pred_dict = create_dicts_from_csv('/data/EO_opk/trans_matrices/pred_rot_trans_matrices.csv')

    # # 각 이미지 id에 대해 그림 그리기
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.grid(True)
    
    # for image_name in gt_dict.keys():
    #     R_gt, T_gt = gt_dict[image_name]['rotmat'], gt_dict[image_name]['tvec']
    #     plot_camera(ax, R_gt, T_gt, color='g', label=f"GT", length=10, image_id=image_name)

    # for image_name in pred_dict.keys():
    #     R_pred, T_pred = pred_dict[image_name]['rotmat'], pred_dict[image_name]['tvec']
    #     plot_camera(ax, R_pred, T_pred, color='r', label=f"Pred", length=10, image_id=image_name)
        
    # ax.set_title("Camera Poses for All Images")
    # plt.show()

###################################################################################################################################################################

    pd.set_option('display.float_format', '{:.4f}'.format)
    
    # Read GT and prediction CSV files
    gt = pd.read_csv(gt_csv, comment='#', skiprows=[1], header=None, names=['#name', 'x', 'y', 'z', 'omega', 'phi', 'kappa'])
    # print(gt)
    pred = pd.read_csv(pred_csv, comment='#', skiprows=[1], header=None, names=['#name', 'x', 'y', 'z', 'omega', 'phi', 'kappa'])
    # print(pred)

    # Sort the dataframes by the image ID
    gt = gt.sort_values(by=['#name']).reset_index(drop=True)
    pred = pred.sort_values(by=['#name']).reset_index(drop=True)

    # Find common image IDs
    common_image_ids = set(gt['#name']).intersection(set(pred['#name']))

    # Keep only the common image IDs
    gt = gt[gt['#name'].isin(common_image_ids)].reset_index(drop=True)
    pred = pred[pred['#name'].isin(common_image_ids)].reset_index(drop=True)

    # Ensure GT and predicted values are in the same order
    assert (gt['#name'] == pred['#name']).all()

    # Only keep necessary columns
    necessary_columns = ['#name', 'x', 'y', 'z', 'omega', 'phi', 'kappa']
    gt = gt[necessary_columns]
    pred = pred[necessary_columns]

    # Make sure the dataframes are sorted by the image ID
    gt = gt.sort_values(by=['#name'])
    pred = pred.sort_values(by=['#name'])

    # Ensure GT and predicted values are in the same order
    assert (gt['#name'] == pred['#name']).all()

    # Calculate errors
    errors = gt[['x', 'y', 'z', 'omega', 'phi', 'kappa']] - pred[['x', 'y', 'z', 'omega', 'phi', 'kappa']]

    # RMSE calculations
    rmse_values = [
        np.sqrt(mean_squared_error(gt['x'], pred['x'])),
        np.sqrt(mean_squared_error(gt['y'], pred['y'])),
        np.sqrt(mean_squared_error(gt['z'], pred['z'])),
        np.sqrt(mean_squared_error(gt['omega'], pred['omega'])),
        np.sqrt(mean_squared_error(gt['phi'], pred['phi'])),
        np.sqrt(mean_squared_error(gt['kappa'], pred['kappa']))
    ]

    # Total RMSE calculations
    sqrt_sumsq_xyz = np.sqrt(sum([rmse_values[i]**2 for i in range(3)]))
    sqrt_sumsq_omega_phi_kappa = np.sqrt(sum([rmse_values[i]**2 for i in range(3, 6)]))
    
    # Metrics dictionary
    stats = {
        "Min": errors.min(),
        "Max": errors.max(),
        "Mean": errors.mean(),
        "Std": errors.std(),  # 'Std.'가 아니라 'Std'를 사용합니다.
        "RMSE": rmse_values,
        "Total RMSE": [sqrt_sumsq_xyz, np.nan, np.nan, sqrt_sumsq_omega_phi_kappa, np.nan, np.nan]
    }
    
    # stats = convert_angles(stats)
    
    for key in ["Min", "Max", "Mean", "Std", "RMSE", "Total RMSE"]:
        stats[key] = multiply_xyz_by_100(stats[key])
    
    # stats = convert_angles(stats)
    
    group_header1 = "+------------+-----------------------------------------------+--------------------------------------------------+"
    group_header = "|            |                 Positions(cm)                 |                     Orientation                  |"
    
    # prettytable 객체 생성
    table = PrettyTable()

    # 열 제목 설정
    table.field_names = ["", "X", "Y", "Z", "Omega", "Phi", "Kappa"]

    # 각 메트릭에 대한 행 추가
    for metric in ["Min", "Max", "Mean", "Std", "RMSE"]:
        row = [metric] + [f"{value:.10f}" if not np.isnan(value) else "-" for value in stats[metric]]
        table.add_row(row)
        # table.add_row(['-' * len(metric)] + ['-' * len(f"{value:.10f}") for value in stats[metric]])  # 구분선 추가

    # Total RMSE 특별 처리
    total_rmse = ["Total RMSE"] + [f"{value:.10f}" if not np.isnan(value) else "-" for value in stats["Total RMSE"]]
    table.add_row(total_rmse)
    print(group_header1)
    print(group_header)
    print(table)
    
    RMSE_filename = f"{save_base_path}/output/RMSE_results_gt.csv"
    
    with open(RMSE_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(table.field_names)

        for row in table._rows:
            csv_writer.writerow(row)

    print(f"Complete!!!")
    
base_path = "/data/EO_opk"
pred_csv = base_path + "/240105_rtk_eo/240105_loftr_n3.csv"
gt_csv = base_path + "/240105_rtk_eo/factory_134_gt_eo.csv"
main(pred_csv, gt_csv, base_path)
