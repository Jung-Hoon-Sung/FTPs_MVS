# IMC_innopam

## Dataset Directory

- **NAS4**
  - Path: `/volume1/1_InternalCompany/Auto_marker_data/datasets/`

## Pretrained Model Directory

- **NAS4**
  - Path: `/volume1/1_InternalCompany/Auto_marker_data/weights/`

## Installation

**Clone the repository**
```bash
git clone https://github.com/innopam/IMC_innopam.git
```
**Move into the cloned directory**
```
cd IMC_innopam
```
## Build Dockerfile

```bash
docker build . -t auto_marker_api
```
## docker run
```bash
docker run --gpus all -it --name auto_markers_api -v <your/local/directory/data>:/data -p 7000:80 auto_marker_api bash
```
- docker run example: `docker run --gpus all -it --name auto_markers_api -v /media/jhun/4TBHDD/auto_marker_docker_data/data:/data -p 7000:80 auto_marker_api bash`

- docker run example2: When using `mAA_calculate.py`, matplotlib visualization can be implemented together. When you want to do this, do it like this:
```bash
xhost +local:docker

docker run --gpus all -it --name auto_markers \
-v /media/jhun/4TBHDD/auto_marker_docker_data/data:/data \
-p 7000:80 \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
auto_marker_api bash
```
```
cd app
```
## Fast-API run
```bash
python3 main.py
```
- example url: `http://192.168.0.88:7000/docs`으로 접속

## Sample datasets for test
- download link : https://drive.google.com/drive/folders/1IJC0Z1XyOr7E0StbeYWvPn4uTHxHIXwf?usp=drive_link
## Step1 Automaic_marker_generation_endpoint

Automaic_marker_generation_endpoint는 다음과 같이 작업하십시오:

- **Image datasets ZIP 파일을 업로드(주의사항: 꼭 .zip으로 압축해서 업로드하십시오.**
    - 업로드 된 ZIP 파일은 처리된 후, 처리된 이미지들이 저장되는 디렉토리로 사용됩니다.
    -  Request body -> images_zip에 업로드 하시면 됩니다.

- **모든 결과 저장 경로 featureout**:
    - 예시: `/data/featureout`

- **모든 weights 저장 경로 <pth,chpt path>**:
    - checkpoint_path: `/data/weights/convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384.pth`
    - orinet_path: `/data/weights/OriNet.pth`
    - keynet_path: `/data/weights/keynet_pytorch.pth`
    - affnet_path: `/data/weights/AffNet.pth`
    - hardnet_path: `/data/weights/HardNet.pth`
    - sosnet_path: `/data/weights/sosnet_32x32_liberty.pth`
    - disk_path: `/data/weights/epipolar-save.pth`
    - loftr_path: `/data/weights/outdoor_ds.ckpt`

- **선택 가능한 모델과 해상도 (콤마로 구분)**:
    - KeyNetAffNetHardNet_1920 = "KeyNetAffNetHardNet:1920"
    - KeyNetAffNetSoSNet_1920 = "KeyNetAffNetSoSNet:1920"
    - LoFTR_1280 = "LoFTR:1280"
    - DISK_1280 = "DISK:1280"
    - ensemble_KeyNetAffNetHardNet_1920_1696 = "KeyNetAffNetHardNet:1920, 1696"
    - ensemble_KeyNetAffNetHardNet_KeyNetAffNetSoSNet_1920 = "KeyNetAffNetHardNet, KeyNetAffNetSoSNet:1920"

- **keypoints 수**:
    - num_feats = 40000 (사용자가 조정 가능, 예: 40000, 20000, 8192, 4096, 2048)
    
- **RC_CP 포맷과 MS_Marker 포맷으로 변환될 CSV 파일의 출력 경로**:
    - output_csv_rc: RC_CP 형식으로 변환된 데이터를 저장할 경로 (예: `/data/converted_RC_CP.csv`)
    - output_csv_ms: MS_Marker 형식으로 변환된 데이터를 저장할 경로 (예: `/data/converted_MS_Marker.csv`)

- **Limit Pair Marker List (선택 사항)**:
    - limit_pair_markerlist: 마커 쌍의 최대 개수를 지정합니다 (예: 3)


## Step 2 download_file
- **format_type**: 반환하려는 파일 형식을 지정합니다. 사용 가능한 값은 'RealityCapture' 또는 'MetaShape'입니다.
    - 'RealityCapture': RealityCapture csv형식의 파일을 다운로드합니다.
    - 'MetaShape': MetaShape csv형식의 파일을 다운로드합니다.
 
## Other works
- The user can obtain the marker csv format required for Metashape.
- This can be customized by using the `import_csv_marker.py` script in MetaShape GUI, and the method is as follows:
- Append `import_csv_marker.py` to MetaShape GUI Console/Run Script. Then, a Custom menu will be created at the top. You just need to insert `python3 DL_based_auto_marker.py` into the newly created Custom menu.


