from fastapi import FastAPI, Query, File, UploadFile, BackgroundTasks
from typing import List, Optional
import os
import logging
import uvicorn
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
from LS_OF_based_canny_api import Outlier_filtering
from DL_based_auto_marker_api import main_function
from convert_RC_CP_format_api import convert_RC_CP_format
from convert_MS_Marker_format_api import convert_MS_Marker_format
from enum import Enum
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
import shutil
from pathlib import Path
from zipfile import is_zipfile
import zipfile

app = FastAPI()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class FormatType(str, Enum):
    RealityCapture = "RealityCapture"
    MetaShape = "MetaShape"

@app.post("/Automaic_marker_generation")
async def Automaic_marker_generation_endpoint(
        images_zip: UploadFile = File(...),
        sharpness_threshold: float = 0.03,
        featureout: str = Query('/data/featureout', description="Feature output directory path"),
        checkpoint_path: str = Query('/data/weights/convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384.pth', description="Path to shortlist_pair_checkpoint file"),
        orinet_path: str = Query('/data/weights/OriNet.pth', description="Path to OriNet file"),
        keynet_path: str = Query('/data/weights/keynet_pytorch.pth', description="Path to KeyNet file"),
        affnet_path: str = Query('/data/weights/AffNet.pth', description="Path to AffNet file"),
        hardnet_path: str = Query('/data/weights/HardNet.pth', description="Path to HardNet file"),
        sosnet_path: str = Query('/data/weights/sosnet_32x32_liberty.pth', description="Path to SOSNet file"),
        disk_path: str = Query('/data/weights/epipolar-save.pth', description="Path to DISK file"),
        loftr_path: str = Query('/data/weights/outdoor_ds.ckpt', description="Path to LoFTR file"),
        features_resolutions: List[str] = Query(['KeyNetAffNetHardNet:2088'], description="List of features and resolutions"),
        num_feats: Optional[int] = 40000,
        output_csv_rc: str = Query('/data/converted_RC_CP.csv', description="Path to output csv file for RC_CP format"),
        output_csv_ms: str = Query('/data/converted_MS_Marker.csv', description="Path to output csv file for MS_Marker format"),
        limit_pair_markerlist: Optional[int] = 3):
        """
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
        """

        # 임시 폴더 생성
        temp_dir = "/data/sample_datasets"
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, images_zip.filename)

        # ZIP 파일 저장
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(images_zip.file, buffer)

        # 압축 해제
        extracted_path = os.path.join(temp_dir, "images")
        with zipfile.ZipFile(temp_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_path)

        # 이미지 처리
        save_dir = os.path.join(temp_dir, "images/processed_images")
        os.makedirs(save_dir, exist_ok=True)
        Outlier_filtering(temp_path, save_dir, sharpness_threshold)

        logger.info("DL_based_auto_marker process started with src: %s", save_dir)

        features_resolutions_dict = {item.split(":")[0]: [int(item.split(":")[1])] for item in features_resolutions}

        logger.info("Creating featureout directory: %s", featureout)
        os.makedirs(featureout, exist_ok=True)

        logger.info("Starting main_function with selected parameters")
        generated_files = main_function(save_dir, featureout, features_resolutions_dict, checkpoint_path, orinet_path, keynet_path, affnet_path, hardnet_path,
                    sosnet_path, disk_path, loftr_path, num_feats=num_feats)

        logger.info("DL_based_auto_marker process completed successfully!")
        
        keypoint_h5_dir = generated_files[0]
        marker_h5_dir = generated_files[1]

        # RC_CP 형식 변환
        csv_path_rc = convert_RC_CP_format(keypoint_h5_dir, marker_h5_dir, save_dir, output_csv_rc, limit_pair_markerlist)

        # MS_Marker 형식 변환
        logger.info("Conversion process (MS marker) started")
        csv_path_ms = convert_MS_Marker_format(keypoint_h5_dir, marker_h5_dir, save_dir, output_csv_ms, limit_pair_markerlist)
        
        return {
            "DL_based_auto_marker": "Successfully!!",
            "RC_CP_conversion": "Successfully!!",
            "MS_Marker_conversion": "Successfully!!"
        }

@app.get("/download_file")
def download_file(format_type: FormatType = Query(..., description="Specify 'RealityCapture' for RC format or 'MetaShape' for MS format")):
    """
    - **format_type**: 반환하려는 파일 형식을 지정합니다. 사용 가능한 값은 'RealityCapture' 또는 'MetaShape'입니다.
        - 'RealityCapture': RealityCapture csv형식의 파일을 다운로드합니다.
        - 'MetaShape': MetaShape csv형식의 파일을 다운로드합니다.
    """
    if format_type == FormatType.RealityCapture:
        output_csv = '/data/converted_RC_CP.csv'
    elif format_type == FormatType.MetaShape:
        output_csv = '/data/converted_MS_Marker.csv'
    else:
        return {"error": "Invalid format type specified."}

    if os.path.exists(output_csv):
        filename = os.path.basename(output_csv)
        return FileResponse(output_csv, filename=filename)
    else:
        return {"error": "File not found."}

    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=True)
