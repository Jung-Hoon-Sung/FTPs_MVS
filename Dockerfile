# Use an official Ubuntu 20.04 runtime as a parent image
FROM ubuntu:20.04

# Set the timezone
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Make sure the interactive dialog is not shown during the build process
ENV DEBIAN_FRONTEND=noninteractive

# Update and install required packages and cleanup in one RUN to reduce image size
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && apt-get update && \
    apt-get install -y python3.8 python3-pip libgl1-mesa-dev libglib2.0-0 python3-tk && \
    rm -rf /var/lib/apt/lists/* && \
    # Reset the frontend back to its normal state
    DEBIAN_FRONTEND=""

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python packages
RUN pip3 install --no-cache-dir \
    fastapi uvicorn \
    opencv_python==4.5.5.64 \
    albumentations==0.5.1 --no-binary=imgaug,albumentations \
    ray==1.0.1 \
    einops==0.3.0 \
    kornia==0.6.12 \
    loguru==0.5.3 \
    yacs==0.1.8 \
    tqdm autopep8 pylint ipython jupyterlab matplotlib \
    h5py==3.1.0 \
    pytorch-lightning==1.3.5 \
    torchmetrics==0.6.0 \
    joblib==1.0.1 \
    fastprogress timm pydegensac pycolmap \
    pandas==2.0.3 \
    scipy==1.10.1 \
    scikit-image==0.21.0 \
    scikit-learn==1.3.0 \
    python-multipart

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run main.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
