git submodule update --init --recursive

cd co-tracker
git checkout main
pip install -e .
pip install matplotlib flow_vis tqdm tensorboard imageio[ffmpeg]
mkdir -p checkpoints
cd checkpoints
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth
cd ../../

cd dift
pip install xformers==0.0.29.post1
git checkout main
cd ../

pip install torchvision==0.20.0
pip install mediapipe==0.10.11
pip install  --force-reinstall transformers==4.45.2
pip install --force-reinstall huggingface_hub==0.25.2
pip install --force-reinstall numpy==2.1.3
pip install --force-reinstall torch==2.5.1
