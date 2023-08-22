git clone https://github.com/microsoft/CodeXGLUE/
cp -r ./CodeXGLUE/Code-Code/Defect-detection/dataset  dataset
cd dataset
pip install gdown
gdown https://drive.google.com/uc?id=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF
python preprocess.py