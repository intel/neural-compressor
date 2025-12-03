pip install -r requirements.txt

# opencv-python lacks deps
pip uninstall opencv-python
pip install opencv-python-headless==4.11.0.86

if [ ! -e "FramePack" ]; then
    git clone --depth 1 https://github.com/lllyasviel/FramePack.git
    cd FramePack
    git fetch origin 97fe5dbe06ac1f337ece08935b1076a35eefeeb9 --depth=1
    git reset --hard FETCH_HEAD
    cd ..
    cp -r FramePack/diffusers_helper/ .
fi

if [ ! -e "VBench" ]; then
    git clone --depth 1 https://github.com/Vchitect/VBench.git
    cd VBench
    git fetch origin 07bc8a4b74d5e0a23de42ed5880b899a1ff705f0 --depth=1
    git reset --hard FETCH_HEAD
    sh vbench2_beta_i2v/download_data.sh
    cd ..
fi
