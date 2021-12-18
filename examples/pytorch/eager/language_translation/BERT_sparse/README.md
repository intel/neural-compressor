## Quick Start Guide

To train your model using pruning, perform the following steps using the default parameters of the BERT model. 

1. Download the NVIDIA pretrained checkpoint & Download and preprocess the dataset, visit [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)

2. Setup environment of neural_compressor and BERT.

   1. Install [neural_compressor](https://github.com/intel-innersource/frameworks.ai.lpot.intel-lpot#installation)


   2. Install pytorch, visit [pytorch.org](https://pytorch.org/)

   3. pip3 install -r requirements.txt --ignore-installed PyYAML

   4. Install [apex](https://github.com/NVIDIA/apex)

3. Enter your created conda env, then run the script 

   ```
   bash scripts/run_squad_sparse.sh
   ```
then results save to <OUTPUT_DIR>, set in scripts/run_squad_sparse.sh