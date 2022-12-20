from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.data.dataloaders.dataloader import DataLoader
from neural_compressor.data import Datasets

def main():

    dataset = Datasets('tensorflow')['dummy'](shape=(1, 224, 224, 3))
    from neural_compressor.quantization import fit
    config = PostTrainingQuantConfig()
    fit(
        model="./mobilenet_v1_1.0_224_frozen.pb",
        conf=config,
        calib_dataloader=DataLoader(framework='tensorflow', dataset=dataset),
        eval_dataloader=DataLoader(framework='tensorflow', dataset=dataset))

if __name__ == "__main__":
    main()