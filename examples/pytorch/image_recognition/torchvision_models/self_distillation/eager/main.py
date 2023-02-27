import argparse
import importlib
import logging
import os
import random
import shutil
import warnings

import torch
# Added by Xin Chen
import torchvision
import torchvision.models.resnet as models
import torchvision.transforms as transforms
from neural_compressor.data.dataloaders.pytorch_dataloader import \
    PyTorchDataLoader
from neural_compressor.utils import logger

from autoaugment import CIFAR10Policy
from cutout import Cutout
from resnet import *


def is_optuna_available():
    return importlib.util.find_spec("optuna") is not None


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="self distillation")
parser.add_argument(
    "-t",
    "--topology",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=128,
    type=int,
    metavar="N",
    help="mini-batch size (default: 128), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--config", default=None, help="tuning config")
parser.add_argument("--output-model", default=None, help="output path", type=str)
parser.add_argument("--autoaugment", default=True, type=bool)
parser.add_argument("--cpu", action="store_true", help="using cpu for training")
parser.add_argument("--hpo", action="store_true", help="enable HPO search")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
# optimizer
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.1, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                    help='weight decay (default: 1e-3)')
# criterion
parser.add_argument("--layer_mappings", default=[
                    [['resblock.1.feature.output', 'resblock.deepst.feature.output'],
                    ['resblock.2.feature.output','resblock.deepst.feature.output']],
                    [['resblock.1.fc','resblock.deepst.fc'],
                    ['resblock.2.fc','resblock.deepst.fc'],
                    ['resblock.3.fc','resblock.deepst.fc']]
                    ], type=str, nargs='+',
                    help='layer pairs used for self distillation')
parser.add_argument("--temperature", default=3, type=float,
                    help='temperature parameter of distillation')
parser.add_argument("--loss_types", default=['L2', 'CE'], type=str, nargs='+',
                    help='loss types of distillation, should be a list of length 2, '
                    'first for student targets loss, second for teacher student loss.')
parser.add_argument("--loss_weights", default=[0.03, 0.3], type=float, nargs='+',
                    help='loss weights of distillation, should be a list of length 2, '
                    'and sum to 1.0, first for student targets loss weight, '
                    'second for teacher student loss weight.')
parser.add_argument('--add_origin_loss', default=True,
                    help='whether add origin loss')

args = parser.parse_args()
logger.info(f"{args}")

best_score = 0

def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )
    main_worker(args)


def train(model, compression_manager, device, trainloader, testloader, trial=None):
    start_epoch = args.start_epoch
    end_epoch = args.epochs

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9
    )
    weight_logit =args.loss_weights[1]
    init = False
    for nepoch in range(start_epoch, end_epoch):
        if nepoch in [end_epoch // 3, end_epoch * 2 // 3, end_epoch - 10]:
            for param_group in optimizer.param_groups:
                param_group["lr"] /= 10
        correct = [0 for _ in range(5)]
        predicted = [0 for _ in range(5)]
        logger.info("Epoch at {}".format(nepoch))
        model.train()
        cnt = 0
        sum_loss, total = 0.0, 0.0
        compression_manager.callbacks.on_epoch_begin(nepoch)
        for image, target in trainloader:
            cnt = cnt + 1
            image = image.to(device)
            target = target.to(device)
            compression_manager.callbacks.on_step_begin(cnt)
            outputs, features = model(image)
            ensemble = sum(outputs[:-1]) / len(outputs)
            ensemble.detach_()
            if init is False:
                #   init the adaptation layers.
                #   we add feature adaptation layers here to soften the influence from feature distillation loss
                #   the feature distillation in our conference version :  | f1-f2 | ^ 2
                #   the feature distillation in the final version : |Fully Connected Layer(f1) - f2 | ^ 2
                layer_list = []
                teacher_feature_size = features[0].size(1)
                for index in range(1, len(features)):
                    student_feature_size = features[index].size(1)
                    layer_list.append(
                        nn.Linear(student_feature_size, teacher_feature_size)
                    )
                model.adaptation_layers = nn.ModuleList(layer_list)
                if device.type == "cuda":
                    model.adaptation_layers.cuda()
                init = True
            # compute loss
            loss = torch.FloatTensor([0.0]).to(device)
            outputs_features = dict()
            outputs_features["resblock.deepst.feature.output"] = features[0].detach()
            outputs_features["resblock.2.feature.output"] = model.adaptation_layers[1](
                features[2]
            )
            outputs_features["resblock.1.feature.output"] = model.adaptation_layers[2](
                features[3]
            )
            outputs_features["resblock.deepst.fc"] = outputs[0].detach()
            outputs_features["resblock.3.fc"] = outputs[1]
            outputs_features["resblock.2.fc"] = outputs[2]
            outputs_features["resblock.1.fc"] = outputs[3]
            #  for deepest classifier
            loss += criterion(outputs[0], target)
            #   for shallow classifier
            for index in range(1, len(outputs)):
                loss += criterion(outputs[index], target) * (1 - weight_logit)
            loss = compression_manager.callbacks.on_after_compute_loss(
                image, outputs_features, loss, teacher_output=outputs_features
            )
            sum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(target.size(0))
            outputs.append(ensemble)

            for classifier_index in range(len(outputs)):
                _, predicted[classifier_index] = torch.max(
                    outputs[classifier_index].data, 1
                )
                correct[classifier_index] += float(
                    predicted[classifier_index].eq(target.data).cpu().sum()
                )
            if cnt % 50 == 0:
                logger.info(
                    "[epoch:%d, iter:%d] Loss: %.03f | Acc: 4/4: %.2f%% 3/4: %.2f%% 2/4: %.2f%%  1/4: %.2f%%"
                    " Ensemble: %.2f%%"
                    % (
                        nepoch + 1,
                        (cnt + 1),
                        sum_loss / (cnt + 1),
                        100 * correct[0] / total,
                        100 * correct[1] / total,
                        100 * correct[2] / total,
                        100 * correct[3] / total,
                        100 * correct[4] / total,
                    )
                )
            compression_manager.callbacks.on_step_end()
        compression_manager.callbacks.on_epoch_end()
        validate(model, compression_manager, device, testloader, trial, nepoch)
    return model


def validate(model, compression_manager, device, testloader, trial=None, nepoch=None):
    """
    output:
    correct[0]: 4/4 result
    correct[1]: 3/4 result
    correct[2]: 2/4 result
    correct[3]: 1/4 result
    correct[4]: ensemble result
    """
    logger.info("Run validation phase")
    with torch.no_grad():
        correct = [0 for _ in range(5)]
        pred = [0 for _ in range(5)]
        total = 0.0
        for image, target in testloader:
            model.eval()
            image = image.to(device)
            target = target.to(device)
            outputs, outputs_features = model(image)
            ensemble = sum(outputs) / len(outputs)
            outputs.append(ensemble)
            for classifier_index in range(len(outputs)):
                _, pred[classifier_index] = torch.max(outputs[classifier_index].data, 1)
                correct[classifier_index] += float(
                    pred[classifier_index].eq(target.data).cpu().sum()
                )
            total += float(target.size(0))
        correct[:] = [x / total for x in correct]
        logger.info(
            "Validation Set AccuracyAcc: 4/4: %.4f%% 3/4: %.4f%% 2/4: %.4f%%  1/4: %.4f%%"
            " Ensemble: %.4f%%"
            % (
                100 * correct[0],
                100 * correct[1],
                100 * correct[2],
                100 * correct[3],
                100 * correct[4],
            )
        )
    global best_score
    if trial and nepoch:
        trial.report(correct[-1], step=nepoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    if correct[-1] > best_score:
        best_score = correct[-1]
        if trial:
            trial.set_user_attr("first block Accurcy", correct[0])
            trial.set_user_attr("second block Accurcy", correct[1])
            trial.set_user_attr("third block Accurcy", correct[2])
            trial.set_user_attr("fourth block Accurcy", correct[3])
    return correct[-1]


def main_worker(args):
    logging.info("=> creating model '{}'".format(args.topology))
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.autoaugment:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4, fill=128),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4, fill=128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    trainset = torchvision.datasets.CIFAR100(
        root=args.data,
        train=True,
        download=True,
        transform=transform_train,
    )
    trainloader = PyTorchDataLoader(
        trainset, batch_size=args.batch_size, num_workers=4, shuffle=True
    )

    testset = torchvision.datasets.CIFAR100(
        root=args.data, train=False, download=True, transform=transform_test
    )

    testloader = PyTorchDataLoader(testset, batch_size=args.batch_size)

    if args.hpo:
        if is_optuna_available():
            import optuna
        else:
            raise RuntimeError(
                "no optuna is found, please 'pip install optuna' firstly for hpo case"
            )

    if not args.hpo:
        model = resnet50()
        model = model.to(device)

        def train_func(model, compression_manager):
            return train(model, compression_manager, device, trainloader, testloader)

        def eval_func(model, compression_manager):
            return validate(model, compression_manager, device, testloader)

        import copy
        from neural_compressor.training import prepare_compression
        from neural_compressor.config import DistillationConfig, \
                                             SelfKnowledgeDistillationLossConfig
        distil_loss = SelfKnowledgeDistillationLossConfig(
            layer_mappings=args.layer_mappings,
            temperature=args.temperature,
            loss_types=args.loss_types,
            loss_weights=args.loss_weights,
            add_origin_loss=args.add_origin_loss,
        )
        conf = DistillationConfig(teacher_model=model, criterion=distil_loss)
        compression_manager = prepare_compression(copy.deepcopy(model), conf)
        model = compression_manager.model
        train_func(model, compression_manager)
        compression_manager.save(args.output_model)
        accu = eval_func(model, compression_manager)
        logging.info("Distilled model Accuracy:", accu)
    else:

        def objective(trial):
            global best_score
            best_score = 0
            model = resnet50()
            model = model.to(device)

            def train_func(model, compression_manager, trial):
                return train(model, compression_manager, device, trainloader, testloader, trial)

            def eval_func(model, compression_manager, trial=None):
                return validate(model, compression_manager, device, testloader, trial)

            import copy
            from neural_compressor.training import prepare_compression
            from neural_compressor.config import DistillationConfig, \
                                                SelfKnowledgeDistillationLossConfig
            args.loss_weights[1] = trial.suggest_float("loss_coefficient", 0, 1)
            args.loss_weights[0] = trial.suggest_float(
            "feature_loss_coefficient", 0, 0.1)
            args.temperature = trial.suggest_int("temperature", 1, 5)
            args.epochs = trial.suggest_int("epochs", 1, 20)
            logger.info(f"trial param: {trial.params}")
            distil_loss = SelfKnowledgeDistillationLossConfig(
                layer_mappings=args.layer_mappings,
                temperature=args.temperature,
                loss_types=args.loss_types,
                loss_weights=args.loss_weights,
                add_origin_loss=args.add_origin_loss,
            )
            conf = DistillationConfig(teacher_model=model, criterion=distil_loss)
            compression_manager = prepare_compression(copy.deepcopy(model), conf)
            model = compression_manager.model
            train_func(model, compression_manager, trial)
            compression_manager.save(args.output_model)
            accu = eval_func(model, compression_manager)
            return accu

        study = optuna.create_study(
            study_name="self_distillation_study",
            storage="sqlite:///distillation.db",
            load_if_exists=True,
            direction="maximize",
        )
        study.optimize(objective, n_trials=2)
        best_trial = study.best_trial
        logger.info(f"best trial: {best_trial}")
    return


if __name__ == "__main__":
    main()
