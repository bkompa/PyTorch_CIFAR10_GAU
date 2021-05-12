import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from data import CIFAR10Data
from module import CIFAR10Module

import json
import mlflow 
import mlflow.pytorch 


def main(args):

    if bool(args.download_weights):
        CIFAR10Data.download_weights()
    else:
        seed_everything(0)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

        if args.logger == "wandb":
            logger = WandbLogger(name=args.classifier, project="cifar10")
        elif args.logger == "tensorboard":
            logger = TensorBoardLogger("cifar10", name=args.classifier)
        elif args.logger == 'mlflow':
            with open(os.path.join(os.path.expanduser("~"), "mlsecrets.json"), "r") as file:
                mlsecrets = json.load(file)

            os.environ["AWS_ACCESS_KEY_ID"] = mlsecrets['s3_user']
            os.environ["AWS_SECRET_ACCESS_KEY"]= mlsecrets['s3_key']
            os.environ["MLFLOW_TRACKING_USERNAME"] = mlsecrets['user']
            os.environ["MLFLOW_TRACKING_PASSWORD"]= mlsecrets['pwd']

            mlflow.set_tracking_uri(mlsecrets['server'])
            mlflow.set_experiment('cifar10_gau')
            
            mlflow.pytorch.autolog()

        checkpoint = ModelCheckpoint(monitor="acc/val", mode="max", save_last=False)

        trainer = Trainer(
            fast_dev_run=False,
            gpus=-1,
            deterministic=True,
            weights_summary=None,
            log_every_n_steps=1,
            max_epochs=args.max_epochs,
            checkpoint_callback=checkpoint,
            precision=args.precision,
        )

        model = CIFAR10Module(args)
        data = CIFAR10Data(args)

        if bool(args.pretrained):
            state_dict = os.path.join(
                "cifar10_models", "state_dicts", args.classifier + ".pt"
            )
            model.model.load_state_dict(torch.load(state_dict))

        if bool(args.test_phase):
            trainer.test(model, data.test_dataloader())
        else:
            trainer.fit(model, data)
            trainer.test()

        torch.save(model.model.state_dict(), f"{args.model_dir}/{args.model_name}")


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--data_dir", type=str, default="data/cifar10/")
    parser.add_argument("--model_dir", type=str, default="mnt/medqaresourcegroupdiag/medqa-fileshare/users/bk117/models/")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--download_weights", type=int, default=0, choices=[0, 1])
    parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb", "mlflow"]
    )

    # TRAINER args
    parser.add_argument("--classifier", type=str, default="resnet18_RBF")
    parser.add_argument("--pretrained", type=int, default=0, choices=[0, 1])

    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_split", type=int, default=4, choices=[0,1,2,3,4])
    parser.add_argument("--gpu_id", type=str, default="0")

    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    args = parser.parse_args()
    main(args)
