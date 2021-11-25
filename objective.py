import optuna
from typing import Any, Dict
from search import search_hyperparam, search_model
import torch
import torch.nn as nn
import sys
import yaml
sys.path.append('../../')
from src.dataloader import create_dataloader
from src.model import Model
from src.utils.torch_utils import check_runtime
from src.trainer import TorchTrainer, count_model_params
from typing import Any, Dict, List, Tuple, Sequence, Optional, Union

def objective(trial: optuna.trial.Trial, data_yaml:str, model_path:str, device, model_yaml: str = None) -> float:
    model_config: Dict[str, Any] = {}
    with open(data_yaml, 'r') as stream:
        data_config = yaml.safe_load(stream)
    model_config["input_channel"]=3
    img_size = data_config["IMG_SIZE"]
    model_config["INPUT_SIZE"] = [img_size, img_size]
    model_config["depth_multiple"] = 0.6
    model_config["width_multiple"] = 0.6

    if model_yaml is None:
        model_config["backbone"], model_infos = search_model(trial)
    else:
        with open(model_yaml, 'r') as stream:
            parsed_model = yaml.safe_load(stream)
            model_config["backbone"] = parsed_model["backbone"]

    hyperparams = search_hyperparam(trial)

    model = Model(model_config, verbose=False)
    #import pdb;pdb.set_trace()
    try:
        model.model.to(device)
    except RuntimeError:
        torch.cuda.empty_cache()
        return 0,0,0
    #import pdb;pdb.set_trace()
    
    data_config["BATCH_SIZE"] = hyperparams["BATCH_SIZE"]
    
    mean_time = check_runtime(
        model.model,
        [model_config["input_channel"]] + model_config["INPUT_SIZE"],
        device,
    )

    train_loader, val_loader, test_loader = create_dataloader(data_config)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), hyperparams["LEARNING_RATE"], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        steps_per_epoch=len(train_loader),
        epochs=20,
        pct_start=0.05,
    )

    trainer = TorchTrainer(
        model,
        criterion,
        optimizer,
        scheduler,
        device=device,
        verbose=1,
        model_path = model_path,
    )
    #import pdb;pdb.set_trace()
    trainer.train(train_loader, 20, val_dataloader = val_loader)
    loss, f1_score, acc = trainer.test(model, test_dataloader=test_loader)
    params_nums = count_model_params(model)

    return f1_score, params_nums, mean_time