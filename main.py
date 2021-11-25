import logging
import optuna
import torch
import sys
from wandb_log import WeightsAndBiasesCallback
from objective import objective
from utils import get_best_trial_with_condition
import argparse
import os


def tune(
        project_name: str, 
        data_yaml: str, 
        model_yaml: str,
    ):
    #if not torch.cuda.is_available():
    #    device = torch.device("cpu")
    #elif 0 <= gpu_id < torch.cuda.device_count():
    #    device = torch.device(f"cuda:{gpu_id}")
    # import pdb;pdb.set_trace()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = optuna.samplers.MOTPESampler()
    #if storage is not None:
    #    rdb_storage = optuna.storages.RDBStorage(url=storage)
    #else:
    #    rdb_storage = None
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage_name = f"sqlite:///./exp/{project_name}/{project_name}.db".format(project_name)
    study = optuna.create_study(
        directions=["maximize", "minimize", "minimize"],
        study_name=project_name,
        sampler=sampler,
        storage=storage_name,
        load_if_exists=True,
    )
    wandb_kwargs = {"project": project_name, "reinit": True}
    wandbc = WeightsAndBiasesCallback(["f1_score", "params_num", "mean_time"],wandb_kwargs=wandb_kwargs)
    study.optimize(lambda trial: objective(trial, data_yaml, './exp/'+project_name+'/'+project_name+'.pt', device, model_yaml), n_trials=100, callbacks=[wandbc])

    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trials:")
    best_trials = study.best_trials

    ## trials that satisfies Pareto Fronts
    for tr in best_trials:
        print(f"  value1:{tr.values[0]}, value2:{tr.values[1]}")
        for key, value in tr.params.items():
            print(f"    {key}:{value}")
            
    best_trial = get_best_trial_with_condition(study)
    print(best_trial)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--data', type=str, default='/opt/ml/code/configs/data/taco.yaml')
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()
    dirName = './exp/'+args.project
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")
    tune(
        args.project,
        args.data,
        args.model
    )