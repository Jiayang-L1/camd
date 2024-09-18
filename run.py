import gc
import logging
import os
import time
from pathlib import Path
import torch
from config import get_config
from crowdfunding_dataloader import MMDataLoader
from trains import ATIO
from utils import assign_gpu, setup_seed
from trains.singleTask.model import augment_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
logger = logging.getLogger('CAMD')


def _set_logger(log_dir, model_name, dataset_name, verbose_level):
    # base logger
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
    logger = logging.getLogger('CAMD')
    logger.setLevel(logging.DEBUG)

    # file handler
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # stream handler
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def CAMD_run(
        model_name, dataset_name, config=None, config_file="", seeds=[],
        model_save_dir="", model_path='', log_dir="",
        gpu_ids=[0], num_workers=4, verbose_level=1, mode=''
):
    # Initialization
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()

    if config_file != "":
        config_file = Path(config_file)
    else:  # use default config files
        config_file = Path(__file__).parent / "config" / "config.json"
    if not config_file.is_file():
        raise ValueError(f"Config file {str(config_file)} not found.")

    if model_save_dir == "":
        model_save_dir = Path.home() / "CAMD" / "saved_models"
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    if log_dir == "":
        log_dir = Path.home() / "CAMD" / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    seeds = seeds if seeds != [] else [1, 2, 3, 4, 5]
    logger = _set_logger(log_dir, model_name, dataset_name, verbose_level)

    args = get_config(model_name, dataset_name, config_file)
    args.mode = mode  # train or test
    args['model_save_path'] = Path(model_save_dir) / f"{args['model_name']}-{args['dataset_name']}.pth"
    args['device'] = assign_gpu(gpu_ids)
    if config:
        args.update(config)

    for i, seed in enumerate(seeds):
        setup_seed(seed)
        args['cur_seed'] = i + 1
        _run(args, num_workers, model_path)


def _run(args, num_workers=0, model_path=''):
    dataloader = MMDataLoader(args, num_workers)
    model_augment = getattr(augment_model, 'AUG')(args)
    model_augment = model_augment.to(args.device)

    trainer = ATIO().getTrain(args)

    current_time = int(time.time())
    if args.mode == 'test':
        model_augment.load_state_dict(torch.load(model_path))
        trainer.test(model_augment, dataloader['test'])
    elif args.mode == 'train':
        trainer.train(model_augment, dataloader, current_time=current_time)
        model_augment.load_state_dict(torch.load(f'pt/{current_time}-static.pth'))
        trainer.test(model_augment, dataloader['test'])

        del model_augment
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)
    return
