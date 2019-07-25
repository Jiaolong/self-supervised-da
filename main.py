import random
from utils.dirs import create_dirs
from utils.config import process_config
from utils.utils import get_args, get_logger

from models.aux_model import AuxModel
from data.data_loader import get_train_val_dataloader
from data.data_loader import get_target_dataloader
from data.data_loader import get_test_dataloader

def main():
    args = get_args()
    config = process_config(args.config)

    # create the experiments dirs
    create_dirs([config.cache_dir, config.model_dir,
        config.log_dir, config.img_dir])

    # logging to the file and stdout
    logger = get_logger(config.log_dir, config.exp_name)
    
    # fix random seed to reproduce results
    random.seed(config.random_seed)
    logger.info('Random seed: {:d}'.format(config.random_seed))

    if config.method in ['src', 'jigsaw', 'rotate']:
        model = AuxModel(config, logger)
    else:
        raise ValueError("Unknown method: %s" % config.method)

    src_loader, val_loader = get_train_val_dataloader(config.datasets.src)
    test_loader = get_test_dataloader(config.datasets.test)

    tar_loader = None
    if config.datasets.get('tar', None):
        tar_loader = get_target_dataloader(config.datasets.tar)

    if config.mode == 'train':
        model.train(src_loader, tar_loader, val_loader, test_loader)
    elif config.mode == 'test':
        model.test(test_loader)

if __name__ == '__main__':
    main()
