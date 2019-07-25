import json
import yaml
from easydict import EasyDict
import os

def get_config_from_json(json_file):
    """
    Get the config from a json file
    Input:
        - json_file: json configuration file
    Return:
        - config: namespace
        - config_dict: dictionary
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config, config_dict

def get_config_from_yaml(yaml_file):
    """
    Get the config from yaml file
    Input:
        - yaml_file: yaml configuration file
    Return:
        - config: namespace
        - config_dict: dictionary
    """

    with open(yaml_file) as fp:
        config_dict = yaml.load(fp)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)
    return config, config_dict

def process_config(config_file):
    if config_file.endswith('json'):
        config, _ = get_config_from_json(config_file)
    elif config_file.endswith('yaml'):
        config, _ = get_config_from_yaml(config_file)
    else:
        raise Exception("Only .json and .yaml are supported!")

    config.cache_dir = os.path.join("cache", config.exp_name)
    config.model_dir = os.path.join("cache", config.exp_name, 'models')
    config.log_dir = os.path.join("cache", config.exp_name, 'logs')
    config.img_dir = os.path.join("cache", config.exp_name, 'imgs')
    return config
