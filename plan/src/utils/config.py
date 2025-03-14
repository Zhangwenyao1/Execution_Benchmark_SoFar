import yaml
from os.path import dirname, join
import torch
from rich import print as rprint

get_model = None

# use this so that one can use config.x.y.z instead of config['x']['y']['z']
class DotDict(dict):
    def __getattr__(self, item):
        if item[0] == '_':
            return super().__getattr__(item)
        if item in self.keys():
            return self[item]
        rprint(f'[bold red]Warning: {item} not found in config, set to the default value None.[/bold red]')
        return None
    
    def __setattr__(self, key, value):
        self[key] = value

def to_dot_dict(dic):
    for k in dic.keys():
        if type(dic[k]) == dict:
            dic[k] = to_dot_dict(dic[k])
    return DotDict(dic)

def to_dict(args):
    result = dict()
    for k, v in args.items():
        if isinstance(v, dict):
            result[k] = to_dict(v)
        else:
            result[k] = v
    return result

def add_commons(dic, common):
    for k in dic.keys():
        if isinstance(dic[k], dict) and k != 'common':
            dic[k]['common'] = common
            dic[k] = add_commons(dic[k], common)
    return dic

def remove_commons(dic):
    dic.pop('common')
    for k in dic.keys():
        if isinstance(dic[k], dict):
            dic[k] = remove_commons(dic[k])
    return dic
    
def add_argparse(parser, arg_mapping):
    for raw_key, (_, arg_type, default) in arg_mapping:
        parser.add_argument('--' + raw_key, type=arg_type, default=default)
    return parser

# combine config from yaml file and argument
# priority: args in console > default args > yaml file
def load_config(yaml_file, arg_mapping=None, args=None):
    with open(yaml_file, 'r') as f:
        dic = yaml.load(f, Loader=yaml.FullLoader)
    if args is not None:
        for raw_key, (new_key, _, _) in arg_mapping:
            value = eval(f'args.{raw_key}')
            if value is None:
                continue
            temp = dic
            for k in new_key.split('/')[:-1]:
                if not k in temp.keys():
                    temp[k] = dict()
                elif not type(temp[k]) == dict:
                    raise ValueError
                temp = temp[k]
            temp[new_key.split('/')[-1]] = value
    config = to_dot_dict(dic)
    if 'common' in config.keys():
        config = add_commons(config, config.common)
    return config

def ckpt_to_config(ckpt_path, arg_mapping=None, args=None):
    config_path = join(dirname(dirname(ckpt_path)), 'config.yaml')
    return load_config(config_path, arg_mapping, args)

def load_model(ckpt_path: str, arg_mapping=None, args=None, device: torch.device='cpu', mode: str = 'eval'):
    global get_model
    if get_model is None:
        from src.network.model import get_model as get_model_func
        get_model = get_model_func
    config = ckpt_to_config(ckpt_path, arg_mapping, args)
    model = get_model(config.model)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.to(device)
    if mode == 'eval':
        model.eval()
    elif mode == 'train':
        model.train()
    return config, model