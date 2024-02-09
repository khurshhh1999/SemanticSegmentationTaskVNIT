import torch
import resource
import argparse
try:
    from utils.common import warnings
except ImportError:
    import warnings

# torch.multiprocessing.set_sharing_strategy('file_system')

from utils.args import parse_arg_cfg, read_config, map_states, add_shortcuts, cmd_dict
from utils.runners import LaneDetTrainer, LaneDetTester


if __name__ == '__main__':
    # ulimit
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

    # Settings (user input > config > argparse defaults)
    parser = argparse.ArgumentParser(description='PytorchAutoDrive Lane Detection', conflict_handler='resolve')
    add_shortcuts(parser)

    # Required args
    parser.add_argument('--config', type=str, help='Path to config file', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--test', action='store_true')
    group.add_argument('--val', action='store_true')
    group.add_argument('--fastval', action='store_true')
    group.add_argument('--state', type=int,
                       help='[Deprecated] validation(3)/final test(2)/fast validation(1)/training(0)')

    # Optional args/to overwrite configs
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--continue-from', type=str,
                        help='[Deprecated] Continue training from a previous checkpoint')
    group2.add_argument('--checkpoint', type=str,
                        help='Continue/Load from a previous checkpoint')

    parser.add_argument('--mixed-precision', action='store_true',
                        help='Enable mixed precision training')
    parser.add_argument('--cfg-options', type=cmd_dict,
                        help='Override config options with \"x1=y1 x2=y2 xn=yn\"')

    states = ['train', 'fastval', 'test', 'val']
    retain_args = ['state', 'mixed_precision']

    args = parser.parse_args()
    if args.state is not None:
        warnings.warn('--state={} is deprecated, it is recommended to specify with --{}'.format(
            args.state, states[args.state]))
    args.state = map_states(args, states)
    if args.mixed_precision and torch.__version__ < '1.6.0':
        warnings.warn('PyTorch version too low, mixed precision training is not available.')

    # Parse configs and execute runner
    cfg = read_config(args.config)
    cfg_runner_key = 'train' if args.state == 0 else 'test'
    Runner = LaneDetTrainer if args.state == 0 else LaneDetTester
    args, cfg = parse_arg_cfg(args, cfg)
    for k in retain_args:
        cfg[cfg_runner_key][k] = vars(args)[k]
    runner = Runner(cfg=cfg)
    runner.run()
    runner.clean()
