import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        help='directory path of data')
    parser.add_argument(
        '--data_scaling',
        default='linear',
        type=str,
        help='scaling of data: linear or log')
    parser.add_argument(
        '--train_path',
        type=str,
        help='training filelist(csv) path')
    parser.add_argument(
        '--valid_path',
        type=str,
        help='validation filelist(csv) path')
    parser.add_argument(
        '--test_path',
        type=str,
        help='validation filelist(csv) path')
    parser.add_argument(
        '--result_path',
        default='results',
        type=str,
        help='Result directory path')
    parser.add_argument(
        '--tdim_use',
        default=12,
        type=int,
        help='Temporal duration to be used')
    parser.add_argument(
        '--optimizer',
        type=str,
        help='Optimizer type adam or rmsprop')
    parser.add_argument(
        '--learning_rate',
        default=0.01,
        type=float,
        help='Learning rate')
    parser.add_argument(
        '--lr_decay',
        default=1.0,
        type=float,
        help='Learning rate decay')
    parser.add_argument(
        '--batch_size',
        default=10,
        type=int,
        help='Batch Size')
    parser.add_argument(
        '--n_epochs',
        default=10,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument(
        '--no_val',
        action='store_true',
        help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument(
        '--test',
        action='store_true',
        help='If true, test is performed.')
    parser.set_defaults(test=False)
    parser.add_argument(
        '--eval_threshold',
        default=0.5,
        type=float,
        help='Threshold in [mm/h] for precipitation evaluation')
    parser.add_argument(
        '--n_threads',
        default=4,
        type=int,
        help='Number of threads for multi-thread loading')
    parser.add_argument(
        '--checkpoint',
        default=10,
        type=int,
        help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--manual_seed',
        default=1,
        type=int,
        help='Manually set random seed')
    # ConvLSTM Structure
    parser.add_argument(
        '--hidden_channels',
        default=12,
        type=int,
        help='Number of hidden channels in ConvLSTM.')
    parser.add_argument(
        '--kernel_size',
        default=3,
        type=int,
        help='kernel size in ConvLSTM.')
    
    args = parser.parse_args()

    return args
