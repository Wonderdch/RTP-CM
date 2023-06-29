import argparse


def parse():
    parser = argparse.ArgumentParser(description="Run RTP-CM.")
    parser.add_argument('--name',
                        type=str,
                        default='Test_RTP_CM',
                        help='Results name')
    parser.add_argument('--run-times',
                        type=int,
                        default=1,
                        help='Run times')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='Device, cpu or cuda:x')

    # Data
    parser.add_argument('--dataset',
                        type=str,
                        default='NYC',
                        help='Dataset name')

    # Model hyper-parameters
    parser.add_argument('--mask-strategy',
                        type=int,
                        default=2,
                        help='Masking strategy, 0:Simple, 1:Living, 2:Auto or 3:Prediction')
    parser.add_argument('--mask-proportion',
                        type=float,
                        default=0.5,
                        help='Masking proportion')
    parser.add_argument('--area-proportion',
                        type=float,
                        default=0.2,
                        help='Area auxiliary loss proportion')
    parser.add_argument('--embed-size',
                        type=int,
                        default=60,
                        help='Check-in feature embedding dimensions')
    parser.add_argument('--transformer-layers',
                        type=int,
                        default=1,
                        help='Num of Transformer encoder layer')
    parser.add_argument('--transformer-heads',
                        type=int,
                        default=1,
                        help='Num of heads of Transformer encoder')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.2,
                        help='Dropout proportion for encoder')

    # Training hyper-parameters
    parser.add_argument('--epochs',
                        type=int,
                        default=40,
                        help='Number of epochs to train')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-5,
                        help='Learning rate')

    return parser.parse_args()
