import argparse
from pathlib import Path
import torch

import models

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('model',
                           help=f'Model Name. One of: {models.__all__}')
    argparser.add_argument('--model-path', type=str)
    argparser.add_argument('--inp-size', type=int, default=1250)

    args = argparser.parse_args()
    # import pdb; pdb.set_trace()

    # pytorch2onnx(args.model_path, args.model, args.inp_size)
    net = torch.load(args.model_path, map_location=torch.device('cpu'))

    dummy_input = torch.randn(1, 1, args.inp_size)

    optName = Path(args.model_path).parent / (str(args.model) + '.onnx')
    torch.onnx.export(net, dummy_input, optName, verbose=True)
