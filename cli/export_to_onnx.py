import argparse

from tools.export_to_onnx import export_to_onnx
from tools.model_loader import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--onnx', required=True)
    args = parser.parse_args()

    export_to_onnx(
        model_loader=load_model,
        weights_path=args.weights,
        onnx_path=args.onnx,
    )


if __name__ == '__main__':
    main()
