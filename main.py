import torch
from models.device import DEVICE
from models.eval import evaluate
from models.model_stats import get_model_size
from models.vgg import VGG

from models.data import get_data_loader
from qtorch.quantization import quantize


Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB


def extra_preprocess(x):
    # convert the original fp32 input of range (0, 1)
    # into int8 format of range (-128, 127)
    x = (x - 0.5) * 255
    return x.clamp(-128, 127).to(torch.int8)


def main():
    checkpoint_path = "models/pretrained_vgg/vgg.cifar.pretrained.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    data_loader = get_data_loader()
    model = VGG().to(DEVICE)

    print(f"=> loading checkpoint '{checkpoint_path}'")
    model.load_state_dict(checkpoint['state_dict'])

    print("\nEvaluating Float Model")
    fp32_model_accuracy = evaluate(model, data_loader['test'])
    fp32_model_size = get_model_size(model)
    print(f"fp32 model has accuracy={fp32_model_accuracy:.2f}%")
    print(f"fp32 model has size={fp32_model_size/MiB:.2f} MiB")

    # Quantizing the model
    sample_data = iter(data_loader['train']).__next__()[0]
    feature_bitwidth = 8
    weight_bitwidth = 8
    quantized_model = quantize(model, feature_bitwidth, weight_bitwidth, sample_data)
    quantized_model_size = get_model_size(model, data_width=8)

    print("\nEvaluating Int8 Model")
    int8_model_accuracy = evaluate(quantized_model, data_loader['test'],
                                   extra_preprocess=[extra_preprocess])
    print(f"int8 model has accuracy={int8_model_accuracy:.2f}%")
    print(f"Int8 model has size={quantized_model_size/MiB:.2f} MiB")


if __name__ == "__main__":
    main()
