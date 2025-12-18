# Binary Event-Driven Spiking Transformer (IJCAI)

## Highlights

- **Pure Binary Spiking Computation**: All internal activations are binary spike events.
- **Fully Event-Driven Attention**
- **Binary Weight**

------

## Reference

If you find this repository useful, please consider citing:

```bibtex
@inproceedings{ijcai2025p458,
  title     = {Binary Event-Driven Spiking Transformer},
  author    = {Cao, Honglin and Zhou, Zijian and Wei, Wenjie and Liang, Yu and Belatreche, Ammar and Zhang, Dehao and Zhang, Malu and Yang, Yang and Li, Haizhou},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on
               Artificial Intelligence, {IJCAI-25}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {James Kwok},
  pages     = {4110--4118},
  year      = {2025},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2025/458},
  url       = {https://doi.org/10.24963/ijcai.2025/458},
}
```

------

## Requirements

```bash
timm==0.6.12
cupy==11.4.0
torch==1.12.1
spikingjelly==0.0.0.0.12
pyyaml
```

------

## Dataset Preparation

### ImageNet-1K

The ImageNet dataset should be organized as follows:

```
imagenet/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_10026.JPEG
│   │   ├── ...
│   ├── ...
├── val/
│   ├── n01440764/
│   │   ├── ILSVRC2012_val_00000293.JPEG
│   │   ├── ...
│   ├── ...
```

You may extract ImageNet using this script:
https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4

------

## Training

### Training on ImageNet

Configure hyper-parameters in `imagenet.yml`:

```bash
cd imagenet
./train.sh
```

### Training on CIFAR

Configure hyper-parameters in `cifar10.yml` or `cifar100.yml`. You can download the teacher checkpoint from [Release model_best.pth-16.tar && model_best.pth-100.tar · CaoHLin/BESTFormer](https://github.com/CaoHLin/BESTFormer/releases/tag/Teacher_checkpoint_for_CIFAR_training):

```bash
cd cifar
./train.sh
```

------

------

## Acknowledgement

This project is built upon and inspired by:

- Spikingformer
- Spikformer
- SpikingJelly

