For working with __GITHUB__ check __github_commands.txt__

# Liver Metastasis MRI Segmentation

Code release accompanying the paper *"Impact of Annotation Style and Dataset Size on CNN-based MRI Segmentation of Colorectal Liver Metastasis"*.

__1. Create an env__
Create a Conda env
*Python==3.8+
GPU
conda install -c anaconda cudatoolkit
conda install -c anaconda cudnn
*Pytorch==1.9.0+ (https://pytorch.org/get-started/previous-versions/)

pip install -r requirements.txt

__2. Organize the data__
```
├── img/            # MRI volumes
│   ├── train/
│   │   ├──001_seg.nii.gz
│   ├── valid/
│   └── test/
├── seg/            # GT with THE SAME names as in img/
│   ├── train/
│   │   ├──001_seg.nii.gz
│   ├── valid/
│   └── test/
```

__3. Create a .yaml file__
(e.g. `yaml_files/configuration_template.yaml`)
- specify where the data is (`data.image`, `data.label`)
- where to save checkpoints and logs (`log.save_dir`)
- specify the network (`net`, e.g. `networks.dedc_net.VGGResYNetED3D`)
- specify the loss function, optimizer and lr
- transformations and patch size for train/valid/inference

__4. Run *main_segm.py*__ (or *main_batch.py* for batch/server runs)
```
    4.1 For training
    $ python main_segm.py -c yaml_files/configuration_template.yaml --train

    4.2 To restore training
    $ python main_segm.py -c yaml_files/configuration_template.yaml --restore -w checkpoint_name.ckpt

    4.3 Test
    $ python main_segm.py -c yaml_files/configuration_template.yaml --test -w checkpoint_name.ckpt

    4.4 Inference
    $ python main_segm.py -c yaml_files/configuration_template.yaml --inference -w checkpoint_name.ckpt -in_dir path_to_folder

    could also specify where to save with -O / --output-dir
```

## Repository contents

| File | Purpose |
|---|---|
| `main_segm.py` | Main entry point: train / test / infer / restore, driven by a `.yaml` config |
| `main_batch.py` | Batch/server variant of the same training-and-inference pipeline |
| `set_up.py` | PyTorch Lightning `Framework` module: dataset loading, training/validation steps |
| `set_inf.py` | Inference pipeline: sliding-window prediction, post-processing, volume export |
| `utils.py` | Config helpers (dynamic class loading from the `.yaml` config) |
| `networks/dedc_net.py` | DEDC-Net architecture (`VGGResYNetED3D`): dual-encoder dual-decoder network |
| `tumor_dilation.py` | Controlled spherical dilation of tumor masks (label 2), used for the annotation-style-harmonization experiments |
| `evaluation.py` | Per-volume prediction saving and TensorBoard visualization helpers |
| `postprocessing.py` / `postprocess_predictions.py` | Keep the largest connected component for liver; keep only tumor components inside the liver mask |
| `round_labels.py` | Round soft/interpolated label volumes back to integer class labels |
| `Dice_check.py` | Per-tumor Dice and Hausdorff-distance computation (a tumor counted as detected if overlap exceeds 10%) |
| `dataset_statistics.py` | Acquisition-parameter, volume, and lesion-size statistics for the dataset description |
| `evaluate_all_experiments.py` | Aggregate evaluation across multiple dataset-size experiment subfolders |

## Citation

If you use this code, please cite the associated paper (details to be added upon publication).
