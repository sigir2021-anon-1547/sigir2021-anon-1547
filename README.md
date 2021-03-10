# Code repository containing code to Shor Paper submitted to SIGIR 2021. Paper ID: 1547


## Get Started

The whole model is implemented in PyTorch-Lightning framework.

1. `cd` to a directory where you want to clone this repo
2. Run `git clone https://github.com/sigir2021-anon-1547/sigir2021-anon-1547`
3. Install required packages `pip install -r requirements.txt`
4. Compile Cython evaluation code
    ```bash
    cd utils/eval_cython
    python setup.py build_ext --inplace
    ```
5. Download pre-trained weights into `models/` directory for:
    - Resnet50 from here: [[link]](https://download.pytorch.org/models/resnet50-19c8e357.pth)
    - Resnet50-IBN-A from here: [[link]](https://drive.google.com/open?id=1_r4wp14hEMkABVow58Xr4mPg7gvgOMto)

6. Prepare datasets:

    Market1501

    * Extract dataset and rename to `market1501` inside `/data/`
    * The data structure should be following:

    ```bash
    /data
        market1501
            bounding_box_test/
            bounding_box_train/
            ......
    ```
    DukeMTMC-reID

    * Extract dataset to `/data/` directory
    * The data structure should be following:

    ```bash
    /data
        DukeMTMC-reID
           	bounding_box_test/
           	bounding_box_train/
           	......
    ```

    Street2Shop & Deep Fashion (Consumer-to-shop)

    1. These fashion datasets requires the annotation data in COCO-format witha additional fields in `annotations`
        ```
        JSON:{
            'images' : [...],
            'annotations': [
                        {...,
                        'pair_id': 100,         # an int type
                        'source': 'user'        # 'user' or 'shop'
                        },
                        ...
                    ]
        }
        ```
    2. The product images should be pre-cropped to the given input format (either 256x128 or 320x320) using original images and provided bounding boxes to allow faster training.

    ### Path to the data root and JSON files (only for Street2shop and Deep Fashion) can be adjusted by passing the paths as parameters to train scripts
    ### You can familiarize yourself with the detailed configuration and its meaning in `config.defaults.py`, which includes all parameters available to the user.

## Train
Each Dataset and Model has its own train script.  
All train scripts are in `train_scirpts` folder with corresponding dataset name.

Example run command to train CTL-Model on DukeMTMC-reID
```bash
CUDA_VISIBLE_DEVICES=3 ./train_scripts/dukemtmc/train_ctl_model_s_r50_dukemtmc.sh
```
`CUDA_VISIBLE_DEVICES` controls which GPUs are visible to the scripts.  
`GPU_IDS` parameter in train scripts allows to adjust the number of used GPUs for the given training.

By default all train scripts will launch 3 experiments.

## Test
To test the trained model you can use provided scripts in `train_scripts`, just two parameters need to be added:  
    
    TEST.ONLY_TEST \  
    MODEL.PRETRAIN_PATH "path/to/pretrained/model/checkpoint.pth"
    
Example train script for testing trained CTL-Model on Market1501
```bash
python train_ctl_model.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0] \
DATASETS.NAMES 'market1501' \
DATASETS.ROOT_DIR '/data/' \
SOLVER.IMS_PER_BATCH 16 \
TEST.IMS_PER_BATCH 128 \
SOLVER.BASE_LR 0.00035 \
OUTPUT_DIR './logs/market1501/256_resnet50/' \
SOLVER.EVAL_PERIOD 40 \
TEST.ONLY_TEST \
MODEL.PRETRAIN_PATH "logs/market1501/256_resnet50/train_ctl_model/version_0/checkpoints/epoch=119.ckpt"
```