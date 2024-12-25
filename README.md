# V2PE: Improving Multimodal Long-Context Capability of Vision-Language Models with Variable Visual Position Encoding

The official implementation of the paper "[V2PE: Improving Multimodal Long-Context Capability of Vision-Language Models with Variable Visual Position Encoding](https://arxiv.org/abs/2412.09616)". 

<div align="center">
    <img src="assets/fig1_hf_00.png" alt="drawing" width="600"/>
</div>

<div align="center">

[\[🆕 Blog\]](https://zzdhybthu.github.io/V2PE.github.io)  [\[📜 ArXiv Paper\]](https://arxiv.org/abs/2412.09616)  [\[🤗 HF Models\]](https://huggingface.co/OpenGVLab/V2PE)  [\[📖 HF Datasets\]](https://huggingface.co/datasets/OpenGVLab/V2PE-Data)

</div>


## 📖 Summary

The main contributions of this work are as follows:

- We construct mixed datasets for VLMs' long-context training and evaluation by augmenting existing multimodal instruction tuning datasets and conduct a thorough investigation into why current VLMs struggle with long-context multimodal inputs, revealing that directly applying LLM positional encoding to visual tokens is ineffective. 
- We propose Variable Visual Position Encoding (V2PE), a novel positional encoding strategy that employs variable and smaller increments for visual tokens, significantly enhancing VLMs' ability to understand and reason over long multimodal contexts.
- We apply our V2PE method and extend training data on the open-source VLM, InternVL2-2B. The fine-tuned VLM performs exceptionally well on both general multimodal benchmarks and long-context multimodal tasks, with the capacity to handle sequences of up to 1M tokens.

## 🛠️ Installation

See [INSTALLATION.md](INSTALLATION.md)

In addition, using this codebase requires executing the following steps:

- Install other requirements:

  ```bash
  pip install --upgrade pip  # enable PEP 660 support
  pip install -e .
  ```

## 📦 Model Preparation

Our models are built from InternVL2-2B.
Please download the above model weights and place them in the `pretrained/` folder.

```sh
cd pretrained/
# pip install -U huggingface_hub
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-8B --local-dir InternVL2-8B
```

The directory structure is:

```sh
pretrained
└── InternVL2-8B/
```

## 🔥 Supervised Fine-tuning

### Start Training

We provide slurm scripts for multi-node multi-GPU training. You can use 32 GPUs to train this model, and it will take approximately 48 hours.

```sh
# using 32 GPUs
PARTITION='your partition' GPUS=32 sh shell/internlm2_2b/internvl_chat_v2_internlm2_2b_dynamic_res_v2pe_32k.sh
```
### Training using ring-attention

When training on 256k length or longer dataset, you may need using [ring attention](https://github.com/zhuzilin/ring-flash-attention.git) to limit GPU memory usage. To use ring attention, you need to set two variables in the training script:
```bash
  --chunk_num 8 \
  --attn_type 'ring' \
```
Here, chunk_num specifies the number of chunks each sample is split into, which are distributed across chunk_num GPUs. The use_chunkTrainer flag indicates that ring attention is used during training.

We provide an example training script that utilizes ring attention at:
shell/internlm2_2b/internvl_chat_v2_internlm2_2b_dynamic_res_v2pe_256k.sh.
You can run this script with the following command:
```bash
# using 32 GPUs
PARTITION='your partition' GPUS=32 sh shell/internlm2_2b/internvl_chat_v2_internlm2_2b_dynamic_res_v2pe_256k.sh
```

## ❓ How to Evaluate

### Preparing General MLLM Benchmarks


#### [ChartQA test-human & test-augmented](https://aclanthology.org/2022.findings-acl.177/)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/chartqa && cd data/chartqa

# download images from https://drive.google.com/file/d/1Lm_w6zeET1Hyl_9ks6w5nEsgpoyPHalV/view

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/chartqa/train_human.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/chartqa/train_augmented.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/chartqa/test_human.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/chartqa/test_augmented.jsonl

cd ../..
```

</details>

#### [DocVQA val & test](https://www.docvqa.org/datasets)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/docvqa && cd data/docvqa

# download images and annotations
wget https://datasets.cvc.uab.es/rrc/DocVQA/train.tar.gz --no-check-certificate # (optional)
wget https://datasets.cvc.uab.es/rrc/DocVQA/val.tar.gz --no-check-certificate
wget https://datasets.cvc.uab.es/rrc/DocVQA/test.tar.gz --no-check-certificate

# unzip files
tar -zxvf train.tar.gz
tar -zxvf val.tar.gz
tar -zxvf test.tar.gz

# download converted jsonl files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/docvqa/train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/docvqa/val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/docvqa/test.jsonl
cd ../..
```

</details>

#### [AI2D test](https://allenai.org/data/diagrams)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/ai2diagram && cd data/ai2diagram
# download converted files
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/ai2d_test_vlmevalkit.jsonl -O test_vlmevalkit.jsonl
wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/AI2D_TEST.zip && unzip AI2D_TEST.zip

# download images from Google drive (optional, provided by InternLM-XComposer)
# https://drive.google.com/file/d/1dqqa3MnrxMXaU_K9JA6C83je32ibwdOY/view?usp=sharing
# images should be placed in `data/ai2diagram/ai2d/abc_images` and `data/ai2diagram/ai2d/images`
cd ../..
```

</details>

#### [InfoVQA](https://rrc.cvc.uab.es/?ch=17)

<details>
<summary>Data Preparation</summary>

Please refer to https://rrc.cvc.uab.es/?ch=17 for details

</details>

#### [ScienceQA test](https://github.com/lupantech/ScienceQA)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/scienceqa/images && cd data/scienceqa/images

# download images
wget https://scienceqa.s3.us-west-1.amazonaws.com/images/test.zip && unzip test.zip

cd ..

# download original questions
wget https://github.com/lupantech/ScienceQA/blob/main/data/scienceqa/problems.json

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/scienceqa/scienceqa_test_img.jsonl

cd ../..
```

</details>

#### [POPE](https://github.com/AoiDragon/POPE/tree/main)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/pope && cd data/pope

# make sure you have downloaded COCO images
ln -s ../coco/val2014 ./
wget https://github.com/OpenGVLab/InternVL/releases/download/data/llava_pope_test.jsonl

# download `coco` from POPE
mkdir -p coco && cd coco
wget https://github.com/AoiDragon/POPE/raw/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_adversarial.json
wget https://github.com/AoiDragon/POPE/raw/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_popular.json
wget https://github.com/AoiDragon/POPE/raw/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_random.json
cd ../../..
```

</details>

#### [MMMU](MMMU_validation_240124181104.json)

<details>
<summary>Data Preparation</summary>

The evaluation code will automatically download the dataset from huggingface.

</details>

#### [MMBench dev & test](https://github.com/open-compass/mmbench)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/mmbench && cd data/mmbench

# download csv files of mmbench
wget http://opencompass.openxlab.space/utils/MMBench/CCBench_legacy.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_en_20231003.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_test_cn_20231003.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_test_en_20231003.tsv

cd ../..
```

</details>

#### [SEED](https://github.com/AILab-CVC/SEED-Bench/)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/SEED && cd data/SEED
# 1. Follow the official instructions [Data Preparation for SEED-Bench-1](https://github.com/AILab-CVC/SEED-Bench/blob/main/DATASET.md#data-preparation-for-seed-bench-1)
#    to download the images and the videos. Put images under `./data/SEED/SEED-Bench-image`.
# 2. Extract the video frame in the middle from the downloaded videos, and put them under `./data/SEED/SEED-Bench-image`.
#    LLaVA provided the script [`extract_video_frames.py`](../internvl_chat/tools/extract_video_frames.py) modified from the official one.

wget https://huggingface.co/OpenGVLab/InternVL/raw/main/seed.jsonl
cd ../..
```

</details>

### Preparing Long-Context MLLM Benchmarks

#### [MM-NIAH](https://huggingface.co/datasets/OpenGVLab/MM-NIAH/tree/main)

<details>
<summary>Data Preparation</summary>

1. Download MM-NIAH dataset from [HuggingFace](https://huggingface.co/datasets/OpenGVLab/MM-NIAH/tree/main) and put the files in `dataset/benchmark/MM-NIAH` folder.

2. Unzip images using the following command

    ```bash
    tar -xzvf dataset/benchmark/MM-NIAH/mm_niah_test/images.tar.gz -C dataset/benchmark/MM-NIAH/mm_niah_test/
    tar -xzvf dataset/benchmark/MM-NIAH/mm_niah_val/annotations.tar.gz -C dataset/benchmark/MM-NIAH/mm_niah_val/
    ```
3. The directory structure should look like this:

    ```none
    dataset
    └── benchmark
        └── MM-NIAH
            ├── mm_niah_test
            │   ├── annotations/
            │   └── images/
            └── mm_niah_val/
                ├── annotations/
                └── images/
    ```

</details>

#### [Milebench](https://huggingface.co/datasets/FreedomIntelligence/MileBench/tree/main)

<details>
<summary>Data Preparation</summary>

1. Download milebench dataset from [hugging face](https://huggingface.co/datasets/FreedomIntelligence/MileBench/tree/main)

2. Unzip them using the following command
    
    ```bash
    for file in MileBench_part*.tar.gz
    do
    tar -xzvf "$file"
    done
    ```

3. Put the unzipped files in `dataset/benchmark/MileBench` folder. The directory structure should look like this:

    ```none
    dataset
    └── benchmark
        └── MileBench
            ├── ActionLocalization
            │   ├── images/
            │   └── ActionLocalization.json
            ├── ActionPrediction
            │   ├── images/
            │   └── ActionPrediction.json 
            |── ActionSequence
            │   ...
    ```

</details>

### Evaluation Steps

#### Evaluating General MLLM Benchmarks

<details>
<summary>Evaluation</summary>

For all general MLLM benchmarks, you can only run this one scripts to get all results.

```bash
# use STRIDE=64 as an example
STRIDE=64 sh scripts/evaluate_auto.sh <checkpoint> --rope_pos_id_version v2pe_fix --rope_pos_id_stride 64
```

</details>

#### Evaluating Long-Context MLLM Benchmarks

<details>
<summary>Evaluation for milebench</summary>

```bash
# use STRIDE=64 as an example
STRIDE=64 sh scripts/evaluate_milebench.sh <checkpoint> --rope_pos_id_version v2pe_fix --rope_pos_id_stride 64
```

</details>


<details>
<summary>Evaluation for mm_niah</summary>

```bash
# use STRIDE=64 as an example
STRIDE=64 sh scripts/evaluate_mmniah.sh <checkpoint> --rope_pos_id_version v2pe_fix --rope_pos_id_stride 64
```

</details>
<details>
<summary>Evaluation for mm_niah-1M</summary>

```bash
# use STRIDE=64 as an example
STRIDE=64 sh scripts/evaluate_mmniah_long.sh <checkpoint> --rope_pos_id_version v2pe_fix --rope_pos_id_stride 64
```

</details>

<details>
<summary>Evaluation for long-vqa</summary>

```bash
# use STRIDE=64 as an example
STRIDE=64 GROUP=32 GPUS_PER_TASK=1 sh scripts/evaluate_longvqa.sh <checkpoint> --rope_pos_id_version v2pe_fix --rope_pos_id_stride 64
STRIDE=64 GROUP=40 GPUS_PER_TASK=2 sh scripts/evaluate_longvqa.sh <checkpoint> --rope_pos_id_version v2pe_fix --rope_pos_id_stride 64
STRIDE=64 GROUP=48 GPUS_PER_TASK=2 sh scripts/evaluate_longvqa.sh <checkpoint> --rope_pos_id_version v2pe_fix --rope_pos_id_stride 64
STRIDE=64 GROUP=56 GPUS_PER_TASK=4 sh scripts/evaluate_longvqa.sh <checkpoint> --rope_pos_id_version v2pe_fix --rope_pos_id_stride 64
STRIDE=64 GROUP=64 GPUS_PER_TASK=4 sh scripts/evaluate_longvqa.sh <checkpoint> --rope_pos_id_version v2pe_fix --rope_pos_id_stride 64
```

</details>

## 🎫 License

This project is released under the [MIT License](LICENSE).

## 🖊️ Citation

If you find this work helpful in your research, please consider citing:

```bibtex
@misc{ge2024v2peimprovingmultimodallongcontext,
      title={V2PE: Improving Multimodal Long-Context Capability of Vision-Language Models with Variable Visual Position Encoding}, 
      author={Junqi Ge and Ziyi Chen and Jintao Lin and Jinguo Zhu and Xihui Liu and Jifeng Dai and Xizhou Zhu},
      year={2024},
      eprint={2412.09616},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.09616}, 
}
```
