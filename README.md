This is my fork of the InternVLM architecture.

## Installation

Install the latest version of Pytorch and FlashAttention.

```bash
sudo apt-get install ffmpeg libsm6 libxext6  -y
pip install -r requirements.txt
```

## Training and Evaluation

1. If you are training 4K VLMs, open the ShortVLM/ folder.
2. If you are training 32K VLMs, open the LongVLM/ folder.
3. If you are evaluating VLMs, open the VLMEvalKit/ folder.

## Others

1. The datasets are stored in dataset/
2. The models are stored in work_dirs/