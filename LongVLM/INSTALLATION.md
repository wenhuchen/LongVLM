## 🛠️ Installation

- Clone this repository:

  ```bash
  git clone https://github.com/OpenGVLab/V2PE.git
  ```
  
- Create a conda virtual environment and activate it:

  ```bash
  conda create -n v2pe python=3.9 -y
  conda activate v2pe
  ```

- Install dependencies using `requirements.txt`:

  ```bash
  pip install -r requirements.txt
  ```
  
### Additional Instructions

- Install `flash-attn==2.3.6`:

  ```bash
  pip install flash-attn==2.3.6 --no-build-isolation
  ```

  Alternatively you can compile from source:

  ```bash
  git clone https://github.com/Dao-AILab/flash-attention.git
  cd flash-attention
  git checkout v2.3.6
  python setup.py install
  ```

