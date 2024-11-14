# cRAG: Collective Retrieval Augmented Generation: A Multi-Agent LLM Framework for Distributed Knowledge Synthesis in Scientific Literature
Contributors: Toby Nguyen, Luis Pimental, Stanley Wong, Allen Zhang


## Installation

### Cloning the Repository
Login to PACE:
```bash
ssh <GT_USERNAME>@login-ice.pace.gatech.edu
salloc --gres=gpu:H100:1 --ntasks-per-node=1 --time=1:00:00
```

Setup SSH keys:
```bash
ssh-keygen -t ed25519 -C "<your_email@example.com>"
```
Click enter twice to accept default file location
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```
- Copy the output (it should start with "ssh-ed25519" and end with your email) and go to your Github.com -> Settings -> SSH and GPG Keys
- Click "New SSH key"
- Paste your key into the "Key" field
- Click "Add SSH key"

Test your connection
```bash
ssh -T git@github.com
```
It should say something like ""Hi username! You've successfully authenticated..."

Clone the repo:
```bash
git clone git@github.com:tobynguyen03/cRAG.git
```

### Install environment and dependencies:
```bash
cd cRAG
module load anaconda3
conda env create -f environment.yaml
conda activate cRAG
cd paper-qa
pip install -e .
```


### Install ollama manually
Download binaries:
```bash
curl -L https://github.com/ollama/ollama/releases/download/v0.4.1/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
mkdir -p ~/.local
tar -C ~/.local -xzf ollama-linux-amd64.tgz
```

Open `~/.bashrc`:

```bash
nano ~/.bashrc
```

Add the following:

```bash 
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/lib/ollama:$LD_LIBRARY_PATH
export OLLAMA_MODELS=/home/hice1/<GT_USERNAME>/scratch/ollama_models/ # path to your scratch directory where models will be stored
```

Run the following to update changes:

```bash
source ~/.bashrc
```

Verify
```bash 
ollama serve # run this one terminal
ollama -v # run this in another
```

Dowload ollama models:
```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
```

### Example

Start ollama server:
```bash 
ollama serve
```
In seperate terminal:
```bash 
python testing_paperqa_ollama.py
```



#### Debugging

- https://github.com/Future-House/paper-qa/issues/582#issuecomment-2418906436
- https://github.com/ollama/ollama/issues/7421
- https://github.com/ollama/ollama/issues/680


