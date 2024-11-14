# cRAG: Collective Retrieval Augmented Generation: A Multi-Agent LLM Framework for Distributed Knowledge Synthesis in Scientific Literature
Contributors: Toby Nguyen, Luis Pimental, Stanley Wong, Allen Zhang


## Installation
Login to PACE:

```bash
ssh <GT_USERNAME>@login-ice.pace.gatech.edu
```

Install environment and dependencies:
```bash 
conda env create -f environment.yaml
conda env activate cRAG
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


