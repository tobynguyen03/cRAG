# cRAG: Collective Retrieval Augmented Generation: A Multi-Agent LLM Framework for Distributed Knowledge Synthesis in Scientific Literature
Contributors: Toby Nguyen, Luis Pimental, Stanley Wong, Allen Zhang


## Installation

Install environment and dependencies:
```bash 
conda env create -f environment.yaml
conda env activate cRAG
cd paper-qa
pip install -e .
```


### Install ollama manually
Dowload binaries:
```bash
curl -L https://github.com/ollama/ollama/releases/download/v0.4.1/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
mkdir -p ~/.local
tar -C ~/.local -xzf ollama-linux-amd64.tgz
```

Add the following to your `~/.bashrc`:

```bash 
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/lib/ollama:$LD_LIBRARY_PATH
export OLLAMA_MODELS=/home/hice1/gtusername/scratch/ollama_models/ # path to your scratch directory where models will be stored
```

Verify
```bash 
ollama serve # run this one terminal
ollama -v # run thi in another
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


