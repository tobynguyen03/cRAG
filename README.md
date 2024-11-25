# cRAG: Collective Retrieval Augmented Generation: A Multi-Agent LLM Framework for Distributed Knowledge Synthesis in Scientific Literature
Contributors: Toby Nguyen, Luis Pimental, Stanley Wong, Allen Zhang


## Installation

### Cloning the repository
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


## Using SciQAG

### Getting the SciQAG data
The SciQAG data is available for download at the [SciQAG GitHub](https://github.com/MasterAI-EAM/SciQAG/tree/master/data).

There are three datasets: 
- `final_all_select1000.json` contains all 22,743 entries in the SciQAG data, including full paper texts and associated question-answer pairs. This dataset is generally referred to as "final."
- `train_qas_179511.json` contains the training set, with only question-answer pairs. This dataset is referred to as "train."
- `test_qas_8531.json` contains the test set, with only question-answer pairs. This dataset is referred to as "test."

You can download these three datasets to your desired directory.

### Extracting SciQAG papers
Extracting the papers present in the SciQAG data can be done with `sciqag_utils.py` and the `final_all_select1000.json` data file. Simply initialize the `SciQAGData` class with `"final_all_select1000.json"` as a parameter, call the `SciQAGData.load_data` class function, and call the `SciQAGData.extract_papers_text_from_final` class function with the directory you want the papers to be placed in. The directory can be already existing.

```python
sciqag_data = SciQAGData("final_all_select1000.json")
sciqag_data.load_data
sciqag_data.extract_papers_text_from_final("DESIRED_DIRECTORY")
```

### Extracting SciQAG question-answer pairs
The question-answer pairs in the three SciQAG datasets can be extracted with their respective class functions. The resulting `SciQAGData.qa_pairs` class variable will contain a list of the all the question-answer pairs as dictionaries, where the question is the key and the answer is the value.

```python
data_full = SciQAGData("final_all_select1000.json")
data_full.load_data
data_full.extract_qa_pairs_from_final()

full_qa_pairs = data_full.qa_pairs

data_train = SciQAGData("train_qas_179511.json")
data_train.load_data
data_train.extract_qa_pairs_from_train()

train_qa_pairs = data_train.qa_pairs

data_test = SciQAGData("test_qas_8531.json")
data_test.load_data
data_test.extract_qa_pairs_from_test()

test_qa_pairs = data_test.qa_pairs
```

The three functions can't be used with their non-respective datasets due to differences in the formatting of the data.

### Testing SciQAG
Testing on and evaluating performance on SciQAG can be done with the `EvaluatorSciQAG` class in `evaluate_sciqag.py`.

`EvaluatorSciQAG` is initalized with `llm_config`, `question_dataset_path`, `document_path`, and `dataset_setting`.
- `llm_config`: Configuration for the local LLM being used.
- `question_dataset_path`: Path to the SciQAG dataset being used, which should be one of `final_all_select1000.json`, `train_qas_179511.json`, or `test_qas_8531.json`.
- `document_path`: Path to the directory containing the papers that PaperQA will use for RAG, generated by `SciQAGData.extract_papers_text_from_final`.
- `dataset_setting`: Setting to confirm which dataset is being used, `"final"`, `"train"`, or `"test"`, respectively.

For instance:
```python
evaluator = EvaluatorSciQAG(
    llm_config=local_llm_config,
    question_dataset_path="train_qas_179511.json",
    document_path="sciqag_papers_txt_only",
    dataset_setting="train"
)
```

Based on which dataset is being used, the `EvaluatorSciQAG.load_questions` function will load the ground truth question-answer pairs of the dataset as a list of dictionaries into the class variable `EvaluatorSciQAG.qa_pairs_ground_truth`.

```python
evaluator.load_questions()

evaluator.qa_pairs_ground_truth  ### list of dictionaries with ground truth question-answer pairs

[
    {"What unique properties make Polyaniline (PANI) a potential material for various applications?": "Polyaniline is unique among conducting polymers because its electrical properties can be reversibly controlled both by charge transfer doping and by protonation. This makes it a potential material for use in chemical and biological sensors, actuators, and microelectronic devices."},
    {"What has been done to overcome the insolubility, low processability, and poor mechanical properties of PANI?": "Several strategies have been developed to overcome these problems. One successful approach has been the preparation of conventional thermoplastic–electroconductive polymer composites."}
]
```

`EvaluatorSciQAG.answer_questions` will iterate through `EvaluatorSciQAG.qa_pairs_ground_truth` and ask PaperQA to provide an answer. All answers are saved, together with the original question and the ground truth answer, in `EvaluatorSciQAG.generated_answers`.

```python
evaluator.answer_questions()

evaluator.generated_answers  ### list of dictionaries with question, ground truth answer, and generated answer

[
    {
        "question": "What is the capital of France?",
        "ground_truth": "Paris",
        "system_answer": "Probably Paris"
    },
    {
        "question": "What is 2 + 2?",
        "ground_truth": "4",
        "system_answer": "It might be 4."
    }
]
```

Finally, `EvaluatorSciQAG.save_results_to_json` will output `EvaluatorSciQAG.generated_answers` to the desired path in a `JSON` format.

```python
evaluator.save_results_to_json("train_generated_answers")
```


### Evaluating SciQAG
For now, evaluating SciQAG is done with the BERTScore metric, which must be first installed with instructions at the [BERTScore GitHub](https://github.com/Tiiiger/bert_score).

`EvaluatorSciQAG.evaluate_answers` will evaluate the answers generated by the system in `EvaluatorSciQAG.generated_answers` and return three Tensors, `precision`, `recall`, and `F1 score`. Each Tensor has the same number of items with the candidate and reference lists, and each item in the list is a scalar, representing the score for the corresponding candidates and references. The current implementation uses score rescaling from BERTScore.

For instance:
```python
P, R, F1 = evaluator.evaluate_answers()

F1
tensor([0.9834, 0.9782, 0.9162, 0.9589, 0.9675, 0.9680, 0.9602, 0.9663, 0.9438, 0.9508])
```


When called on the command line, BERTScore produces a hash code, in the form of `"roberta-large_L17_no-idf_version=0.3.0(hug_trans=2.3.0)"`. This should be reported eventually for others to know the settings being used.

For instance:
```bash
bert-score -r example/refs.txt -c example/hyps.txt --lang en --rescale_with_baseline

roberta-large_L17_no-idf_version=0.3.0(hug_trans=2.3.0)-rescaled P: 0.747044 R: 0.770484 F1: 0.759045
```