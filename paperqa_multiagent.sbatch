#!/bin/bash
#SBATCH -J paperqa_multiagent_sciqag              # Job name
#SBATCH -N 1                             # Number of nodes required
#SBATCH --ntasks-per-node=4              
#SBATCH --gres=gpu:H100:1                # Request one H100 GPU
#SBATCH --mem=64G                       # Adjust memory allocation 
#SBATCH --time=16:00:00                  # Time limit (16 hours)
#SBATCH -o Report-%j.out                 # Standard output and error log
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail notifications for job events
#SBATCH --mail-user=lpimentel3@gatech.edu # Email address for notifications

# Activate your Python environment (if needed)
source activate cRAG    # Replace with your conda environment name

# Change to the directory where the job was submitted from
cd /home/hice1/lpimentel3/scratch/cRAG/

# Export necessary environment variables
export CUDA_VISIBLE_DEVICES=0 

# Start ollama serve in the background
OLLAMA_HOST=127.0.0.1:11443 ollama serve &

# Wait a few seconds to ensure ollama serve starts properly
sleep 20


# ablation on llama3.2 model with agents
python evaluate_sciqag.py --question_dataset_path datasets/SciQAG/questions_raw_subset_10-40 --document_path datasets/SciQAG/papers --method paperqa_multiagent --llm_model llama3.2 --num_agents 4 --num_rounds 2 --port 11443



# ablation on llama3.1 model with agents with 2 rounds of communication 

python evaluate_sciqag.py --question_dataset_path datasets/SciQAG/questions_raw_subset_10-40 --document_path datasets/SciQAG/papers --method paperqa_multiagent --llm_model llama3.1 --num_agents 2  --num_rounds 2 --port 11443

python evaluate_sciqag.py --question_dataset_path datasets/SciQAG/questions_raw_subset_10-40 --document_path datasets/SciQAG/papers --method paperqa_multiagent --llm_model llama3.1 --num_agents 3  --num_rounds 2 --port 11443

python evaluate_sciqag.py --question_dataset_path datasets/SciQAG/questions_raw_subset_10-40 --document_path datasets/SciQAG/papers --method paperqa_multiagent --llm_model llama3.1 --num_agents 4  --num_rounds 2 --port 11443


# ablation on llama3.2 model with agents with 3 rounds of communication
python evaluate_sciqag.py --question_dataset_path datasets/SciQAG/questions_raw_subset_10-40 --document_path datasets/SciQAG/papers --method paperqa_multiagent --llm_model llama3.2 --num_agents 2  --num_rounds 3 --port 11443

python evaluate_sciqag.py --question_dataset_path datasets/SciQAG/questions_raw_subset_10-40 --document_path datasets/SciQAG/papers --method paperqa_multiagent --llm_model llama3.2 --num_agents 3  --num_rounds 3 --port 11443

python evaluate_sciqag.py --question_dataset_path datasets/SciQAG/questions_raw_subset_10-40 --document_path datasets/SciQAG/papers --method paperqa_multiagent --llm_model llama3.2 --num_agents 4  --num_rounds 3 --port 11443


# ablation on llama3.1 model with agents with 3 rounds of communication

python evaluate_sciqag.py --question_dataset_path datasets/SciQAG/questions_raw_subset_10-40 --document_path datasets/SciQAG/papers --method paperqa_multiagent --llm_model llama3.1 --num_agents 2  --num_rounds 3 --port 11443

python evaluate_sciqag.py --question_dataset_path datasets/SciQAG/questions_raw_subset_10-40 --document_path datasets/SciQAG/papers --method paperqa_multiagent --llm_model llama3.1 --num_agents 3  --num_rounds 3 --port 11443

python evaluate_sciqag.py --question_dataset_path datasets/SciQAG/questions_raw_subset_10-40 --document_path datasets/SciQAG/papers --method paperqa_multiagent --llm_model llama3.1 --num_agents 4  --num_rounds 3 --port 11443



# Wait for all background processes to finish before exiting
wait
