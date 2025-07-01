# Codes for LLM2SQL coding challenge
This repo is for LLM2SQL challenge on CORDIS dataset. There are 3 python files in the repo:

* **[llm2sql.py]** Simple implementation with only LLM
* **[llm2sql_chain.py]** SQLDatabaseChain implementation
* **[llm2sql_langgraph.py]** Langgraph implementation

All python files use 6 arguments:
* --test_path, string, path to the json file of test data 
* --schema_path, string, path to the json file of database schema
* --train_path, string, path to the json file of train data
* --model_name, string, deciding which LLM to use
* --sampling_rate, integer, defining number of samples to be evaluated from test set
* --training_size, integer, defining number of samples to be used for prompt

Experiment results are stored in `results/`