import json
from langchain_community.utilities import SQLDatabase
from langchain_ollama.llms import OllamaLLM
import pandas as pd
import random

import argparse

parser = argparse.ArgumentParser(description='LLM2SQL')

CORDIS_CONFIG = {
    "drivername": "postgresql",
    "host": "localhost",
    "port": 5432,
    "database": "cordis_temporary",
    "username": "postgres",
    "password": "root",
    "schema": "unics_cordis"
}

parser.add_argument('--test_path', type=str, default="E:/GitHub/sciencebenchmark_dataset/cordis/dev.json")
parser.add_argument('--schema_path', type=str, default="E:/GitHub/sciencebenchmark_dataset/cordis/tables.json")
parser.add_argument('--train_path', type=str,
                    default="E:/GitHub/nl-ql-data-augmentation/data/cordis/handmade_training_data/handmade_data_dev.json")
parser.add_argument('--model_name', type=str, default='mistral')
parser.add_argument('--sampling_rate', type=int, default=100)
parser.add_argument('--training_size', type=int, default=100)
args = parser.parse_args()

if __name__ == "__main__":
    # Connect to database
    connection_uri = (
        f"postgresql://{CORDIS_CONFIG['username']}:{CORDIS_CONFIG['password']}"
        f"@{CORDIS_CONFIG['host']}:{CORDIS_CONFIG['port']}/{CORDIS_CONFIG['database']}"
    )
    print('Connecting')
    db = SQLDatabase.from_uri(
        connection_uri,
        schema=CORDIS_CONFIG['schema']
    )
    if not db:
        print("Database connection failed.")
        exit(1)
    print("Connection success!")
    db.autocommit = True

    # Create LLM instance
    llm = OllamaLLM(model=args.model_name)

    results = []

    # Load json files
    with open(args.schema_path, encoding="utf-8") as f:
        schemas = json.load(f)
    with open(args.train_path, encoding="utf-8") as f:
        training_data = json.load(f)
    with open(args.test_path, encoding="utf-8") as f:
        test_set = json.load(f)

    # Sampling from test set
    if args.sampling_rate >= len(test_set):
        examples = test_set.copy()
    else:
        examples = random.sample(test_set, args.sampling_rate)

    # print(training_set)

    # Reading schemas
    tables = schemas[0]["table_names_original"]
    cols = schemas[0]["column_names_original"]
    types = schemas[0]["column_types"]
    lines = [f"{tables[tid]}.{col} ({types[i]})"
             for i, (tid, col) in enumerate(cols)
             if col != "*"]
    schema = "\n".join(lines)
    training_set = ''
    for data in training_data[:args.training_size]:
        db_id = data["db_id"]
        query = data["query"]
        question = data["question"]
        example = f"db_id: {db_id}\nquery: {query}\nquestion: {question}\n\n\n"
        training_set = training_set + example

    # Evaluate examples
    for ex in examples:
        question = ex["question"]
        GT = ex["query"].strip().rstrip(";")
        db_id = ex["db_id"]
        prompt = f'''You are an expert data analyst of Community Research and Development Information Service(CORDIS) database.
        When given a question, you must reply with exactly one valid PostgreSQL SQL statement. Do NOT include any explanation, comments, or trailing semicolons—only the SQL. 
        Pay attention to the examples provided, they all come from the same database.
        Below are {args.training_size} training examples from the same dataset:
        {training_set}
        -- Schema (table.column (type)):
        {schema}
        -- Question:        
        {question}
        Requirement:
        - Output only the SQL statement that can be directly executed (no explanations, no comments, no trailing semicolons).  

        Now output the corresponding SQL statement:'''
        try:
            output = llm.invoke(prompt)
        except Exception as e:
            output = "[LLM ERROR]"
            prediction = "[LLM ERROR]"
            print(f"[LLM ERROR]: {e}")

        # Extract SQL statement
        print(f"[LLM output]: {output}")
        prediction = output.strip().split("\n")[0].rstrip(";")
        print(f"[Predicted SQL]: {prediction}")
        print(f"[GT SQL]: {GT}")

        # Execution match
        try:
            GT_res = db.run(GT)
            pred_res = db.run(prediction)

            sorted_GT = sorted(GT_res)
            sorted_pred = sorted(pred_res)
            exec_match = (sorted_GT == sorted_pred)

            if exec_match:
                print("Execution Match:")
                print(f"GT_res: {GT_res[:40]} ...")
                print(f"pred_res: {pred_res[:40]} ...")
            else:
                print("Mismatch:")
                print(f"GT_res: {GT_res[:40]} ...")
                print(f"pred_res: {pred_res[:40]} ...")
        except Exception as e:
            exec_match = False
            print(f"[SQL EXECUTION ERROR]: {e}")

        results.append({
            "question": question,
            "GT_sql": GT,
            "predicted_sql": prediction,
            "execution_match": exec_match
        })

    df = pd.DataFrame(results)
    execution_accuracy = df["execution_match"].mean()
    print("Execution Accuracy:    ", execution_accuracy)
    experiment = f'simple_{args.training_size}_{args.model_name}'

    # Save best result
    improved = False
    with open("results/accuracy.json", 'r') as f:
        existing_results = json.load(f)
    if experiment not in existing_results:
        improved = True
    elif execution_accuracy > existing_results[experiment]:
        improved = True

    if improved:
        existing_results[experiment] = execution_accuracy
        existing_results = {k: existing_results[k] for k in sorted(existing_results)}
        with open("results/accuracy.json", 'w') as f:
            json.dump(existing_results, f, indent=4)

        df.to_csv(f"results/text2sql_results_{experiment}.csv", index=False)
        print(f"Results saved to results/text2sql_results_{experiment}.csv")

    del llm

