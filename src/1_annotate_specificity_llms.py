import os
import csv
from pathlib import Path
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from datetime import datetime


def promptize_tuple(word_tuple):
    string_prompt = f"0 - {word_tuple[0].upper()},\n1 - {word_tuple[1].upper()},\n2 - {word_tuple[2].upper()},\n3 - {word_tuple[3].upper()}\n"
    return string_prompt


def annotate(chain, tup):
    prompt_tuple = promptize_tuple(tup)
    request = {
        "input": prompt_tuple,
    }
    response = chain.invoke(request)

    return response.content


def index_to_words(tup, reply):
    try:
        idx_b, idx_w = reply.split(",", 1)
        idx_b = int(idx_b.strip()[-1])
        idx_w = int(idx_w.strip()[0])
        best = tup[idx_b]
        worst = tup[idx_w]
    except:
        idx_b = None
        idx_w = None
        best = None
        worst = None

    return best, worst


def write_out(out_file_name, results_dict):
    out_annotation_file = Path(str(out_file_name.absolute()))
    if not out_annotation_file.exists():
        with out_annotation_file.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results_dict.keys(), delimiter="\t")
            writer.writeheader()
            writer.writerow(results_dict)
    else:
        with out_annotation_file.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results_dict.keys(), delimiter="\t")
            writer.writerow(results_dict)


if __name__ == "__main__":
    execution_time = datetime.now()
    execution_time = str(execution_time.isoformat().replace(":", "-").split(".")[0])

    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    MODEL = "gpt-4"
    TASK_INSTRUCTIONS = open("./specificity_task_instructions.txt", "r").read()
    llm = ChatOpenAI(model=MODEL, max_completion_tokens=48)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", TASK_INSTRUCTIONS),
            ("user", "{input}"),
        ]
    )
    chain = prompt_template | llm

    data_path = "../data"
    tuple_df = pd.read_csv(Path(data_path, "bws_tuples_brys_n_v_a.tsv"), sep="\t", na_values=[''], keep_default_na=False)

    out_file_name = Path(data_path, f"brys_spec_bws_annotation_{MODEL.upper()}.tsv")

    for i, r in tuple_df.iterrows():
        tup = (r["item_1"], r["item_2"], r["item_3"], r["item_4"])
        reply = annotate(chain, tup)
        best, worst = index_to_words(tup, reply)

        row = {
            "pos": r["pos"],
            "item_1": tup[0],
            "item_2": tup[1],
            "item_3": tup[2],
            "item_4": tup[3],
            "best": best,
            "worst": worst,
            "explanation": reply,
            "model": MODEL,
        }

        write_out(out_file_name=out_file_name, results_dict=row)

        print(tup)
        print(f"best: {best}, worst: {worst}")
        print(f"{i+1} of {len(tuple_df)} tuples annotated")
        print("\n\n")

