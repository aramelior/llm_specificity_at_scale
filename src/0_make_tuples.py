# Description: This script generates tuples for the BWS experiment using the words from the Brysbaert concreteness ratings dataset.
import pandas as pd
from pathlib import Path
from itertools import combinations
from utils import make_tuples


if __name__ == "__main__":
    data_path = Path("../data")
    brys_file = Path(data_path, "brysbaert_concreteness_ratings.tsv")
    brys = pd.read_csv(brys_file, sep="\t", na_values=[""], keep_default_na=False)

    target_pos = [
        "Noun",
        "Verb",
        "Adjective",
    ]

    combinations_count = {}
    all_combinations = []

    bws_tuples = []
    for pos in target_pos:
        df = brys[brys["Dom_Pos"] == pos]
        word_list = df["Word"].sample(frac=1).tolist()
        all_combinations += list(combinations(word_list, 2))
        tuples = make_tuples(instances=word_list, tuple_size=4, repetition_factor=8)

        for tup in list(tuples.values()):
            bws_tuples.append(
                {
                    "pos": pos,
                    "item_1": tup[0],
                    "item_2": tup[1],
                    "item_3": tup[2],
                    "item_4": tup[3],
                }
            )

    bws_tuples = pd.DataFrame(bws_tuples)

    combinations_count = {}

    for i, r in bws_tuples.iterrows():
        tuple_comb = list(
            combinations(
                [
                    r["item_1"],
                    r["item_2"],
                    r["item_3"],
                    r["item_4"],
                ],
                2,
            )
        )
        for t_comb in tuple_comb:
            if t_comb in combinations_count:
                combinations_count[t_comb] += 1
            else:
                combinations_count[t_comb] = 1

    combinations_count = pd.DataFrame(
        combinations_count.items(), columns=["Combination", "Count"]
    ).sort_values("Count", ascending=False)
    combinations_count

    if len(combinations_count[combinations_count["Count"] > 1]) == 0:
        print("No repeated combinations")
        
    else:
        print("Repeated combinations found")
        print(combinations_count[combinations_count["Count"] > 1])

bws_tuples.to_csv(
            Path(data_path, "bws_tuples_brys_n_v_a.tsv"), sep="\t", index=False
        )