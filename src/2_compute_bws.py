import pandas as pd
from pathlib import Path
from utils import bw_scale

data_path = '../data'
annotations_df = pd.read_csv(Path(data_path, "brys_spec_bws_annotation_GPT-4.tsv"), sep='\t', na_values=[''], keep_default_na=False)
specificity = bw_scale(annotations_df=annotations_df)
specificity.to_csv(Path(data_path, "brys_specificity_GPT-4.tsv"), sep='\t', index=False)