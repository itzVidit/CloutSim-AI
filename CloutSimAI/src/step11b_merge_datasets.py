import pandas as pd
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent.parent

MAIN_DATA=BASE_DIR/"data"/"proceed"/"dreams_validated.csv"
MIXED_DATA=BASE_DIR/"data"/"synthetic"/"dreams_mixed_intent.csv"

OUTPUT_FILE=BASE_DIR/"data"/"proceed"/"dreams_augmented.csv"

df_main=pd.read_csv(MAIN_DATA)
df_mixed=pd.read_csv(MIXED_DATA)

df_final=pd.concat([df_main,df_mixed],ignore_index=True)

df_final.to_csv(OUTPUT_FILE,index=False)
print("Datasets merged")
print("Final number of rows: ",len(df_final))