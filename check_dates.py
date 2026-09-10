import pandas as pd
df = pd.read_csv(r"C:\GradResearch\WildFireData\data output\datasets\firms_palisades_bbox_w_m118p900_s_33p960_e_m118p380_n_34p200\firms_palisades.csv")
print(df.groupby("firms_source")["acq_date"].agg(["min", "max", "count"]))