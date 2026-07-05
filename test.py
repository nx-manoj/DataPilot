import pandas as pd 
import datapilot as dp 

df = pd.read_csv("/home/manoj/Downloads/titanic.csv")

dp.analyze(df,use_ai=True,ai_model="llama3")

