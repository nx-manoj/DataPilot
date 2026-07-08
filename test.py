import pandas as pd 
import datapilot as dp 

df = pd.read_csv("/home/manoj/Downloads/titanic.csv")

dp.configure(ai_provider='ollama', ai_model='llama3', api_key='ollama')
dp.analyze(df, use_ai=True)
# test code
