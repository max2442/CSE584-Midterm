from transformers import pipeline
import pandas as pd
#import time


temp = pd.read_csv('t', sep='\t')
temp.columns = [0,1]
seen_before = set()
data = []
for a in range(1250):
    sen_start = ' '.join(temp[0][a].replace("'", "").strip().split()[:5])
    if sen_start not in seen_before:
        seen_before.add(sen_start)
        data.append(sen_start)

model_list = ["openai-community/gpt2","utter-project/EuroLLM-1.7B","amd/AMD-Llama-135m",
              "openai-community/openai-gpt","HuggingFaceTB/SmolLM-135M","Norod78/SmolLM-135M-FakyPedia-EngHeb",
              "h2oai/h2o-danube3-500m-chat","Den4ikAI/DLM_500m"]

fin_data = pd.DataFrame(columns=["Text","Label"])
#start = time.perf_counter()
for model in range(len(model_list)):
    generator = pipeline('text-generation', model = model_list[model])
    completed_text = generator(data, max_length = 32, num_return_sequences=1)
    for a in completed_text:
        fin_data.loc[len(fin_data.index)] = [a[0]['generated_text'],model]
        # print(a[0]['generated_text'])
#end = time.perf_counter()
#print(end-start)
fin_data.to_csv("total_data.csv",index=False)
