import json 
from tqdm import tqdm
path = "results_combined.json"
with open("./data/private_test.json", "r", encoding="utf-8") as f:
    questions = json.load(f)
mapping_qid = {q["qid"]: q for q in questions}
with open(path, "r", encoding="utf-8") as f:
    results = json.load(f)
new_results = []
threshold = 0.55
for result in tqdm(results, desc="Processing results"):
    qid = result["qid"]
    question = mapping_qid[qid]['question']
    relevant_laws_detail = result["relevant_laws_detail"]
    list_aid_ids_gold = []
    list_aid_ids_pred = []
    for law_detail in relevant_laws_detail:
        if law_detail["distance"] > threshold:
            if law_detail["aid"] not in list_aid_ids_gold:
                list_aid_ids_gold.append(law_detail["aid"])
    
        if law_detail["aid"] not in list_aid_ids_pred:
            list_aid_ids_pred.append(law_detail["aid"])
    if list_aid_ids_gold and len(list_aid_ids_gold):
        if len(list_aid_ids_gold) > 10:
            relevant_laws = list_aid_ids_gold[:10]
        else:
            relevant_laws = list_aid_ids_gold
    else:
        if len(list_aid_ids_pred) > 10:
            relevant_laws = list_aid_ids_pred[:10]
        else:
            relevant_laws = list_aid_ids_pred
    new_results.append({
        "qid": qid,
        "question": question,
        "relevant_laws": relevant_laws
    })

with open("results_combined_top10.json", "w", encoding="utf-8") as f:
    json.dump(new_results, f, ensure_ascii=False, indent=4)