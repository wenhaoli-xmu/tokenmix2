import json


method = "success17.json"
object_files = [
    "multifieldqa_en.jsonl",
    "multifieldqa_zh.jsonl",
    "narrativeqa.jsonl",
    "qasper.jsonl",
]

for object_file in object_files:
    with open(f"pred/{method}/{object_file}", 'r') as f:
        x = f.readlines()
        
    y = []
    for line in x:
        line = json.loads(line)

        if '\n' in line['pred']:
            index = line['pred'].index('\n')
            line['pred'] = line['pred'][:index]

        if '\t' in line['pred']:
            index = line['pred'].index('\t')
            line['pred'] = line['pred'][:index]
        
        y.append(line)


    with open(f"pred/{method}/{object_file}", 'w') as f:
        for line in y:
            f.write(json.dumps(line) + '\n')
