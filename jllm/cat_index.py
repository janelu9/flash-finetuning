import json
import os
import time
import sys

if __name__=='__main__':
    output_path = sys.argv[1]
    num_stages = len([i for i in os.listdir(output_path) if i.endswith('.json')])
    file_mode = "model-{stage_id:05d}-of-"+f"{num_stages:05d}.safetensors" if num_stages>1 else "model.safetensors"
    index_json = "model.safetensors.index.json"
    index = {"metadata":{"total_size":0},"weight_map":{}}
    for i in range(num_stages):
        idx_file = os.path.join(output_path,f"{i + 1:05d}.json")
        while not os.path.exists(idx_file):
            print(f'file not found: {idx_file}! Waiting for seconds')
            time.sleep(3)
        with open(idx_file,'r')as f:
            tmp_js = json.load(f)
            index["weight_map"].update({k:file_mode.format(stage_id=i+1) for k in tmp_js.keys()})
            index["metadata"]["total_size"]+=sum(tmp_js.values())
        print(f'{idx_file} has been joined.')
        os.remove(idx_file)
    with open(os.path.join(output_path,index_json),'w') as f:
        json.dump(index,f,indent=2)