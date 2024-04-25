from app.model.NSNP import NumericalSNPSystem
from app.model.converter.converter import convert_to_nsnapse
from app.model.converter.convert_validation import NSNPSchema

import os, sys, time
from pprint import pprint
import json, glob

files = glob.glob(f"app\\tests\\custom\\*.json")

for file in files:

    if 'converted' in file:
        continue

    start_time = time.time()
    new_path = file.replace('tests', 'tests\\converted')
    
    with open(file, 'r') as f:
        data = json.load(f)

        schema = NSNPSchema()
        system = NumericalSNPSystem(
            schema.load({
                'neurons' : data['nodes'],
                'syn' : data['edges']
            })
        )

        output = convert_to_nsnapse(system)
        with open(new_path, 'w') as f:
            json.dump(output, f)

    print(f"{file:50} - \t{time.time() - start_time}s")