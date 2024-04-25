import os, sys, time

from app.model.NSNP import NumericalSNPSystem
from app.model.converter.convert_validation import NSNPSchema
from pprint import pprint
import json, glob

def convert_to_nsnapse(system: NumericalSNPSystem):
    VL = []
    for i in system.neuron_to_var:
        for j in system.neuron_to_var[i]:
            VL.append(i+1)

    T = []
    for i, func in enumerate(system.functions):
        if func[1] is not None:
            T.append([i + 1, int(func[1])])

    S = []
    for source in system.neuron_to_neuron:
        for target in system.neuron_to_neuron[source]:
            S.append([source + 1, target + 1])

    def neuron_base(index, neuron):
        data = system.reg_neurons[neuron]
        return {
            "id": f"Neuron {index + 1}",
            "position": {
                "x": data['x'],
                "y": data['y']
            }
        }
    
    def neuron_content(index, neuron):
        data = system.reg_neurons[neuron]
        return {
            "id": f"neuron-contents{index}",
            "position": {
                "x": data['x'],
                "y": data['y']
            }
        }

    new_data = {
        'C'         : system.config_mx[0].astype(int).tolist(),
        'VL'        : VL,
        'F'         : system.function_mx.astype(float).tolist(),
        'L'         : system.f_location_mx.astype(int).tolist(),
        'T'         : T,
        'syn'       : S,
        'envSyn'    : len(system.reg_neurons),
        'neuronPositions' : [
            mapper(index, neuron)
            for index, neuron in enumerate(system.reg_neurons)
            for mapper in (neuron_base, neuron_content)
        ]
    }

    return new_data

if __name__ == '__main__':
    import glob
    files = glob.glob(f"app\\tests\\*\\*.json")

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