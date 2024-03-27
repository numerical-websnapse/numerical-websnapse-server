import os, sys, time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.append(parent_dir)

from app.model.NSNP import NumericalSNPSystem
from convert_validation import NSNPSchema
from pprint import pprint
import json, glob


def convert_to_nsnapse(data):
    schema = NSNPSchema()
    system = NumericalSNPSystem(
        schema.load({
            'neurons' : data['nodes'],
            'syn' : data['edges']
        })
    )

    VL = []
    for v2f_mapping in system.var_to_func:
        for v in system.var_to_func[v2f_mapping]:
            VL.append(v2f_mapping + 1)

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
        'F'         : system.function_mx.astype(int).tolist(),
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
            data = convert_to_nsnapse(json.load(f))
            with open(new_path, 'w') as f:
                json.dump(data, f)

        print(f"{file:50} - \t{time.time() - start_time}s")