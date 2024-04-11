from random import randint, uniform, choices, choice
from pprint import pprint
import json, string

import numpy as np
import math

def circle_points(r, n):
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 2*np.pi, n, endpoint=False)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    return circles

def grid_points(n):
    grids = []
    for ni in n:
        rows = math.ceil(math.sqrt(ni))
        cols = math.ceil(ni/rows)
        x = np.linspace(0, 1, cols) * rows * 200
        y = np.linspace(0, 1, rows) * cols * 200
        xv, yv = np.meshgrid(x, y)
        grid = np.c_[xv.ravel(), yv.ravel()]
        grids.append(grid[0:ni])

    return grids

def line_points(n):
    lines = []
    for ni in n:
        x = np.linspace(0, 800, ni)
        y = np.zeros(ni)
        lines.append(np.c_[x, y])
    return lines

def id_generator(size=6, chars=string.ascii_lowercase + string.digits):
    return ''.join(choice(chars) for _ in range(size))

def generate_radius(n):
    return [count*100 for count in n]

def generate_data(positions, type = 'one', loop = False, data_template = None):
    nodes = []
    for i, pos in enumerate(positions):
        nodes.append({
            'id': id_generator(8),
            'data': {
                'label' : '\\sigma_{%s}'%(i),
                'type' : 'reg',
                'var_'  : [['x_{1}', '0']] if type == 'one' and i != 0 else [['x_{1}', '1']],
                'prf'   : [['f_{1}', '1', [['x_{1}', '1']]]],
                'train' : [],
                'x'     : pos[0],
                'y'     : pos[1],
            } if data_template is None else {
                'label' : '\\sigma_{%s}'%(i),
                'type' : 'reg',
                'var_'  : data_template['var_'],
                'prf'   : data_template['prf'],
                'train' : [],
                'x'     : pos[0],
                'y'     : pos[1],
            }
        })
    
    last_pos = positions[-1]
    x_offset = positions[1][0] - positions[0][0]
    y_offset = positions[1][1] - positions[0][1]

    nodes.append({
        'id': id_generator(8),
        'data': {
            'label' : 'out',
            'type' : 'out',
            'var_'  : [],
            'prf'   : [],
            'train' : [],
            'x'     : last_pos[0] + x_offset,
            'y'     : last_pos[1] + y_offset,
        }
    })

    edges = []
    for i in range(len(nodes) - 1):

        j = (i+1)%len(nodes)

        edges.append({
            'id': id_generator(12),
            'source': nodes[i]['id'],
            'target': nodes[j]['id'],
            'data'  : {},
        })

    if loop:
        edges.append({
            'id': id_generator(12),
            'source': nodes[-2]['id'],
            'target': nodes[0]['id'],
            'data'  : {},
        })

    return {
        'nodes': nodes,
        'edges': edges,
    }


n = [n for n in range(50,1050,50)]
r = generate_radius(n)
points = grid_points(n) #circle_points(r, n)

for i, point in enumerate(points):
    data = generate_data(point.tolist(), type='one')
    with open(f'app/tests/chain/one-chain-{n[i]}.json', 'w') as f:
        json.dump(data, f)

    data = generate_data(point.tolist(), type='all')
    with open(f'app/tests/chain/all-chain-{n[i]}.json', 'w') as f:
        json.dump(data, f)

    data = generate_data(point.tolist(), type='one', loop=True)
    with open(f'app/tests/chain/one-chain-{n[i]}-loop.json', 'w') as f:
        json.dump(data, f)

    data = generate_data(point.tolist(), type='all', loop=True)
    with open(f'app/tests/chain/all-chain-{n[i]}-loop.json', 'w') as f:
        json.dump(data, f)






