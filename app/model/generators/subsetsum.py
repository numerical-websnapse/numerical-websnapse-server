from random import randint, choice
from pprint import pprint
import json, string

position = {
    'x': 200,
    'y': 200,
}

def id_generator(size=6, chars=string.ascii_lowercase + string.digits):
    return ''.join(choice(chars) for _ in range(size))

def generate_subset_sum(n, sum=20):
    node_ids = [id_generator(8) for i in range(n*2+5)]

    nodes = []
    # 0 to N-1
    for i in range(n):
        nodes.append({
            'id':node_ids[i],
            'data': {
                'var_': [['x',str(randint(1, 10))]],
                'prf': [['f_{1}','',[['x','1']]],['f_{2}','',[['x','-1']]]],
                'train': [],
                'type': 'reg',
                'label': '\sigma_{{{i}}}'.format(i=i),
                'x': position['x'],
                'y': position['y'] * i,
            }
        })

    # N to 2N-1
    for i in range(n):
        nodes.append({
            'id':node_ids[n+i],
            'data': {
                'var_': [['x','0']],
                'prf': [['f_{1}','0',[['x','1']]]],
                'train': [],
                'type': 'reg',
                'label': '\sigma_{{{i}}}'.format(i=n+i),
                'x': position['x'] * 2,
                'y': position['y'] * i,
            }
        })

    # 2N
    nodes.append({
        'id':node_ids[n*2],
        'data': {
            'var_': [['x',str(sum+1)]], # x[S + 1]
            'prf': [['f_{1}','',[['x','-1']]]],
            'train': [],
            'type': 'reg',
            'label': '\sigma_{{{i}}}'.format(i=n*2),
            'x': position['x'] * 3,
            'y': position['y'] * ((n-1)-2),
        }
    })

    # 2N+1
    nodes.append({
        'id':node_ids[(n*2)+1],
        'data': {
            'var_': [['x','0']], # x[S + 1]
            'prf': [['f_{1}','',[['x','1']]]],
            'train': [],
            'type': 'reg',
            'label': '\sigma_{{{i}}}'.format(i=(n*2)+1),
            'x': position['x'] * 3,
            'y': position['y'] * ((n-1)-1),
        }
    })

    # 2N+2
    nodes.append({
        'id':node_ids[(n*2)+2],
        'data': {
            'var_': [['x','0']], # x[S + 1]
            'prf': [['f_{1}','-1',[['x','-1']]]],
            'train': [],
            'type': 'reg',
            'label': '\sigma_{{{i}}}'.format(i=(n*2)+2),
            'x': position['x'] * 3,
            'y': position['y'] * ((n-1)-0),
        }
    })

    # 2N+3
    nodes.append({
        'id':node_ids[(n*2)+3],
        'data': {
            'var_': [['x','0']], # x[S + 1]
            'prf': [['f_{1}','1',[['x','1']]]],
            'train': [],
            'type': 'reg',
            'label': '\sigma_{{{i}}}'.format(i=(n*2)+3),
            'x': position['x'] * 4,
            'y': position['y'] * ((n-1)-0),
        }
    })

    nodes.append({
        'id':node_ids[(n*2)+4],
        'data': {
            'var_': [], # x[S + 1]
            'prf': [],
            'train': [],
            'type': 'out',
            'label': '\sigma_{{{i}}}'.format(i=(n*2)+4),
            'x': position['x'] * 5,
            'y': position['y'] * ((n-1)-0),
            'train': [],
        }
    })

    edges = []

    for i in range(n):
        edges.append({
            'id':'syn_{(%s,%s)}'%(i,n+i),
            'source':node_ids[i],
            'target':node_ids[n+i],
        })

        edges.append({
            'id':'syn_{(%s,%s)}'%(n+i,(n*2)+2),
            'source':node_ids[n+i],
            'target':node_ids[(n*2)+2],
        })

    edges.append({
        'id':'syn_{(%s,%s)}'%(n*2,n*2+1),
        'source':node_ids[(n*2)],
        'target':node_ids[(n*2)+1],
    })

    edges.append({
        'id':'syn_{(%s,%s)}'%(n*2+1,n*2+2),
        'source':node_ids[(n*2)+1],
        'target':node_ids[(n*2)+2],
    })

    edges.append({
        'id':'syn_{(%s,%s)}'%(n*2+2,n*2+3),
        'source':node_ids[(n*2)+2],
        'target':node_ids[(n*2)+3],
    })

    edges.append({
        'id':'syn_{(%s,%s)}'%(n*2+3,n*2+4),
        'source':node_ids[(n*2)+3],
        'target':node_ids[(n*2)+4],
    })

    return { 'nodes':nodes, 'edges':edges }

# print(json.dumps(generate_subset_sum(3), indent=4))
size = [n for n in range(50,1050,50)]
sum = [n for n in range(50,1050,50)]

for i in range(len(size)):
    data = generate_subset_sum(size[i],sum[i])
    with open('app/tests/subsetsum/subset-%s-%s.json'%(size[i], sum[i]), 'w') as f:
        json.dump(data, f, indent=4)