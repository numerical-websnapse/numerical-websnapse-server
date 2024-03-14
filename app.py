from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from asyncio import Queue, create_task

from model.NSNP import NumericalSNPSystem
from middleware.nsnp_validation import NSNPSchema

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

class ConnectionManager:
    def __init__(self):
        self.active_connections = dict()

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = {
            'websocket': websocket,
            'system': None
        }

    def disconnect(self, client_id: str):
        self.active_connections.pop(client_id)

    async def receive(self, client_id: str, message_queue: Queue):
        websocket: WebSocket = self.active_connections[client_id]['websocket']
        try:
            while True:
                data = await websocket.receive_json()
                await message_queue.put(data)
        except WebSocketDisconnect:
            self.disconnect(client_id)

    async def generate(self, client_id: str, message: dict):
        schema = NSNPSchema()
        system = NumericalSNPSystem(schema.load(message.get('NSNP')))
        self.active_connections[client_id]['system'] = system
        system.simulate()
        await self.send(client_id, system.get_state_graph())

    async def next(self, client_id: str, message: dict):
        system = self.active_connections[client_id]['system']
        config = message.get('config')
        spike = message.get('spike')
        next = system.next(config, spike)
        await self.send(client_id, next)

    async def active(self, client_id: str, message: dict):
        system = self.active_connections[client_id]['system']
        config = message.get('config')
        active = system.get_active_array(config)
        await self.send(client_id, active)

    async def spike(self, client_id: str, message: dict):
        system = self.active_connections[client_id]['system']
        config = message.get('config')
        spike = system.compute_random_spike(config)
        await self.send(client_id, spike)

    async def config(self, client_id: str, message: dict):
        system = self.active_connections[client_id]['system']
        config = message.get('config')
        details = system.get_config_details(config)
        await self.send(client_id, details)

    async def matrices(self, client_id: str, message: dict):
        system = self.active_connections[client_id]['system']
        config = message.get('config')
        matrices = system.get_config_matrices(config)
        await self.send(client_id, matrices)

    async def router(self, client_id: str, message: dict):
        type = message.get('type')
        try:
            if type == 'generate':
                await self.generate(client_id, message)
            elif type == 'next':
                await self.next(client_id, message)
            elif type == 'active':
                await self.active(client_id, message)
            elif type == 'spike':
                await self.spike(client_id, message)
            elif type == 'config':
                await self.config(client_id, message)
            elif type == 'matrices':
                await self.matrices(client_id, message)
        except Exception as e:
            await self.send(client_id, {'error': str(e)})

    async def send(self, client_id: str, message: dict):
        websocket: WebSocket = self.active_connections[client_id]['websocket']
        await websocket.send_json(message)
        
manager = ConnectionManager()

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    message_queue = Queue()
    try:
        receive_task = create_task(manager.receive(client_id, message_queue))
        while True:
            request = await message_queue.get()
            await manager.router(client_id, request)

    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)
        receive_task.cancel()

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Websocket Demo</title>
           <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">

    </head>
    <body>
    <div class="container mt-3">
        <h1>FastAPI WebSocket Chat</h1>
        <h2>Your ID: <span id="ws-id"></span></h2>
        <form action="">
            <input type="text" class="form-control" id="messageText" autocomplete="off"/>
            <button class="btn btn-outline-primary mt-2">Send</button>
        </form>
        <ul id='messages' class="mt-5">
        </ul>
        
    </div>
    
        <script>
            var client_id = Date.now()
            document.querySelector("#ws-id").textContent = client_id;
            var ws = new WebSocket(`ws://localhost:8000/ws/${client_id}`);
            ws.onmessage = function(event) {
                console.log(JSON.parse(event.data));
            };

            ws.onopen = function(event) {
                ws.send(
                    JSON.stringify({
                        type: 'generate',
                        NSNP: {
                            "neurons": [
                                {
                                "id": "n_1",
                                "data": {
                                    "var_":[
                                        ["x_{(1,1)}","1"],
                                        ["x_{(2,1)}","1"]
                                    ],
                                    "prf": [
                                        [
                                        "f_{(1,1)}",
                                        null,
                                        [
                                            ["x_{(1,1)}","1"],
                                            ["x_{(2,1)}","1"]
                                        ]
                                        ],
                                        [
                                        "f_{(2,1)}",
                                        null,
                                        [
                                            ["x_{(1,1)}","0.5"],
                                            ["x_{(2,1)}","0.5"]
                                        ]
                                        ]
                                    ],
                                    "label": "n_1",
                                    "ntype": "reg",
                                    "x": 100,
                                    "y": -50
                                }
                                },
                                {
                                "id": "n_2",
                                "data": {
                                    "var_": [
                                        ["x_{(1,2)}", "2"]
                                    ],
                                    "prf": [
                                        [
                                            "f_{(1,2)}",
                                            null,
                                            [
                                                ["x_{(1,2)}", "1"]
                                            ]
                                        ],
                                        [
                                            "f_{(2,2)}",
                                            "4",
                                            [
                                                ["x_{(1,2)}", "0.5"]
                                            ]
                                        ]
                                    ],
                                    "label": "n_2",
                                    "ntype": "reg",
                                    "x": 100,
                                    "y": 150
                                }
                                },
                                {
                                    "id": "n_out",
                                    "data": {
                                        "label": "train",
                                        "ntype": "out",
                                        "var_" : [],
                                        "prf": [],
                                        "train": []
                                    }
                                }
                            ],
                            "syn": [
                                {
                                "id": "syn1",
                                "source": "n_1",
                                "target": "n_2",
                                "data": {}
                                },
                                {
                                "id": "syn2",
                                "source": "n_2",
                                "target": "n_1",
                                "data": {}
                                },
                                {
                                    "id": "syn3",
                                    "source": "n_2",
                                    "target": "n_out",
                                    "data": {}
                                }
                            ]
                        }
                    })
                );

                ws.send(JSON.stringify({type: 'next', config: [1,1,2], spike: null}));
                ws.send(JSON.stringify({type: 'active', config: [1,1,2]}));
                ws.send(JSON.stringify({type: 'active', config: [4,4,4]}));
                ws.send(JSON.stringify({type: 'spike', config: [4,4,2]}));
                ws.send(JSON.stringify({type: 'config', config: [1,1,2]}));
            };
            
        </script>
    </body>
</html>
"""