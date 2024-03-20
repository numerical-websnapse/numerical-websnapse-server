from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from asyncio import Queue, create_task

from app.model.NSNP import NumericalSNPSystem
from app.middleware.nsnp_validation import NSNPSchema

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1",
]

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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
        system.simulate(branch='initial')
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

@app.post("/matrices/{client_id}")
async def matrices(client_id: str, message: Request):
    data = await message.json()
    try:
        client = manager.active_connections[client_id]
        system = client['system']
        config = data.get('config')
        matrices = system.get_config_matrices(config)
        return matrices
    except Exception as e:
        return {'error': str(e)}

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    print(f'Client {client_id} connected')
    message_queue = Queue()
    
    try:
        receive_task = create_task(manager.receive(client_id, message_queue))
        while True:
            request = await message_queue.get()
            await manager.router(client_id, request)

    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)
        receive_task.cancel()
