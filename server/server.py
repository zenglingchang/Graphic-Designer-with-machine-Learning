import asyncio
import aiohttp
import json
import base64
from aiohttp import web

async def handle(request):
    index = open("index.html", 'rb')
    content = index.read()
    return web.Response(body=content, content_type='text/html')

async def wshandler(request):
    app = request.app
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    while True:
        try:
            msg = await ws.receive()
        except Exception as e:
            print(e)
            break
        try:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                print(data)
                if data[0] == 'Img':
                    imgdata = base64.b64decode(data[1].replace('data:image/png;base64,',''))
                    print(imgdata)
                    with open(r'd:\test.jpg', 'wb') as f:
                        f.write(imgdata)
        except Exception as e:
            print(e)
            break
    return ws

app = web.Application()
app.router.add_route('GET', '/connect', wshandler)
app.router.add_route('GET', '/', handle)
app.router.add_static('/css/',
                       path='.../css',
                       name='css')
app.router.add_static('/js/',
                       path='.../js',
                       name='js')
app.router.add_static('/font-awesome/',
                       path='.../font-awesome',
                       name='font-awesome')
                       
web.run_app(app, host='127.0.0.1', port=8080)
