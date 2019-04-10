import os, sys, time, json, base64, io
import asyncio
import aiohttp
import numpy as np
from PIL import Image
from aiohttp import web
from multiprocessing import Process

#############################
#
#   Other Function 
#
##############################
x = 'asdfasdvzxcvsad'
def show():
    print(x)
    return 
    
def help():
    content = '''Command:
        -help  | help infomation
        -show  | Show network frameWork
        -exit  | exit!
    '''
    print(content)
    return 
    
Description = {
    '-help' : {
        'function' : help,
    },
    '-show' : {
        'function' : show,
    }
}

###################################
#
#   Web Config & Function
#
####################################

async def handle(request):
    index = open(os.path.join(sys.path[0], '../index.html'), 'rb')
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
                    image = io.BytesIO(imgdata)
                    img = Image.open(image)
                    img = img.resize((192,256), Image.ANTIALIAS)
                    img.show()
                    c = np.array(img)
                    print(c.reshape([-1,192*256*4]))
                    
        except Exception as e:
            print(e)
            break
    return ws

def WebInit():
    print('Web Server: Process (%s) start...' % os.getpid())
    app = web.Application()
    app.router.add_route('GET', '/connect', wshandler)
    app.router.add_route('GET', '/', handle)
    app.router.add_static('/css/',
                           path = os.path.join(sys.path[0], '../css'),
                           name = 'css')
    app.router.add_static('/js/',
                            path = os.path.join(sys.path[0], '../js'),
                           name = 'js')
    app.router.add_static('/font-awesome/',
                           path = os.path.join(sys.path[0], '../font-awesome'),
                           name = 'font-awesome')
    web.run_app(app, host='127.0.0.1', port=8080)

#####################################
#
#   Program entry & Main Loop
#
#####################################

def Main():
    print('Main Process (%s) start...' % os.getpid())
    time.sleep(1)
    while 1:
        Input = input('Manager:')
        if Input == '-exit':
            break
        elif Input in Description :
            Description[Input]['function']()
        else:
            print('Can\'t find Command: %s ' % Input)
            help()
    return 
    
if __name__ == '__main__':
    p1 = Process(target = WebInit, args = ())
    p1.start()
    Main()
    p1.terminate()