//***********************************
//
// class ImageElement and TextElement
//
//***********************************

class ImageElement{
    constructor(content, x, y, width, height){
        this.type = 'img';
        this.content = content;
        this.x = x;
        this.y = y;
        if (height == 0 || width == 0){
            this.height = content.height;
            this.width = content.width;
        }
        else{
            this.height = height;
            this.width = width;
        }
    }
    
    InsideRect(MouseX, MouseY){
        return this.x < MouseX && MouseX < this.width + this.x && this.y < MouseY && MouseY < this.height + this.y;
    }
    
    move(OffsetX, OffsetY){
        this.x += OffsetX;
        this.y += OffsetY;
    }
}

class TextElement{
    constructor(content, x, y, width, height){
        this.type = 'text';
        this.content = content;
        this.x = x;
        this.y = y;
        this.height = height;
        this.width = width;
    }
    
    InsideRect(MouseX, MouseY){
        return this.x < MouseX && MouseX < this.width + this.x && this.y < MouseY && MouseY < this.height + this.y;
    }
}

//***********************************
//
// init
//
//***********************************


SrcCanvas = document.getElementById("srcCanvas");
CanvasWidth = SrcCanvas.width;
CanvasHeight = SrcCanvas.height;
RenderList = new Array();

function windowToCanvas(x,y) {
    var box = SrcCanvas.getBoundingClientRect();  
    return {
        x: x - box.left - (box.width - SrcCanvas.width) / 2,
        y: y - box.top - (box.height - SrcCanvas.height) / 2
    };
}

function CanvasInit(){
    SrcCanvas.onmousedown  = function (event){
        var index = 0;
        var pos = windowToCanvas(event.clientX, event.clientY);
        for(var i = RenderList.length-1; i!=-1; i--){
            if(RenderList[i].InsideRect(pos.x, pos.y)){
                index = i;
                break;
            }
        }
        
        if (index == 0){
            return ;
        }
        Oldmove = SrcCanvas.onmousemove;
        SrcCanvas.onmousemove = function (evt) {
            SrcCanvas.style.cursor = 'move';
            var posl = windowToCanvas(evt.clientX, evt.clientY);
            var x = posl.x - pos.x;
            var y = posl.y - pos.y;
            pos=posl;
            RenderList[index].move(x,y);
            CanvasUpdate();
        };
        SrcCanvas.onmouseup = function () {
            SrcCanvas.onmousemove = Oldmove;
            SrcCanvas.onmouseup = null;
            SrcCanvas.style.cursor = 'default';
        };
    }
    SrcCanvas.onmousemove = function (evt) {
        var posl = windowToCanvas(evt.clientX, evt.clientY);
        var index = 0;
        var pos = windowToCanvas(event.clientX, event.clientY);
        for(var i = RenderList.length-1; i!=-1; i--){
            if(RenderList[i].InsideRect(pos.x, pos.y)){
                index = i;
                break;
            }
        }
        
        if (index == 0){
            SrcCanvas.style.cursor = 'default';
            return ;
        }
        SrcCanvas.style.cursor = 'move';
    };
}
//***********************************
//
// function 
//
//***********************************

function RenderListSwap(i, j){
    [RenderList[i],RenderList[j]] = [RenderList[j],RenderList[i]]
}

function CanvasUpdate(){
    var ctx = SrcCanvas.getContext("2d");
    ctx.clearRect(0,0,CanvasWidth,CanvasHeight);
    RenderList.forEach( function(e, i){
        if (e.type == "img"){
            ctx.drawImage(e.content, e.x, e.y, e.width, e.height);
        }
    })
}


function AddTextElement(type, content, x, y, height, width){
    RenderList.push(new ImageElement(type, content, x, y, height, width));
    CanvasUpdate();
}

function AddImageElement(type, content, x, y, height, width){
    RenderList.push(new ImageElement(type, content, x, y, height, width));
    CanvasUpdate();
}

function CanvasClear(CanvasName){
    var cv = document.getElementById(CanvasName);  
    var ctx = cv.getContext("2d");
    ctx.clearRect(0,0,CanvasWidth,CanvasHeight);
}

function DrawBackgroud(CanvasName, Img){
    AddImageElement(Img, 0, 0, CanvasWidth, CanvasHeight);
}

function DrawText(CanvasName, Text){
    AddTextElement(Text, 0, 0, CanvasWidth, CanvasHeight);
}

function DrawLogo(CanvasName, Img){
    AddImageElement(Img, 0, 0, 0, 0);
}