//***********************************
//
// class ImageElement and TextElement
//
//***********************************

SrcCanvas = $("srcCanvas");
RenderList = new Array();
var ChooseIndex = 0;

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


function WindowsUpdate(){
	RenderList[0].height = CanvasHeight = SrcCanvas.height = $("dstCanvas").height = $("right").clientHeight* 0.8;
	RenderList[0].width = CanvasWidth = SrcCanvas.width = $("dstCanvas").width = CanvasHeight * 0.75;
	$("right").style.marginLeft = $("left").clientWidth+"px";
	SrcCanvas.style.marginLeft = CanvasWidth *0.1 +"px";
	$("dstCanvas").style.marginLeft = CanvasWidth * 1.2+"px";
	CanvasUpdate();
}

function windowToCanvas(x,y) {
    var box = SrcCanvas.getBoundingClientRect();  
    return {
        x: x - box.left - (box.width - SrcCanvas.width) / 2,
        y: y - box.top - (box.height - SrcCanvas.height) / 2
    };
}

function CanvasInit(){
	RenderList.push(new ImageElement(null, 0, 0, SrcCanvas.width, SrcCanvas.height));
    SrcCanvas.onmousedown  = function (event){
        var index = 0;
        var pos = windowToCanvas(event.clientX, event.clientY);
        for(var i = RenderList.length-1; i!=-1; i--){
            if(RenderList[i].InsideRect(pos.x, pos.y)){
                index = i;
                break;
            }
        }
        ChooseIndex = index;
		CanvasUpdate();
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
        $("right").onmouseup = function () {
            SrcCanvas.onmousemove = Oldmove;
            $("right").onmouseup = null;
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
            SrcCanvas.style.cursor = 'e-resize';
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
function DrawChosenRect(ctx, e){
	ctx.lineWidth = 3;
	ctx.beginPath();
	ctx.setLineDash([10,5]);
	ctx.strokeStyle = '#666666';
	ctx.moveTo(e.x, e.y);
	ctx.lineTo(e.x + e.width, e.y);
	ctx.lineTo(e.x+e.width, e.y+e.height);
	ctx.lineTo(e.x, e.y + e.height);
	ctx.lineTo(e.x, e.y);
	ctx.stroke();
  ctx.setLineDash([]);
	ctx.strokeStyle = "black";
	TempArray = [[e.x, e.y], [e.x+e.width/2, e.y], [e.x+e.width, e.y], 
	 [e.x, e.y + e.height/2], [e.x+e.width, e.y+e.height2],
	[e.x, e.y+e.height], [e.x+e.width/2, e.y+e.height], [e.x+e.width, e.y+e.height]];
	for(var i=0; i<TempArray.length; i++){
		ctx.beginPath();
		ctx.arc(TempArray[i][0], TempArray[i][1], 5, 0, Math.PI*2, true);
		ctx.fillStyle = "white";
		ctx.fill();
    ctx.stroke();
	}
}

function RenderListSwap(i, j){
    [RenderList[i],RenderList[j]] = [RenderList[j],RenderList[i]]
}

function CanvasUpdate(){
    var ctx = SrcCanvas.getContext("2d");
    ctx.clearRect(0,0,CanvasWidth,CanvasHeight);
    RenderList.forEach( function(e, i){
        if (e.type == "img"){
			if (e.content == null){
				return ;
			}
            ctx.drawImage(e.content, e.x, e.y, e.width, e.height);
        }
    })
	if (ChooseIndex != 0){
		DrawChosenRect(ctx, RenderList[ChooseIndex]);
	}
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
	RenderList[0].content = Img;
	CanvasUpdate();
}

function DrawText(CanvasName, Text){
    AddTextElement(Text, 0, 0, CanvasWidth, CanvasHeight);
}

function DrawLogo(CanvasName, Img){
    AddImageElement(Img, 0, 0, 0, 0);
}
