//***********************************
//
// Static var
//
//***********************************

RenderList       = new Array();
var CanvasX      = 0;
var CanvasY      = 0;
var CanvasHeight = 40; 
var CanvasWidth  = 30;
DrectionCursor   = ["nw-resize", "n-resize", "ne-resize", "w-resize", "e-resize", "sw-resize", "s-resize", "se-resize", "move"];    
var ChooseIndex  = 0;

//***********************************
//
// class ImageElement and TextElement
//
//***********************************

class Element{
    constructor(x, y, width, height){
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }
    
    InsideRect(MouseX, MouseY){
        return  this.x - 5  < MouseX && MouseX  <  this.width + this.x + 5 && this.y  - 5 < MouseY && MouseY < this.height + this.y + 5;
    }
    
    GetDirection(MouseX, MouseY){
        var i = 0;
        TempArray = this.GetEightDirection();
        for( ; i<TempArray.length ; i++){
            if( (TempArray[i][0] - MouseX)**2 + (TempArray[i][1] - MouseY)**2 < 50)
                break;
        }
        return DrectionCursor[i];
    }
    
    DirectFunction(Direction, OffsetX, OffsetY){
        switch (Direction) {
            case "move":
                this.x += OffsetX;
                this.y += OffsetY;
                break;
            case "nw-resize":
                this.x += OffsetX;
                this.y += OffsetY;
                this.width -= OffsetX;
                this.height -= OffsetY;
                break;
            case "n-resize":
                this.y += OffsetY;
                this.height -= OffsetY;
                break;
            case "ne-resize":
                this.y += OffsetY;
                this.width += OffsetX;
                this.height -= OffsetY;
                break;
            case "w-resize":
                this.x += OffsetX;
                this.width -= OffsetX;
                break;
            case "e-resize":
                this.width += OffsetX;
                break;
            case "sw-resize":
                this.x += OffsetX;
                this.width -= OffsetX;
                this.height += OffsetY;
                break;
            case "s-resize":
                this.height += OffsetY;
                break;
            case "se-resize":
                this.height += OffsetY;
                this.width += OffsetX;
                break;
        }
        this.width  =  (this.width < 10) ? 10: this.width;
        this.height = (this.height < 10) ? 10: this.height;
    }
    
    GetEightDirection(){
        return [[this.x, this.y], [this.x+this.width/2, this.y], [this.x+this.width, this.y], 
            [this.x, this.y + this.height/2], [this.x+this.width, this.y+this.height/2],
            [this.x, this.y+this.height], [this.x+this.width/2, this.y+this.height], [this.x+this.width, this.y+this.height]];
    }
    
}

class ImageElement extends Element{
    constructor(content, x, y, width, height){
        super(x, y, width, height);
        this.type = 'img';
        this.content = content;
        if (height == 0 || width == 0){
            this.height = content.height;
            this.width = content.width;
        }
    }
    
    GetDirection(MouseX, MouseY){
        var i = 0;
        TempArray = this.GetEightDirection();
        for( ; i<TempArray.length ; i++){
            if( (TempArray[i][0] - MouseX)**2 + (TempArray[i][1] - MouseY)**2 < 50)
                break;
        }
        return DrectionCursor[i];
    }

}

class TextElement extends Element{
    constructor(content, x, y, width, height){
        super(x, y, width, height);
        this.type = 'text';
        this.content = content;
    }
    
}

//***********************************
//
// Mouse & Window Event Function 
//
//***********************************

function windowToCanvas(x,y) {
    var box = $('srcCanvas').getBoundingClientRect();  
    return {
        x: x - box.left - (box.width - $('srcCanvas').width) / 2,
        y: y - box.top - (box.height - $('srcCanvas').height) / 2
    };
}

function WindowUpdate(){
    ScaleX                          = CanvasWidth;
    ScaleY                          = CanvasHeight;
    $('left').style.height          = (document.body.clientHeight - $('head').clientHeight) + 'px';
    $('right').style.height         = (document.body.clientHeight - $('head').clientHeight) + 'px';
    $('right').style.width          = (document.body.clientWidth  - $('left').clientWidth)  + 'px';
    $("right").style.marginLeft     = $("left").clientWidth+"px";
    $('srcCanvas').width            = $("right").clientWidth;
    $('srcCanvas').height           = $("right").clientHeight;
	CanvasHeight                    = $('srcCanvas').height *0.8;
	CanvasWidth                     = CanvasHeight * 0.75;
	CanvasX                         = $('srcCanvas').width * 0.4;
    CanvasY                         = CanvasHeight *0.1;
    ScaleX                          = CanvasWidth/ScaleX;
    ScaleY                          = CanvasHeight/ScaleY;
    ResizeRenderList(ScaleX, ScaleY);
	DrawElement();
}

function DefaultMove(evt) {

        var pos = windowToCanvas(event.clientX, event.clientY);
        if (ChooseIndex != 0){
            $('srcCanvas').style.cursor = RenderList[ChooseIndex].GetDirection(pos.x - CanvasX, pos.y - CanvasY);
        }
        
        var index = 0;
        for(var i = RenderList.length-1; i!=-1; i--){
            if(RenderList[i].InsideRect(pos.x - CanvasX, pos.y- CanvasY)){
                index = i;
                break;
            }
        }
        
        if (index == 0){
            $('srcCanvas').style.cursor = 'default';
            return ;
        }
}

function CanvasInit(){
    $('srcCanvas').onmousedown  = function (event){
        var index = 0;
        var pos = windowToCanvas(event.clientX, event.clientY);
        for(var i = RenderList.length-1; i!=-1; i--){
            if(RenderList[i].InsideRect(pos.x - CanvasX, pos.y - CanvasY)){
                index = i;
                break;
            }
        }
        ChooseIndex = index;
		CanvasUpdate();
        if (index == 0){
            return ;
        }
        $('srcCanvas').onmousemove = function (evt) {
            var posl = windowToCanvas(evt.clientX, evt.clientY);
            var x = posl.x - pos.x;
            var y = posl.y - pos.y;
            pos=posl;
            RenderList[index].DirectFunction($('srcCanvas').style.cursor, x, y);
            CanvasUpdate();
        };
        $("right").onmouseup = function () {
            $('srcCanvas').onmousemove = DefaultMove;
            $("right").onmouseup = null;
        };
    }
    WindowUpdate();
    window.onresize       = WindowUpdate;   
    $('srcCanvas').onmousemove = DefaultMove;
    RenderList.push(new ImageElement(null, 0, 0, CanvasWidth, CanvasHeight));
}

//***********************************
//
// Canvas Draw function 
//
//***********************************

function CanvasUpdate(){
    ClearCanvas();
    DrawElement();
    DrawChosenRect();
    DrawTools();
}

function ClearCanvas(){
    var ctx = $('srcCanvas').getContext("2d");
    ctx.fillStyle="#CCCAC4";
    ctx.fillRect(0, 0, $('srcCanvas').width, $('srcCanvas').height);
    ctx.fillStyle = 'white';
    ctx.fillRect(CanvasX, CanvasY, CanvasWidth, CanvasHeight);
}

function DrawTools(){
    var ctx = $('srcCanvas').getContext("2d");
}

function DrawChosenRect(){
    if (ChooseIndex == 0) return ;
    var ctx = $('srcCanvas').getContext("2d");
    e = RenderList[ChooseIndex];
    var TempX = CanvasX + e.x;
    var TempY = CanvasY + e.y;
	ctx.lineWidth = 3;
	ctx.beginPath();
	ctx.setLineDash([10,5]);
	ctx.strokeStyle = '#666666';
	ctx.moveTo(TempX, TempY);
	ctx.lineTo(TempX + e.width, TempY);
	ctx.lineTo(TempX+e.width, TempY+e.height);
	ctx.lineTo(TempX, TempY + e.height);
	ctx.lineTo(TempX, TempY);
	ctx.stroke();
    ctx.setLineDash([]);
	ctx.strokeStyle = "black";
	TempArray = e.GetEightDirection();
	for(var i=0; i<TempArray.length; i++){
		ctx.beginPath();
		ctx.arc(CanvasX+ TempArray[i][0], CanvasY+TempArray[i][1], 5, 0, Math.PI*2, true);
		ctx.fillStyle = "white";
		ctx.fill();
    ctx.stroke();
	}
}

function DrawElement(){
    var ctx = $('srcCanvas').getContext("2d");
    ClearCanvas();
    RenderList.forEach( function(e, i){
        if (e.type == "img"){
			if (e.content == null){
				return ;
			}
            if(CanvasWidth < e.x ||  e.x + e.width < 0 || CanvasHeight < e.y || e.y + e.height < 0) return ;
            if(e.x >= 0){
                var DrawX = e.x + CanvasX;
                var CutX = 0;
                var DrawWidth = (e.x + e.width < CanvasWidth) ? e.width: CanvasWidth - e.x;
            }
            else {
                var DrawX = CanvasX;
                var CutX = - e.x;
                var DrawWidth = (e.x + e.width < CanvasWidth) ? e.width - CutX : CanvasWidth;
            }
            
            if(e.y >= 0){
                var DrawY = e.y + CanvasY;
                var CutY = 0;
                var DrawHeight = ( e.y + e.height < CanvasHeight) ? e.height: CanvasHeight -e.y;
            }
            else{
                var DrawY = CanvasY;
                var CutY = - e.y;
                var DrawHeight = ( e.y + e.height < CanvasHeight) ? e.height - CutY: CanvasHeight;
            }
            ctx.drawImage(e.content, CutX*e.content.width/e.width, CutY*e.content.height/e.height, e.content.width*DrawWidth/e.width, e.content.height*DrawHeight/e.height, DrawX, DrawY, DrawWidth, DrawHeight);
        }
    })
}

function ElementMoveUp(index){
    if(index >= RenderList.length - 1) return ;
    [RenderList[index], RenderList[index + 1]] = [RenderList[index + 1], RenderList[index]];
}

function ElementMoveUp(index){
    if(index < 2) return ;
    [RenderList[index], RenderList[index - 1]] = [RenderList[index - 1], RenderList[index]];
}

function ResizeRenderList(ScaleX, ScaleY){
    RenderList.forEach( function(e, i){
        e.x *= ScaleX;
        e.y *= ScaleY;
        e.width *= ScaleX;
        e.height *= ScaleY;
    });
}

function AddTextElement(content, x, y, height, width){
    RenderList.push(new TextElement(content, x, y, height, width));
    CanvasUpdate();
}

function AddImageElement(content, x, y, height, width){
    RenderList.push(new ImageElement(content, x, y, height, width));
    CanvasUpdate();
}

function ResetCanvas(){
    RenderList.length = 1;
    RenderList[0].content = null;
    ChooseIndex = 0;
    ClearCanvas();
}

function DrawBackgroud(Img){
	RenderList[0].content = Img;
	DrawElement();
}

function DrawText(Text){
    AddTextElement(Text, 0, 0, CanvasWidth, CanvasHeight);
}

function DrawLogo(Img){
    AddImageElement(Img, 0, 0, 0, 0);
}
