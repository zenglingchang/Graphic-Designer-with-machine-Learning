//***********************************
//
// Static var
//
//***********************************
var E = window.wangEditor;
var editor = new E('#text-toolbar', '#text-rect');
editor.customConfig.onchange = function (html) {
	if(RenderList[ChooseIndex].type == 'text'){
		RenderList[ChooseIndex].html = html;
	}
	console.log(html);
    return ;
	var json = editor.txt.getJSON();
	var jsonStr = JSON.stringify(json);
	console.log(json);
	console.log(jsonStr);
}
editor.create();
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
    
 

}

class TextElement extends Element{
    constructor(content, x, y, width, height, html){
        super(x, y, width, height);
        this.type = 'text';
        this.content = content;      
		this.html = html;
		if (height == 0 || width == 0){
            this.height = content.height;
            this.width = content.width;
        }
    }

	GetDirection(MouseX, MouseY){
        if (MouseX > this.x + 20 && MouseX < this.x + this.width-20 && MouseY > this.y + 20 && MouseY < this.y + this.height -20){
			return 'text';
		}
		else 
        	return 'move';
    }

	DirectFunction(Direction, OffsetX, OffsetY){
        switch (Direction) {
            case "move":
                this.x += OffsetX;
                this.y += OffsetY;
                break;
			case "text":
				break;
		}
	}
    
	GetEightDirection(){
    	return [];
	}
}

//***********************************
//
// Mouse & Window Event Function 
//
//***********************************

function windowToCanvas(x,y) {
    var box = $('#srcCanvas')[0].getBoundingClientRect();  
    return {
        x: x - box.left - (box.width - $('#srcCanvas')[0].width) / 2,
        y: y - box.top - (box.height - $('#srcCanvas')[0].height) / 2
    };
}

function CanvasToWindow(x,y){
	var box = $('#srcCanvas')[0].getBoundingClientRect();
	return {
		x: x + box.left,
		y: y + box.top
	};
}

function WindowUpdate(){
    ScaleX                          = CanvasWidth;
    ScaleY                          = CanvasHeight;
    
	// Update left, right, canvas
	$('#left').height (  document.body.clientHeight - $('#head').height() );
    $('#right').height(  document.body.clientHeight - $('#head').height() );
    $('#right').width  (  document.body.clientWidth - $('#left').width()  );
    $("#right").css( 'marginLeft', $("#left").width() );
    $('#srcCanvas')[0].width  = $("#right").width()  ;
    $('#srcCanvas')[0].height = $("#right").height() ;
	
	// Update text editor
	$(".toolbar").css('marginLeft', $('#left').width());
	$('.toolbar').css('background-color','white');
	$('.toolbar').css('position', 'fixed');
	$('.toolbar').css('z-index',100);
	$('.toolbar').css('width','100%');
	
	CanvasHeight                    = $('#srcCanvas')[0].height * 0.8;
	CanvasWidth                     = CanvasHeight * 0.75;
	CanvasX                         = $('#srcCanvas')[0].width  * 0.4;
    CanvasY                         = CanvasHeight *0.1;
    ScaleX                          = CanvasWidth/ScaleX;
    ScaleY                          = CanvasHeight/ScaleY;
    ResizeRenderList(ScaleX, ScaleY);
	DrawElement();
}

function DefaultMove(evt) {

        var pos = windowToCanvas(event.clientX, event.clientY);
        if (ChooseIndex != 0){
            $('#srcCanvas')[0].style.cursor = RenderList[ChooseIndex].GetDirection(pos.x - CanvasX, pos.y - CanvasY);
        }
        
        var index = 0;
        for(var i = RenderList.length-1; i!=-1; i--){
            if(RenderList[i].InsideRect(pos.x - CanvasX, pos.y- CanvasY)){
                index = i;
                break;
            }
        }
        
        if (index == 0){
            $('#srcCanvas')[0].style.cursor = 'default';
            return ;
        }
}

function SaveEditor(index){
	console.log('asdgas');
	var node = $('div.w-e-text')[0];
	domtoimage.toPng(node)
    	.then(function (dataUrl) {
        	var text = new Image();
        	text.onload = function(event){
				console.log(dataUrl);
				console.log()
				RenderList[index].html = editor.txt.html();
				RenderList[index].content = text;
				RenderList[index].width = text.width;
				RenderList[index].height = text.height;
				$('#text-rect').css('visibility', 'hidden');
				CanvasUpdate();
			}
			text.src = dataUrl;
    	})
    	.catch(function (error) {
        	console.error('oops, something went wrong!', error);
    	});

}

function CanvasInit(){
    $('#srcCanvas')[0].onmousedown  = function (event){
        var index = 0;
        var pos = windowToCanvas(event.clientX, event.clientY);
        for(var i = RenderList.length-1; i!=-1; i--){
            if(RenderList[i].InsideRect(pos.x - CanvasX, pos.y - CanvasY)){
                index = i;
                break;
            }
        }
		if(RenderList[ChooseIndex].type == 'text'){
			SaveEditor(ChooseIndex);
		}
        ChooseIndex = index;
		CanvasUpdate();
        if (index == 0){
            return ;
        }
        $('#srcCanvas')[0].onmousemove = function (evt) {
            var posl = windowToCanvas(evt.clientX, evt.clientY);
            var x = posl.x - pos.x;
            var y = posl.y - pos.y;
            pos=posl;
            RenderList[index].DirectFunction($('#srcCanvas')[0].style.cursor, x, y);
            CanvasUpdate();
        };
        $("#right")[0].onmouseup = function () {
            $('#srcCanvas')[0].onmousemove = DefaultMove;
            $("#right")[0].onmouseup = null;
        };
    }
	WindowUpdate();
    window.onresize       = WindowUpdate;   
    $('#srcCanvas')[0].onmousemove = DefaultMove;
 	$('#element-toolbar').toolbar({content:'#element-toolbar-options', position:'left', event: 'click', hideOnClick: true});
	$('#element-toolbar').css('visibility', 'hidden');
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
    DrawElementToolbar();
}

function ClearCanvas(){
    var ctx = $('#srcCanvas')[0].getContext("2d");
    ctx.fillStyle="#CCCAC4";
    ctx.fillRect(0, 0, $('#srcCanvas')[0].width, $('#srcCanvas')[0].height);
    ctx.fillStyle = 'white';
    ctx.fillRect(CanvasX, CanvasY, CanvasWidth, CanvasHeight);
}

function DrawElementToolbar(){
	if(ChooseIndex == 0){
		$('#element-toolbar').css('visibility', 'hidden');
		return ;
	}
	pos = CanvasToWindow(CanvasX + RenderList[ChooseIndex].x - 50, CanvasY + RenderList[ChooseIndex].y);
	$('#element-toolbar').css('marginLeft', pos.x);
	$('#element-toolbar').css('marginTop', pos.y);
	$('#element-toolbar').css('visibility', '');
	
}

function DrawChosenRect(){
    if (ChooseIndex == 0) return ;
    var ctx = $('#srcCanvas')[0].getContext("2d");
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
    var ctx = $('#srcCanvas')[0].getContext("2d");
    ClearCanvas();
    RenderList.forEach( function(e, i){
        if (e.type == "img" || (e.type == "text" && i != ChooseIndex)) {
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
		else if ( i == ChooseIndex ){
			console.log('show editor');
			pos = CanvasToWindow(CanvasX + e.x, CanvasY + e.y);
        	editor.txt.html(e.html);
			$('#text-rect').css('marginLeft', pos.x);
			$('#text-rect').css('marginTop', pos.y);
			$('#text-rect').css('visibility', '');
		}
    })
}

function ElementMoveUp(){
    if(ChooseIndex >= RenderList.length - 1 || ChooseIndex <= 0) return ;
    [RenderList[ChooseIndex], RenderList[ChooseIndex + 1]] = [RenderList[ChooseIndex + 1], RenderList[ChooseIndex]];
	ChooseIndex += 1;
	CanvasUpdate();
}

function ElementMoveDown(){
    if(ChooseIndex < 2 || ChooseIndex >= RenderList.length) return ;
    [RenderList[ChooseIndex], RenderList[ChooseIndex - 1]] = [RenderList[ChooseIndex - 1], RenderList[ChooseIndex]];
	ChooseIndex -= 1;
	CanvasUpdate();
}

function ElementRemove(){
	if(ChooseIndex > 0 && ChooseIndex < RenderList.length){
		RenderList.splice(ChooseIndex,1);
	}
	ChooseIndex = 0;
	CanvasUpdate();
}

function ResizeRenderList(ScaleX, ScaleY){
    RenderList.forEach( function(e, i){
		if(i == 0){
			e.width  = CanvasWidth;
			e.height = CanvasHeight;
			return ;
		}
        e.x *= ScaleX;
        e.y *= ScaleY;
        e.width *= ScaleX;
        e.height *= ScaleY;
    });
}

function AddTextElement(content, x, y, height, width){
	console.log('add text!');
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

function DrawText(){
	editor.txt.html('<p> Please input</p><p><br></p>');
	var node = document.getElementById('text-rect');
	domtoimage.toPng(node)
    	.then(function (dataUrl) {
        	var text = new Image();
        	text.onload = function(event){
				AddTextElement(text, 0, 0, 0, 0);
			}
			text.src = dataUrl;
    	})
    	.catch(function (error) {
        	console.error('oops, something went wrong!', error);
    	});
}

function DrawLogo(Img){
    AddImageElement(Img, 0, 0, 0, 0);
}
