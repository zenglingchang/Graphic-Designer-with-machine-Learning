function ConnectInit(){
	var ws_url = "ws://127.0.0.1:8080/connect";
    ws = new WebSocket(ws_url);
	ws.onopen = openHandler;
    ws.onmessage = messageHandler;
}

function sendImg(command, Img){
	sendMessage([command,Img]);
}

function sendImgList(command, ImgList){
	sendMessage([command,ImgList]);
}

function sendMessage(msgArray) {
	var msg = JSON.stringify(msgArray);
    ws.send(msg);
}
	
function openHandler(e){
	console.log("open socket!");
};

function messageHandler(e){
	json = JSON.parse(e.data);
	switch (json[0]){
		case "Score":
			DrawScore(json[1]);
			break;
		case "Design":
			DesignList = json[1];
			console.log('GetDesgin', DesignList.toSource());
			for(var i = 0; i<DesignList.length; i++){
				RenderList[i+1].height = DesignList[i][0]*CanvasHeight;
        RenderList[i+1].width = DesignList[i][1]*CanvasWidth;
        RenderList[i+1].y = Math.min(DesignList[i][2]*CanvasHeight, CanvasHeight - RenderList[i+1].height);
        RenderList[i+1].x = Math.min(DesignList[i][3]*CanvasWidth, CanvasWidth - RenderList[i+1].width);
			}
      console.log('GetDesgin',RenderList);
			CanvasUpdate();
			break;

	}
}

