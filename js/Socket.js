function ConnectInit(){
	var ws_url = "ws://127.0.0.1:8080/connect";
    ws = new WebSocket(ws_url);
	ws.onopen = openHandler;
    ws.onmessage = messageHandler;
}

function sendImg(type, Img){
	sendMessage([type,Img]);
}

function sendImgList(type, ImgList){
	sendMessage([type,ImgList]);
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
			for(var i = 0; i<RenderList.length; i++){
				if(i == 0) continue;
				RenderList[i].x = DesignList[i][0]*CanvasX;
				RenderList[i].y = DesignList[i][1]*CanvasY;
				RenderList[i].width = DesignList[i][2]*CanvasWidth;
				RenderList[i].height = DesignList[i][3]*CanvasHeight;
			}
			CanvasUpdate();
			break;

	}
}

