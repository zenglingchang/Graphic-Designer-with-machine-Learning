function initConnect(){
	var ws_url = "ws://127.0.0.1:8080/connect";
    ws = new WebSocket(ws_url);
	ws.onopen = openHandler;
    ws.onmessage = messageHandler;
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
	console.log(json);
}

