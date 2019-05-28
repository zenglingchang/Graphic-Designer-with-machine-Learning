function Init(){
    CanvasInit();
    ConnectInit();
    FileHandlerInit();
    document.getElementById('Reset').onclick = function() {
        ResetCanvas();
    }
	document.getElementById('GetSensitiveMap').onclick = function(){
		SendImg('GetSensitiveMap',GetCanvasContent());
	}
	document.getElementById('GetScore').onclick = function() {
        sendImg('GetScore', GetCanvasContent());
	};
	document.getElementById('ReDesign').onclick = function() {
        if ($('#PerChoose')[0].value != "NoChoose")
            sendImgList('GetDesign', [$('#PerChoose')[0].value].concat(GetElementList()));
        else
            alert("Please Choose personality!");
	}
}
Init();
