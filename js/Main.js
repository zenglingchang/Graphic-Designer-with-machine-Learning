function Init(){
    CanvasInit();
    ConnectInit();
    FileHandlerInit();
    document.getElementById('Reset').onclick = function() {
        ResetCanvas();
    }
	document.getElementById('GetScore').onclick = function(){
		sendImg('GetScore', GetCanvasContent());
	};
	document.getElementById('ReDesign').onclick = function(){
		sendImgList('GetDesign', GetElementList());
	}
}
Init();
