function Init(){
    CanvasInit();
    ConnectInit();
    FileHandlerInit();
    document.getElementById('Reset').onclick = function() {
        ResetCanvas();
    }
	document.getElementById('GetScore').onclick = function() {
        if ($('#PerChoose')[0].value != "NoChoose")
            sendImg('GetScore', $('#PerChoose')[0].value, GetCanvasContent());
        else
            alert("Please Choose personality!");
	};
	document.getElementById('ReDesign').onclick = function() {
        if ($('#PerChoose')[0].value != "NoChoose")
            sendImgList('GetDesign', $('#PerChoose')[0].value, GetElementList());
        else
            alert("Please Choose personality!");
	}
}
Init();
