function Init(){
    CanvasInit();
	ConnectInit();
	FileHandlerInit();
    document.getElementById('Reset').onclick = function() {
        CanvasClear('srcCanvas');
        CanvasClear('dstCanvas');
    }

}
Init();
