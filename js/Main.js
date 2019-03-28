function Init(){
    CanvasInit();
    ConnectInit();
    FileHandlerInit();
    document.getElementById('Reset').onclick = function() {
        CanvasClear('srcCanvas');
        CanvasClear('dstCanvas');
    }
	WindowsUpdate();
	window.onresize= function(){
		WindowsUpdate();
	};
}
Init();
