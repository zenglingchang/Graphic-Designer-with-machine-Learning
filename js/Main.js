function Init(){
    CanvasInit();
    ConnectInit();
    FileHandlerInit();
    document.getElementById('Reset').onclick = function() {
        ResetCanvas();
    }
}
Init();
