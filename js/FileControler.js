function FileHandlerInit(){
	document.getElementById('Logo').onclick = function(){
		console.log('click logo!');
		document.getElementById('LogoFile').click();
	};
	document.getElementById('Background').onclick = function(){
		document.getElementById('BackgroundFile').click();
	}
}

