function AddBackgroud(e){
	var file = document.getElementById('BackgroundFile').files[0];
	if(file){
		var reader = new FileReader();
		reader.onload = function ( event ){
            var image = new Image(); 
            image.onload = function(){
                DrawBackgroud(image);
                document.getElementById('BackgroundFile').value = null;
            }
            image.src = event.target.result;
		}
	}
	reader.readAsDataURL(file);
}

function AddLogo(e){
	var file = document.getElementById('LogoFile').files[0];
	if(file){
		var reader = new FileReader();
		reader.onload = function ( event ){
            var image = new Image();
            image.onload = function(){
                DrawLogo(image);
                document.getElementById('LogoFile').value = null;
            }
            image.src = event.target.result;

		}
	}
	reader.readAsDataURL(file);
}

function FileHandlerInit(){
	document.getElementById('SendSrc').onclick = function(){
		SendLogo();
		SendBackgroud();
	};
	document.getElementById('Logo').onclick = function(){
		document.getElementById('LogoFile').click();
	};
	document.getElementById('Background').onclick = function(){
		document.getElementById('BackgroundFile').click();
	}
	document.getElementById('LogoFile').addEventListener('change', AddLogo, false);
	document.getElementById('BackgroundFile').addEventListener('change', AddBackgroud, false);
}

