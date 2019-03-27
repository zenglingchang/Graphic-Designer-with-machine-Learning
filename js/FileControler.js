function AddBackgroud(e){
	var file = document.getElementById('BackgroundFile').files[0];
	if(file){
		var reader = new FileReader();
		reader.onload = function ( event ){
			var txt = event.target.result;
		}
	}
	reader.readAsBinaryString(file);
}

function AddLogo(e){
	var file = document.getElementById('LogoFile').files[0];
	if(file){
		var reader = new FileReader();
		reader.onload = function ( event ){
			var txt = event.target.result;
			console.log(txt);
		}
	}
	reader.readAsDataURL(file);
}

function SendLogo(){
	var file = document.getElementById('LogoFile').files[0];
	if(file){
		var reader = new FileReader();
		reader.onload = function ( event ){
			var txt = event.target.result;
			sendImg(txt);
		}
	reader.readAsDataURL(file);
	}
}

function SendBackgroud(){
	var file = document.getElementById('BackgroundFile').files[0];
	if(file){
		var reader = new FileReader();
		reader.onload = function ( event ){
			var txt = event.target.result;
			sendImg(txt);
		}
	reader.readAsDataURL(file);
	}
}


function FileHandlerInit(){
	document.getElementById('SendSrc').onclick = function(){
		SendLogo();
		SendBackgroud();
	};
	document.getElementById('Logo').onclick = function(){
		console.log('click logo!');
		document.getElementById('LogoFile').click();
	};
	document.getElementById('Background').onclick = function(){
		document.getElementById('BackgroundFile').click();
	}
	document.getElementById('LogoFile').addEventListener('change', AddLogo, false);
	document.getElementById('BackgroundFile').addEventListener('change', AddBackgroud, false);
}

