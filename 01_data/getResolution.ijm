dir = getDirectory("Choose a Directory ");

fs = File.separator;

File.makeDirectory(dir+"resolution");

list = getFileList(dir);

//print(list.length);

for (i=0; i<list.length; i++) {
	if (endsWith(list[i], ".tif")) {
//		print(list[i]);
		preprocess(dir,list[i]);
	}
}

function preprocess(imdir,imfi) {
	open(imdir+imfi); 
	getDimensions(width, height, channels, slices, frames);
	removeIdx = indexOf(imfi, '.'); 
	imroot = substring(imfi, 0, removeIdx); 
	getVoxelSize(width, height, depth, unit);
	
	f = File.open(imdir + "resolution" + fs + imroot + "_resolution.txt");
	print(f, "z y x resolution");
	print(f, depth + " " + height  + " " + width + " " + unit);
	File.close(f);

	close();
	
}
