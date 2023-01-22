macro 'Measure SL and SA' {

	dir_bf_lif = getDirectory("Choose folder with lif files ");
	list = getFileList(dir_bf_lif);

	//remove dir in list ( LIF_FILE only )
	for (i=0; i<list.length;){
		
		// print(list.length, i, list[i]);
		if(endsWith(list[i], ".lif")){ 
			i++;
		}
		else{
			// print(i, list[i]);
			list = Array.deleteIndex(list, i);
		}
	}
	setBatchMode(true); //hide processing windows in ImageJ
	
	//Create folders for the tifs
	dirName_bf_lif = File.getName(dir_bf_lif); //get folder name
	dir_tiff = dir_bf_lif + File.separator + dirName_bf_lif + "--TIFF";
	dir_result = dir_bf_lif + File.separator + dirName_bf_lif + "--Result";
	dir_cropped = dir_bf_lif + File.separator + dirName_bf_lif + "--Cropped";
	if (File.exists(dir_tiff) == false) { File.makeDirectory(dir_tiff); }
	if (File.exists(dir_result) == false) { File.makeDirectory(dir_result); }
	if (File.exists(dir_cropped) == false) { File.makeDirectory(dir_cropped); }
		
		
		
	for (i=0; i<list.length; i++){

		showProgress( i+1, list.length);
		print("\n\nprocessing ... " + (i+1) + "/" + list.length + "\n         LIF_FILE : " + list[i]);
		path = dir_bf_lif + list[i];

		
		//How many series in this lif file?
		run("Bio-Formats Macro Extensions");
		Ext.setId(path);//-- Initializes the given path (filename).
		Ext.getSeriesCount(seriesCount); //-- Gets the number of image series in the active dataset.


		for (j=1; j<=seriesCount; j++){
		
			run("Bio-Formats", "open=path autoscale color_mode=Default view=Hyperstack stack_order=XYCZT series_"+j);
	
			file_name  = File.nameWithoutExtension;
			file_name  = replace(file_name, ".lif", "");
			image_name = getInfo("Image name");
			//seriesname=getTitle(); //get window_name, not image_name
			seN = file_name + " - " + image_name;
			print("processing ... " + " series " + j + "/" + seriesCount + " in " + (i+1) + "/" + list.length + "\n         " + seN);
			getDimensions(width, height, channels, slices, frames);


			//Process and save
			if (slices>0) {
				saveAs("TIFF", dir_tiff + File.separator + seN + ".tif");

				//Process to cropped image
					setOption("ScaleConversions", true);
					run("8-bit");

					run("Duplicate...", "title=Image_copy"); selectWindow("Image_copy");
					makeRectangle(50, 700, 1950, 700); //crop_window ( can change )

					run("Duplicate...", "title=cropped_Image"); selectWindow("cropped_Image");
					saveAs("Tiff", dir_cropped + File.separator + seN + "--cropped.tif");


				//Process to threshold and measurement
					run("Duplicate...", "title=cropped_Image_analysis"); selectWindow("cropped_Image_analysis");
					run("Auto Threshold", "method=Triangle white");
					setOption("BlackBackground", true);
					run("Convert to Mask");
					saveAs("Tiff", dir_result + File.separator + seN + "_mask");


				//SL and SA measurement
					run("Set Measurements...", "area feret's display redirect=None decimal=2");
					run("Analyze Particles...", "size=1000000-8000000 show=Outlines display add"); //region_size ( can change )
					saveAs("Tiff", dir_result + File.separator + seN + "_outline");

					roiManager("show all with labels");
					if ( RoiManager.size == 1){ //success to get fish
						saveAs    (  "Results", dir_result + File.separator + seN + "_Analysis.csv");
						roiManager(  "Save",    dir_result + File.separator + seN + "_RoiSet.zip");
						roiManager("Deselect");
						roiManager("Delete"); //delete ROI, otherwise, it increases infinitely
					}
					else { 
						print("         number of ROI = ", RoiManager.size);
						print(" #### ERROR : Number of ROI not = 1\n");
						
						if ( RoiManager.size > 1){ //delete ROI, otherwise, it increases infinitely
							roiManager("Deselect");
							roiManager("Delete");
						}
					}
				

				//Update Log
					selectWindow("Log");
					saveAs("Text", dir_bf_lif + File.separator + "Log.txt");

				//Clear and close all windows
					// waitForUser("Pause", j+" fish done.");
					run("Clear Results");
					run("Close All");

			}
			else{ showMessage(" No Lif inside") }

		}
	}

	showMessage(" -- finished --");
	setBatchMode(false);

}// macro
