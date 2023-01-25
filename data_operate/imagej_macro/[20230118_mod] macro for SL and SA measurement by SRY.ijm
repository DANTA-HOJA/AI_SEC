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
	dir_metaimg = dir_bf_lif + File.separator + dirName_bf_lif + "--MetaImage";
	dir_result = dir_bf_lif + File.separator + dirName_bf_lif + "--Result";
	if (File.exists(dir_tiff) == false) { File.makeDirectory(dir_tiff); }
	if (File.exists(dir_metaimg) == false) { File.makeDirectory(dir_metaimg); }
	if (File.exists(dir_result) == false) { File.makeDirectory(dir_result); }
		


	fish_id = 0;
	for (i=0; i<list.length; i++){

		showProgress( i+1, list.length);
		print("\n\n");
		print("|-----------------------------------------  Processing ..." + (i+1) + "/" + list.length, "  -----------------------------------------");
		print("|");
		print("|         LIF_FILE : " + list[i]);
		print("|");
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
			print("|-- processing ... " + " series " + j + "/" + seriesCount + " in " + (i+1) + "/" + list.length);
			print("|         " + seN);
			getDimensions(width, height, channels, slices, frames);
			print("|         Dimensions : ", width, height, channels, slices, frames, "( width, height, channels, slices, frames )");


			//Get fish_id and create subfolder in "--MetaImage" by fish_id
			seN_split = split(seN, "fish _-");
			// Array.print(seN_split); //can print the array on a single line.
			//If 'fish_id' repeated, skip it.
			if (fish_id == seN_split[9]) {
				print("| #### WARNING : Detect ' fish_id ' is same as previous processed fish --> skip this image "); //WARNING:
				print("|");
				continue;
			}
			else { 
				fish_id = seN_split[9]; 
				// print(fish_id); 
			}
			metaimg_subfolder = dir_metaimg + File.separator + "Fish_" + fish_id;
			File.makeDirectory(metaimg_subfolder);


			//Process and save
			if (slices > 0) {

					//Pick up focused slice if slices > 1
					//	Plugin ref : https://sites.google.com/site/qingzongtseng/find-focus
					//	Algorithm  : autofocus algorithm "Normalized variance"  (Groen et al., 1985; Yeo et al., 1993).
					if (slices > 1) {
						print("| #### WARNING : Number of Slices > 1, run ' Find focused slices ' "); //WARNING:
						run("Find focused slices", "select=100 variance=0.000 select_only"); 
					}
					saveAs("TIFF", dir_tiff + File.separator + seN + ".tif");


				//Process to cropped image
					setOption("ScaleConversions", true); //scaling the pixel value when converting the image type (depth)
					run("8-bit");

					//Consociate the unit ( relationship between micron and pixel )
					//	To prevent some series of images are in different scales, for example, "20220708_CE009_palmskin_8dpf.lif"
					//	Microscope Metadata : 1 pixel = 0.0000065 m = 6.5 micron
					run("Set Scale...", "distance=0.3076923076923077 known=1 unit=micron"); 
					

					run("Duplicate...", "title=Image_copy"); selectWindow("Image_copy");
					makeRectangle(50, 700, 1950, 700); //crop_window ( can change )

					run("Duplicate...", "title=cropped_Image"); selectWindow("cropped_Image");
					cropped_name = seN + "_Cropped.tif";
					saveAs("Tiff", metaimg_subfolder + File.separator + cropped_name);


				//Process to threshold and measurement
					run("Duplicate...", "title=cropped_Image_threshold"); selectWindow("cropped_Image_threshold");
					run("Auto Threshold", "method=Triangle white");
					setOption("BlackBackground", true);
					run("Convert to Mask");
					threshold_name = seN + "_Threshold.tif";
					saveAs("Tiff", metaimg_subfolder + File.separator + threshold_name);


				//SL and SA measurement
					run("Set Measurements...", "area feret's display redirect=None decimal=2");
					run("Analyze Particles...", "size=800000-8000000 show=Masks display include add"); //region_size ( can change )
					run("Convert to Mask");
					mask_name = seN + "--Mask.tif";
					saveAs("Tiff", metaimg_subfolder + File.separator + mask_name);

					roiManager("show all with labels");
					if ( RoiManager.size == 1){ //success to get fish
						imageCalculator("AND create", cropped_name, mask_name);
						saveAs    (  "Tiff",    metaimg_subfolder + File.separator + seN + "--MIX.tif");
						roiManager(  "Save",    metaimg_subfolder + File.separator + seN + "_RoiSet.zip");
						saveAs    (  "Results", dir_result  + File.separator + seN + "_AutoAnalysis.csv");
						roiManager("Deselect");
						roiManager("Delete"); //delete ROI, otherwise, it increases infinitely
					}
					else { 
						print("|         number of ROI = ", RoiManager.size);
						print("| #### ERROR : Number of ROI not = 1 ");
						
						if ( RoiManager.size > 1){ //delete ROI, otherwise, it increases infinitely
							roiManager("Deselect");
							roiManager("Delete");
						}
					}
					print("|"); // make Log file looks better.

				//Clear and close all windows
					// waitForUser("Pause", j+" fish done.");
					run("Clear Results");
					run("Close All");

			}
			else { showMessage(" No Lif inside"); }

		}
	}


	//Update Log
	selectWindow("Log");
	saveAs("Text", dir_bf_lif + File.separator + "Log.txt");


	showMessage(" -- finished --");
	setBatchMode(false);

}// macro
