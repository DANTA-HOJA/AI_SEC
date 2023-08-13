import os
import sys
import re
from pathlib import Path

import jpype               # Import module
import jpype.imports       # Enable Java imports
from jpype.types import *  # Pull in types
import scyjava as sj       # scyjava : Supercharged Java access from Python, see https://github.com/scijava/scyjava
import imagej              # pyimagej

from ...shared.logger import init_logger
from ...shared.pathnavigator import PathNavigator


class ZFIJ():
    
    def __init__(self) -> None:
        """ Initialize `ImageJ` and the necessary components used in `ZebraFish_AP_POS`
        """
        self._logger = init_logger(r"Zebrafish IJ")
        self._display_kwargs = {
            "display_on_CLI": True,
            "logger": self._logger
        }
        self._path_navigator = PathNavigator()
        
        """ Store `Current Working Directory(cwd)` and `sys.stdout` """
        orig_wd = os.getcwd()
        orig_stdout = sys.stdout # store original 'sys.stdout'
        
        self._init_imagej()
        self._init_other_components()
        self._redirect_fn()

        """ Recover `Working Directory` and `sys.stdout` """
        os.chdir(orig_wd)
        sys.stdout = orig_stdout
    
    
    def _init_imagej(self):
        """
        """
        """ JVM Configurations """
        # NOTE: The ImageJ2 gateway is initialized through a Java Virtual Machine (JVM). 
        #       If you want to configure the JVM, it must be done before initializing an ImageJ2 gateway.
        # sj.config.add_option('-Xmx10g') # adjust memory available to Java
        sj.config.endpoints.append('ome:formats-gpl:6.11.1')
        
        """ Get path of Fiji(ImageJ) """
        self.fiji_local = self._path_navigator.dbpp.get_fiji_local_dir(**self._display_kwargs)
        
        """ Different methods to start ImageJ """
        # ij = imagej.init(fiji_local) # Same as "ij = imagej.init(fiji_local, mode='headless')", PyImageJ’s default mode is headless
        # ij = imagej.init(fiji_local, mode='gui') # GUI mode (會卡在這一行 -> blocking), for more explainations : https://pyimagej.readthedocs.io/en/latest/Initialization.html#gui-mode
        self.ij = imagej.init(self.fiji_local, mode='interactive') # Interactive mode (可以繼續向下執行 -> non-blocking), for more explainations : https://pyimagej.readthedocs.io/en/latest/Initialization.html#interactive-mode
        self.ij.ui().showUI() # display the Fiji GUI
        
        """ Print Fiji version """
        print(self.ij.getApp().getInfo(True)) # ImageJ2 2.9.0/1.54b
    
    
    def _init_other_components(self):
        """
        """
        # Bio-Format Reader
        loci = jpype.JPackage("loci")
        loci.common.DebugTools.setRootLevel("ERROR")
        self.imageReader = loci.formats.ImageReader()
        
        # [IMPORTANT] some ImageJ plugins need to be new before use
        from ij.plugin.frame import RoiManager
        from ij.plugin import ImageCalculator
        from ij.plugin import ChannelSplitter
        from ij.plugin import ZProjector
        from ij.plugin import RGBStackMerge
        from ij.plugin import RGBStackConverter

        self.roiManager = RoiManager()
        self.imageCalculator = ImageCalculator()
        self.channelSplitter = ChannelSplitter()
        self.zProjector = ZProjector()
        self.rgbStackMerge = RGBStackMerge()
        self.rgbStackConverter = RGBStackConverter()
    
    
    def _redirect_fn(self):
        """
        """
        self.run = self.ij.IJ.run
    
    
    def reset_all_window(self):
        """
        """
        if int(self.roiManager.getCount()) > 0:
            self.roiManager.runCommand("Deselect")
            self.roiManager.runCommand("Delete")
            self.run("Clear Results", "")
        self.run("Close All", "")