import json
import os
import re
import sys

import imagej  # pyimagej
import jpype  # Import module
import jpype.imports  # Enable Java imports
import scyjava as sj  # scyjava : Supercharged Java access from Python built on `JPype` and `jgo`., see https://github.com/scijava/scyjava
from jpype.types import *  # Pull in types

from ...shared.baseobject import BaseObject
# -----------------------------------------------------------------------------/


class ZFIJ(BaseObject):

    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__(display_on_CLI)
        self._cli_out._set_logger("Zebrafish IJ")
        
        # ---------------------------------------------------------------------
        # """ attributes """
        # TODO
        # ---------------------------------------------------------------------
        # """ actions """
        
        self._cli_out.divide()
        
        # store 'Working Directory' and `sys.stdout`
        orig_wd = os.getcwd()
        # orig_stdout = sys.stdout
        
        self._init_imagej()
        self._init_other_components()
        self._redirect_fn()
        
        # recover 'Working Directory' and `sys.stdout`
        os.chdir(orig_wd)
        # sys.stdout = orig_stdout
        # ---------------------------------------------------------------------/


    def _init_imagej(self):
        """
        """
        """ JVM Configurations """
        # NOTE: The ImageJ2 gateway is initialized through a Java Virtual Machine (JVM). 
        #       If you want to configure the JVM, it must be done before initializing an ImageJ2 gateway.
        #       originally, we need to use below `jpype` function ( https://github.com/jpype-project/jpype/issues/245 )
        #
        #       >>> jpype.startJVM(jpype.getDefaultJVMPath(), "-Xms64m", "-Xmx64m")
        #
        #       but `pyimagej` is based on `scyjava` so we can use below function
        sj.config.add_option('-Xmx10g') # adjust memory available to Java
        # sj.config.endpoints.append('ome:formats-gpl:6.11.1')
        
        """ Get path of Fiji(ImageJ) """
        self.fiji_local = self._path_navigator.dbpp.get_fiji_local_dir(self._cli_out)
        
        """ Redirect `JAVA_HOME` ('JAVA_HOME' might already be set in the OS environment)"""
        found_list = list(self.fiji_local.glob("java/**/jre"))
        if len(found_list) != 1:
            raise ValueError(f"Multiple JRE are found, "
                             f"{json.dumps([str(path) for path in found_list], indent=2)}")
        else:
            os.environ["JAVA_HOME"] = str(found_list[0])
        
        """ Different methods to start ImageJ """
        # ij = imagej.init(fiji_local) # Same as "ij = imagej.init(fiji_local, mode='headless')", PyImageJ’s default mode is headless
        # ij = imagej.init(fiji_local, mode='gui') # GUI mode (會卡在這一行 -> blocking), for more explainations : https://pyimagej.readthedocs.io/en/latest/Initialization.html#gui-mode
        self.ij = imagej.init(self.fiji_local, mode='interactive') # Interactive mode (可以繼續向下執行 -> non-blocking), for more explainations : https://pyimagej.readthedocs.io/en/latest/Initialization.html#interactive-mode
        self.ij.ui().showUI() # display the Fiji GUI
        
        """ Print Fiji version """
        self._cli_out.write(self.ij.getApp().getInfo(True)) # ImageJ2 2.9.0/1.54b
        # ---------------------------------------------------------------------/


    def _init_other_components(self):
        """ Create/new the plugin instances
        """
        """ Set `loci`( Bio-Formats ) Warning Level """
        loci = jpype.JPackage("loci")
        loci.common.DebugTools.setRootLevel("ERROR")
        
        """ Bio-Formats Reader """
        self.imageReader = jpype.JClass("loci.formats.ImageReader")()
        
        """ GUI object """
        self.roiManager = jpype.JClass("ij.plugin.frame.RoiManager")()
        
        """ Image operator """
        self.imageCalculator = jpype.JClass("ij.plugin.ImageCalculator")()
        self.channelSplitter = jpype.JClass("ij.plugin.ChannelSplitter")()
        self.zProjector = jpype.JClass("ij.plugin.ZProjector")()
        self.rgbStackMerge = jpype.JClass("ij.plugin.RGBStackMerge")()
        self.rgbStackConverter = jpype.JClass("ij.plugin.RGBStackConverter")()
        # ---------------------------------------------------------------------/


    def _redirect_fn(self):
        """
        """
        self.run = self.ij.IJ.run
        self.save_as_tiff = self.ij.IJ.saveAsTiff
        self.save_as = self.ij.IJ.saveAs
        # ---------------------------------------------------------------------/


    def reset_all_window(self):
        """
        """
        if int(self.roiManager.getCount()) > 0:
            self.roiManager.reset()
        self.run("Clear Results", "")
        self.run("Close All", "")
        # ---------------------------------------------------------------------/