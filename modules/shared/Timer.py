import os
import time



class Timer():
    
    def __init__(self) -> None:
        self.__reset()
    
    def __reset(self):
        self.__start_time = None
        self.__stop_time = None
    
    def start(self):
        self.__reset()
        self.__start_time = time.time()
    
    def stop(self):
        if self.__start_time is None: raise AttributeError("Timer has not been started yet")
        self.__stop_time = time.time()
        
    def calculate_consume_time(self):
        self.consume_time = self.__stop_time - self.__start_time
        
    def save_consume_time(self, dir_path, desc):
        consume_time_str = f"{{ {desc} }}_{{ {self.consume_time:.4f} sec }}"
        write_path = os.path.normpath(f"{dir_path}/{consume_time_str}")
        # write file
        with open(write_path, mode="w") as f_writer: 
            f_writer.write(f"{self.consume_time}")