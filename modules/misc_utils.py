import os
import time



class Timer():
    
    def __init__(self) -> None:
        pass
    
    def start(self):     
        self.__start_time = time.time()
    
    def stop(self): 
        self.__stop_time = time.time()
        
    def calculate_consume_time(self):
        self.__consume_time = self.__start_time - self.__stop_time
        
    def save_consume_time(self, dir_path, desc):
        consume_time_str = f"{{ {desc} }}_{{ {self.__consume_time:.4f} sec }}"
        write_path = os.path.normpath(f"{dir_path}/{consume_time_str}")
        # write file
        with open(write_path, mode="w") as f_writer: 
            f_writer.write(f"{self.__consume_time}")