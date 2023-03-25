import os
import time



class Timer():
    
    def __init__(self) -> None:
        pass
    
    def start(self):     
        self.start_time = time.time()
    
    def stop(self): 
        self.stop_time = time.time()
        
    def calculate_consume_time(self):
        self.consume_time = self.stop_time - self.start_time
        
    def save_consume_time(self, dir_path, desc):
        consume_time_str = f"{{ {desc} }}_{{ {self.consume_time:.4f} sec }}"
        write_path = os.path.normpath(f"{dir_path}/{consume_time_str}")
        # write file
        with open(write_path, mode="w") as f_writer: 
            f_writer.write(f"{self.consume_time}")