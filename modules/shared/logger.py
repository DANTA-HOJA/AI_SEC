import logging
from logging import Handler, StreamHandler
from tqdm.auto import tqdm



class TqdmLoggingHandler(Handler):
    
    def __init__(self, level=logging.NOTSET):
        """
        """
        super().__init__(level)
    
    
    def emit(self, record):
        """
        """
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)



def init_logger(logger_name:str) -> logging.Logger:
    """
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # stream_handler = StreamHandler()
    tqdm_handler = TqdmLoggingHandler()
    
    # formatter: logging.Formatter = logging.Formatter('| %(asctime)s | %(filename)s | %(levelname)s | %(message)s')
    formatter: logging.Formatter = logging.Formatter('| %(asctime)s | %(name)s | %(levelname)s | %(message)s')
    tqdm_handler.setFormatter(formatter)
    logger.addHandler(tqdm_handler)
    
    return logger