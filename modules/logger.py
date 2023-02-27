import logging


def init_logger(logger_name:str) -> logging.Logger:
    
    log: logging.Logger = logging.getLogger(logger_name)
    log.setLevel(logging.DEBUG)
    
    stream_handler: logging.StreamHandler = logging.StreamHandler()
    
    formatter: logging.Formatter = logging.Formatter('| %(asctime)s | %(name)s | %(levelname)s | %(message)s')
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    
    return log