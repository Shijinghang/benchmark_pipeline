from logging import getLogger, Formatter, StreamHandler, FileHandler, DEBUG, INFO, WARNING, ERROR, CRITICAL

def get_logger(name, log_file):
    logger = getLogger(name)
    logger.setLevel(DEBUG)

    # create formatter
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create console handler and set level to debug
    ch = StreamHandler()
    ch.setLevel(DEBUG)
    ch.setFormatter(formatter)

    # create file handler and set level to debug
    fh = FileHandler(log_file)
    fh.setLevel(INFO)
    fh.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger
    
    

