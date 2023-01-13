import logging
import logging.handlers
import time
import random


def get_logger(file_nm):
    '''
    DESC
        스크래핑 차단당하는 것을 방지하기 위해 time sleep 실행해주는 함수
    
    Args

    Return
    '''

    # create logger
    logger = logging.getLogger('logger')
    # set logger level
    logger.setLevel(logging.DEBUG)

    # create consele hendler and set level to debug
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(file_nm)

    # create formmater 
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s >> file::%(filename)s - line::%(lineno)s'
    )
    # add formatter to handler
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # add new_hanler to logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def execute_sleep(start_num, end_num):
    '''
    DESC
        스크래핑 차단당하는 것을 방지하기 위해 time sleep 실행해주는 함수
    
    Args

    Return
    '''
    random_num = random.uniform(start_num, end_num)
    
    return time.sleep(random_num)


