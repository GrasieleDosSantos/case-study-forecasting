import pandas as pd
import numpy as np
import os
import time

def get_path():
    path="output.csv"
    return path

def get_timezone():
    os.environ['TZ'] = 'America/Sao_Paulo'
    time.tzset()
    return True
    
def get_threshold():
    return int(202010)

def get_target():
    return 'QT_VENDA'

def get_time_variable():
    return 'COD_CICLO'

def columns_to_drop():
    return {'COD_MATERIAL'}