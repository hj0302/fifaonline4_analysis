# -*- coding: utf-8 -*- 

import math
import json
import pandas as pd


def calc_dist(shoot_x, shoot_y, compare_x=1, compare_y=0.5):
    '''
    DESC
        골대와 거리 계산
    '''
    dist = math.sqrt(math.pow(shoot_x - compare_x, 2) + math.pow(shoot_y - compare_y, 2))
    return dist


def calc_shootangel(x, y):
    '''
    DESC
        골대와 각도 계산
    '''
    def calc_rad(arr):
        rad = math.atan2(arr[3]-arr[1], arr[2]-arr[0])
        return rad
    
    def trans_rad_to_deg(rad):
        deg = (rad*180)/math.pi
        return round(deg, 2)
    
    rad = calc_rad([0.5, 1, y, x])
    deg = trans_rad_to_deg(rad)
    
    if y <= 0.5 :
        return abs(deg) - 90
    else :
        return 90 - abs(deg)


def dict_to_expend_dataframe(df, col):
    '''
    DESC

    '''
    # Dictionary로 보이는 값들은 사실 String이다. Dictionary로 읽어들인다.
    tmp_df = pd.DataFrame(df, columns=[col])
    
    # 첫번째 행의 Key를 Column 명으로
    cols = tmp_df[col][0].keys() 
    
    # Apply 함수로 Value 값을 List로 추출 -> rows는 List들의 Array
    rows = tmp_df[col].apply(lambda x:list(x.values())) 
    
    # 추출한 행과 칼럼명으로 DataFrame 구성
    tmp_df = pd.DataFrame(rows.values.tolist(), columns=cols)
    df = pd.concat([df, tmp_df], axis=1)
    df.drop([col], axis=1, inplace=True)
    
    return df


def load_txt(files):
    '''
    DESC

    '''
    json_data = []
    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()
            json_data.extend([ json.loads(i) for i in lines])
    return json_data