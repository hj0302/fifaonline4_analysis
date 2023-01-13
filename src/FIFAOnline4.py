# -*- coding: utf-8 -*- 

import os
import pandas as pd
import numpy as np
import datetime
import math
from glob import glob
import joblib
import requests
import json
from bs4 import BeautifulSoup
import joblib
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from utils import get_logger, execute_sleep
from fifa_utils import *

import warnings
warnings.filterwarnings('ignore')

DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

class FIFAOnline4API():
    '''
    피파온라인4 데이터 수집기
    '''
    def __init__(self):
        self.headers = {'Authorization': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJYLUFwcC1SYXRlLUxpbWl0IjoiNTAwOjEwIiwiYWNjb3VudF9pZCI6IjE0NjAwODc0MzgiLCJhdXRoX2lkIjoiMiIsImV4cCI6MTY4ODI4MTY0NSwiaWF0IjoxNjcyNzI5NjQ1LCJuYmYiOjE2NzI3Mjk2NDUsInNlcnZpY2VfaWQiOiI0MzAwMTE0ODEiLCJ0b2tlbl90eXBlIjoiQWNjZXNzVG9rZW4ifQ.PfxPGSlJCuCS2hRTBZ_HJ3wAlrNF-bXz8CRh9lHYVv0'} 
        self.logger = get_logger(file_nm=f'{DIR_PATH}/rsc/log/{datetime.datetime.now().strftime("%Y%m%d")}_log.txt')


    def execute_api_response(self, url):
        '''
        DESC
            API Call 실행함수
        
        Args
            url     string      수집하고자하는 url 주소        
        
        Response
            Data    dict        각 항목별 api response
        '''
        response = requests.get(url=url, headers=self.headers)
        
        if response.status_code == 200:
            res_json = json.loads(response.text)
            #self.logger.info(f'{url} API SUCCESS')
            return res_json
        else :
            self.logger.info(f'{url} API FAIL')
            return {}

    
    def execute_bs4_response(self, url):
        '''
        DESC
            BeautifulSoup Crawl Call 실행 함수
        Args
                
        Return
        '''
        response = requests.get(url=url, headers=self.headers)
        
        if response.status_code == 200:
            html = response.text
            bs = BeautifulSoup(html, "html.parser")
            for i in bs.select('#middle > div > div > div:nth-child(2) > div.content.data_detail > div > div.content_bottom'):
                data= i.get_text()
            #self.logger.info(f'{url} API SUCCESS')
            return data
        else :
            self.logger.info(f'{url} API FAIL')
            return ''


    def get_user_info_by_nickname(self, nickname):
        '''
        DESC
            유저의 닉네임으로 유저 고유 식별자를 조회합니다.

        Args
            nickname     String      유저닉네임

        Return
            Response Description
            데이터      타입           설명
            accessId	String	    유저 고유 식별자 
            nickname	String	    유저 닉네임 
            level	    Integer 	유저 레벨 
        '''
        url = f'https://api.nexon.co.kr/fifaonline4/v1.0/users?nickname={nickname}'
        
        return self.execute_api_response(url)


    def get_user_info_by_accessid(self, accessid):
        '''
        DESC
            유저 고유 식별자{accessid}로 유저의 정보를 조회합니다.

        Args
            accessId	String	유저 고유 식별자 

        Response Description
            데이터	타입	설명
            accessId	String	유저 고유 식별자 
            nickname	String	유저 닉네임 
            level	Integer	유저 레벨 
        '''
        url = f'https://api.nexon.co.kr/fifaonline4/v1.0/users/{accessid}'
                
        return self.execute_api_response(url)


    def get_user_maxdivision_by_accessid(self, accessid):
        '''
        DESC
            유저 고유 식별자{accessid}로 유저별 역대 최고 등급과 달성일자를 조회합니다.

        Args
            accessId	String	유저 고유 식별자 

        Response Description
            데이터	유형	설명
            matchType	Integer	매치 종류 (/metadata/matchtype API 참고) 
            division	Integer	등급 식별자
                (공식경기 : /metadata/division API 참고
                볼타모드 : /metadata/division_volta API 참고) 
            achievementDate	String	최고등급 달성일자 (ex. 2019-05-13T18:03:10) 
        '''
        url = f'https://api.nexon.co.kr/fifaonline4/v1.0/users/{accessid}/maxdivision'
        
        return self.execute_api_response(url)

            
    def get_user_match_by_accessid(self, accessid, matchtype, offset=0, limit=100):
        '''
        DESC
            유저 고유식별자{accessid}와 매치 종류{matchtype}로 유저의 매치 종류별 기록을 조회합니다.
        
        Args
            accessId	String	유저 고유 식별자 
            matchtype   integer 매치 종류
                (/metadata/matchtype API 참고)
            offset      integer 리스트에서 가져올 시작 위치
            limit       integer 리스트에서 가져올 갯수(최대 100개)
        
        Response Description
        데이터	
            유저가 플레이한 매치의 고유 식별자 목록
        '''
        url = f'https://api.nexon.co.kr/fifaonline4/v1.0/users/{accessid}/matches?matchtype={matchtype}&offset={offset}&limit={limit}'
        
        return self.execute_api_response(url)

            
    def get_user_market_by_accessid(self, accessid, tradetype, offset=0, limit=100):
        '''
        DESC
            유저 고유식별자{accessid}와 매치 종류{matchtype}로 유저의 매치 종류별 기록을 조회합니다.
        
        Args
            accessId	String	유저 고유 식별자 
            tradetype   string  거래 종류 (구입 : buy, 판매 : sell)
            offset      integer 리스트에서 가져올 시작 위치
            limit       integer 리스트에서 가져올 갯수(최대 100개)
        
        Response Description
            데이터	유형	설명
            tradeDate	String	거래일자 (ex. 2019-05-13T18:03:10) 
                구매(buy)일	경우	구매 등록 시점 
                판매(sell)일	경우	판매 완료 시점 
            saleSn	String	거래 고유 식별자 
            spid	Integer	선수 고유 식별자 
                (/metadata/spid API 참고) 
            grade	Integer	거래 선수 강화 등급 
            value	Integer	거래 선수 가치(BP) 
        '''
        url = f'https://api.nexon.co.kr/fifaonline4/v1.0/users/{accessid}/markets?tradetype={tradetype}&offset={offset}&limit={limit}'
        
        return self.execute_api_response(url)

            
    def get_match_list_by_matchtype(self, matchtype, offset=0, limit=100, orderby='desc'):
        '''
        DESC
            매치 종류(matchtype)로 모든 매치의 종류별 기록을 조회합니다.
        
        Args
            matchtype	integer	    매치 종류
            offset      integer     리스트에서 가져올 시작 위치
            limit       integer     리스트에서 가져올 갯수(최대 100개)
            orderby     string      매치 기록의 정렬 순서 
                - asc : 가장 오래된 매치부터 매치목록 반환
                - desc : 가장 최근 플레이한 매치부터 매치목록 반환
        
        Response 
            모든 매치의 고유 식별자 목록
        '''
        url = f'https://api.nexon.co.kr/fifaonline4/v1.0/matches?matchtype={matchtype}&offset={offset}&limit={limit}&orderby={orderby}'

        return self.execute_api_response(url)


    def get_match_detail_by_matchid(self, matchid):
        '''
        DESC
            매치 고유 식별자{matchid}로 매치의 상세 정보를 조회합니다.
            ※ 매치 통계가 생성되기 전에 상대방이 매치를 종료할 경우, 상대방의 매치 정보가 보이지 않을 수도 있습니다.
        
        Args
            matchid	    string	    매치 고유 식별자

        Response
            MatchDTO
            ㄴ matchId	        String	    매치 고유 식별자 
            ㄴ matchDate	    String	    매치 일자 (ex. 2019-05-13T18:03:10) 
            ㄴ matchType	    Integer	    매치 종류 
            ㄴ MatchInfoDTO     Array       매치 참여 플레이어 별 매치 내용 상세 리스트
                ㄴ accessod     String      유저 고유 식별자 
                ㄴ nickname     String      유저 닉네임
                ㄴ MatchDetailDTO           매치 결과 상세 정보
                ㄴ ShootDTO                 슈팅 정보
                ㄴ ShootDetailDTO           슈팅별 상세 정보 리스트
                ㄴ PassDTO                  패스 정보
                ㄴ DefenceDTO               수비 정보
                ㄴ PlayerDTO                경기 사용 선수 정보
        '''
    
        url = f'https://api.nexon.co.kr/fifaonline4/v1.0/matches/{matchid}'
            
        return self.execute_api_response(url)

    
    def get_playerlist_by_ranker(self, matchtype, players):
        '''
        DESC
            TOP 10,000 랭커 유저가 사용한 선수의 20경기 평균 스탯을 조회합니다.
            선수의 고유 식별자와 포지션의 목록으로 랭커 유저가 사용한 선수의 평균 스탯 기록을 조회할 수 있습니다.

        Args
            matchtype	integer	매치 종류 (/metadata/matchtype API 참고)
            players     string  조회하고자 하는 선수 목록 (Json Object Array)
        
        Response Description
            RankerPlayerDTO
            RankerPlayerStatDTO
        '''

        url = f'https://api.nexon.co.kr/fifaonline4/v1.0/rankers/status?matchtype={matchtype}&players={players}'

        return self.execute_api_response(url)


    def get_metadata_matchtype(self):
        '''
        DESC
            매치 종류(matchtype) 메타데이터를 조회합니다.
        
        Response Description
            matchtype.json
        '''
        url = f'https://static.api.nexon.co.kr/fifaonline4/latest/matchtype.json'
            
        return self.execute_api_response(url)

            
    def get_metadata_spid(self):
        '''
        DESC
            선수 고유 식별자(spid) 메타데이터를 조회합니다
            * 선수 고유 식별자는 시즌아이디 (seasonid) 3자리 + 선수아이디 (pid) 6자리로 구성
        
        Response Description
            spid.json
        '''
        url = f'https://static.api.nexon.co.kr/fifaonline4/latest/spid.json'
            
        return self.execute_api_response(url)

        
    def get_metadata_seasonid(self):
        '''
        DESC
            시즌아이디(seasonId) 메타데이터를 조회합니다.
            * 시즌아이디는 선수가 속한 클래스를 나타냅니다.
        
        Response Description
            seasonid.json
        '''
        url = f'https://static.api.nexon.co.kr/fifaonline4/latest/seasonid.json'
            
        return self.execute_api_response(url)   


    def get_metadata_seasonid(self):
        '''
        DESC
            시즌아이디(seasonId) 메타데이터를 조회합니다.
            * 시즌아이디는 선수가 속한 클래스를 나타냅니다.
        
        Response Description
            seasonid.json
        '''
        url = f'https://static.api.nexon.co.kr/fifaonline4/latest/seasonid.json'
            
        return self.execute_api_response(url)   


    def get_metadata_spposition(self):
        '''
        DESC
            선수 포지션(spposition) 메타데이터를 조회합니다.
        
        Response Description
            spposition.json
        '''
        url = f'https://static.api.nexon.co.kr/fifaonline4/latest/spposition.json'
            
        return self.execute_api_response(url)   


    def get_metadata_division(self):
        '''
        DESC
            등급 식별자(division) 메타데이터를 조회합니다.
        
        Response Description
            division.json
        '''
        url = f'https://static.api.nexon.co.kr/fifaonline4/latest/division.json'
            
        return self.execute_api_response(url)            


    def get_metadata_division_volta(self):
        '''
        DESC
            볼타 공식경기의 등급 식별자(division) 메타데이터를 조회합니다.

        
        Response Description
            division_volta.json
        '''
        url = f'https://static.api.nexon.co.kr/fifaonline4/latest/division_volta.json'
            
        return self.execute_api_response(url) 

         
    def get_stat_by_spid(self, spid, n1Strong):
        '''
        DESC
            선수 능력치를 조회합니다
        
        Args
            spid        String  선수 고유식별자
            n1Strong    String  강화등급

        Response Description
            
        '''
        url = f'https://fifaonline4.nexon.com/DataCenter/PlayerInfo?spid={spid}&n1Strong={n1Strong}'
        
        return self.execute_bs4_response(url)


    def save_metadata_process(self):
        '''
        DESC
            넥슨 온라인 메타데이터 저장 프로세스 -> 일주일에 한번 업데이트 예정(애주 목요일 12시)
        '''
        # Save SeasonId
        searsonid_list = self.get_metadata_seasonid()
        searsonid_df = pd.DataFrame(searsonid_list)
        searsonid_df.to_pickle(f'{DIR_PATH}/rsc/metadata/fifa_metadata_seasonid.pkl')

        # Save spid
        spid_list = self.get_metadata_spid()
        spid_df = pd.DataFrame(spid_df)
        spid_df.to_pickle(f'{DIR_PATH}/rsc/metadata/fifa_metadata_spid.pkl')

        # Save Spposition
        spposition_list = self.get_metadata_spposition()
        spposition_df = pd.DataFrame(spposition_list)
        spposition_df.to_pickle(f'{DIR_PATH}/rsc/metadata/fifa_metadata_spposition.pkl')

        # Save Player Stat
        #self.save_stat()

    def save_matchdetail_process(self):
        '''
        DESC
            매치 데이터 저장 프로세스 -> 매시 0분에 수집 (50분에 돌려야지 실패가 안되는 느낌이든다..ㅠㅠ)
        '''
        #모든 매치 기록 조회
        for n in range(0, 2000, 100):
            match_list = self.get_match_list_by_matchtype(matchtype=50, offset=n, limit=100, orderby='desc')
            # 매치 기록 상세 조회        
            for matchid in match_list:
                match_detail = self.get_match_detail_by_matchid(matchid=matchid)
                if match_detail == {}:
                    pass
                else :
                    with open(f'{DIR_PATH}/rsc/matchdetail/fifa_match_detail_{datetime.datetime.now().strftime("%Y%m%d")}.txt', 'a', encoding='utf-8') as outfile:
                        outfile.write(json.dumps(match_detail) + '\n')
                execute_sleep(0.2, 0.25)

    def save_stat(self):
        '''
        DESC
            선수 스탯 수집하는 함수
        '''
        spid_df = pd.read_pickle(f'{DIR_PATH}/rsc/metadata/fifa_metadata_spid.pkl')
        spid_df.columns = ['spId', 'spNm']
        for spid in spid_df['spId'].unique().tolist():
            res = self.get_stat_by_spid(spid, 5)
            tmp_json = {}
            tmp_df = pd.DataFrame(np.sum([ i.split('\n\n\n') for i in res.split('\n\n\n\n\n') ]))
            tmp_df = tmp_df.iloc[2:36]
            tmp_df[0] = tmp_df[0].apply(lambda x : x.replace('\n커브', 'n커브').replace('\n스태미너', '스태미너'))
            tmp_df = tmp_df[0].str.split('\n', expand=True)
            tmp_df.columns = ['feature', 'stat']
            tmp_json[spid] = tmp_df.set_index('feature').to_dict()

            with open(f'{DIR_PATH}/rsc/metadata/fifa_player_stat.txt', 'a', encoding='utf-8') as outfile:
                outfile.write(json.dumps(tmp_json) + '\n')


class FIFAOnline4Analysis(FIFAOnline4API):

    def __init__(self):
        self.headers = {'Authorization': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJYLUFwcC1SYXRlLUxpbWl0IjoiNTAwOjEwIiwiYWNjb3VudF9pZCI6IjE0NjAwODc0MzgiLCJhdXRoX2lkIjoiMiIsImV4cCI6MTY4ODI4MTY0NSwiaWF0IjoxNjcyNzI5NjQ1LCJuYmYiOjE2NzI3Mjk2NDUsInNlcnZpY2VfaWQiOiI0MzAwMTE0ODEiLCJ0b2tlbl90eXBlIjoiQWNjZXNzVG9rZW4ifQ.PfxPGSlJCuCS2hRTBZ_HJ3wAlrNF-bXz8CRh9lHYVv0'} 
        self.logger = get_logger(file_nm=f'{DIR_PATH}/rsc/log/analysis_log.txt')


    def preprocessing_df(self, json_data):

        def add_info(df, accessId, nickname, matchId, matchDate, matchType):
            df['accessId'] = ii['accessId']
            df['nickname'] = ii['nickname']

            df['matchId'] = i['matchId']
            df['matchDate'] = i['matchDate']
            df['matchType'] = i['matchType']

            return df

        shootdetail_df = pd.DataFrame()
        matchdetail_df = pd.DataFrame()
        shoot_df = pd.DataFrame()
        pass_df = pd.DataFrame()
        defence_df = pd.DataFrame()
        player_df = pd.DataFrame()

        for i in json_data:
            for ii in i['matchInfo']:    
                # ShootDetail
                tmp_df = pd.DataFrame(ii['shootDetail'])
                tmp_df = add_info(tmp_df, ii['accessId'], ii['nickname'], i['matchId'], i['matchDate'], i['matchType'])
                shootdetail_df = pd.concat([shootdetail_df, tmp_df])

                # matchDetail
                tmp_df = pd.DataFrame.from_records([ii['matchDetail']])
                tmp_df = add_info(tmp_df, ii['accessId'], ii['nickname'], i['matchId'], i['matchDate'], i['matchType'])
                matchdetail_df = pd.concat([matchdetail_df, tmp_df])

                # shoot
                tmp_df = pd.DataFrame.from_records([ii['shoot']])
                tmp_df = add_info(tmp_df, ii['accessId'], ii['nickname'], i['matchId'], i['matchDate'], i['matchType'])
                shoot_df = pd.concat([shoot_df, tmp_df])

                # pass
                tmp_df = pd.DataFrame.from_records([ii['pass']])
                tmp_df = add_info(tmp_df, ii['accessId'], ii['nickname'], i['matchId'], i['matchDate'], i['matchType'])
                pass_df = pd.concat([pass_df, tmp_df])

                # defence
                tmp_df = pd.DataFrame.from_records([ii['defence']])
                tmp_df = add_info(tmp_df, ii['accessId'], ii['nickname'], i['matchId'], i['matchDate'], i['matchType'])
                defence_df = pd.concat([defence_df, tmp_df])

                # player
                tmp_df = pd.DataFrame(ii['player'])

                if tmp_df.shape[0] == 0:
                    pass
                else :
                    tmp_df = dict_to_expend_dataframe(tmp_df, 'status')
                    tmp_df = add_info(tmp_df, ii['accessId'], ii['nickname'], i['matchId'], i['matchDate'], i['matchType'])
                    player_df = pd.concat([player_df, tmp_df])

        # 중복 row 제거
        shootdetail_df = shootdetail_df.drop_duplicates()
        # 좌표가 0인 슈팅 데이터 제외
        shootdetail_df = shootdetail_df[~((shootdetail_df['x'] == 0) & (shootdetail_df['y'] == 0))]
        # 좌표가 1넘는 슈팅 데이터 제외
        shootdetail_df = shootdetail_df[(shootdetail_df['x'] <= 1) & (shootdetail_df['y'] <= 1)]
        # 중앙선 이전(자기)지역에서 골이 들어간 데이터는 제외
        shootdetail_df = shootdetail_df[~((shootdetail_df['x'] <= 0.5) & (shootdetail_df['result'] == 3))]

        self.logger.info(f'rows - {shootdetail_df.shape[0]}')
        self.logger.info(f'matchId - {shootdetail_df.matchId.nunique()}')
        self.logger.info(f'accessId - {shootdetail_df.accessId.nunique()}')
        self.logger.info(f'matchDate - min {shootdetail_df.matchDate.min()}, max : {shootdetail_df.matchDate.max()}')

        # 골대와의 거리 계산
        shootdetail_df['dist'] = shootdetail_df.apply(lambda x : calc_dist(x.x, x.y), axis=1 )
        # 골대와의 각도 계산
        shootdetail_df['angle'] = shootdetail_df.apply(lambda x: calc_shootangel(x.x, x.y), axis=1)

        # column 타입 변경
        for col in ['type', 'result', 'spId', 'spGrade', 'spLevel', 'assistSpId']:
            shootdetail_df[col] = shootdetail_df[col].astype(int)

        shootdetail_df.reset_index(drop=True, inplace=True)

        shootdetail_df.to_pickle(f'{DIR_PATH}/rsc/{datetime.datetime.now().strftime("%Y%m%d_%H")}_preprocessing_data.pkl')

        return shootdetail_df, matchdetail_df, shoot_df, pass_df, defence_df, player_df

    
    def feature_engineering(self, df):
        df['inplay_type'] = df['type'].apply(lambda x : x if x in [8, 9] else 11)
        df['shoot_type'] = df['type'].apply(lambda x : x if x not in [8, 9] else 0)

        df = df.merge(pd.get_dummies(df['inplay_type'], prefix='inplay_type'), left_index=True, right_index=True)
        df = df.merge(pd.get_dummies(df['shoot_type'], prefix='shoot_type'), left_index=True, right_index=True)

        df['assist'] = df['assist'].apply(lambda x : 1 if x == True else 0)
        df['inPenalty'] = df['inPenalty'].apply(lambda x : 1 if x == True else 0)
        df['result'] = df['result'].apply(lambda x : 1 if x == 3 else 0)

        df['angle'] = df['angle'].apply(lambda x : x/90)
        
        col_list = [
            'inplay_type_8', 'inplay_type_9', 'inplay_type_11', 
            'shoot_type_0', 'shoot_type_1', 'shoot_type_2', 'shoot_type_3', 'shoot_type_4', 'shoot_type_6', 'shoot_type_7', 'shoot_type_10'
        ]
    
        for col in col_list :
            if col not in df.columns:
                df[col] = 0
            else :
                pass
        
        return df


    def split_train_test(self, data, feature, target, test_size, random_state):
        X_train, X_test, y_train, y_test = train_test_split(data[feature], data[target], test_size=0.25, random_state=42, stratify=data[target])
        
        self.logger.info(f'X_train Shape {X_train.shape}')
        self.logger.info(f'y_train Shape {y_train.shape}')
        self.logger.info(f'X_test Shape {X_test.shape}')
        self.logger.info(f'y_test Shape {y_test.shape}')

        return X_train, X_test, y_train, y_test 


    def gridsearch_process(self, X_train, X_test, y_train, y_test):
        
        param_grid = {
            'max_iter' : [100, 500, 1000],
            'penalty' : ['l1', 'l2', 'elasticnet'],
            'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'random_state' : [42]
        }

        gridsearch = GridSearchCV(
            LogisticRegression(), 
            return_train_score=True,
            param_grid=param_grid, 
            scoring='roc_auc', 
            cv=10, 
            refit=True
        )
        
        gridsearch.fit(X_train, y_train)

        self.logger.info(f'score : {gridsearch.best_score_}')
        self.logger.info(f'param : {gridsearch.best_params_}')  

        return gridsearch.best_params_


    def save_model(self, X_train, X_test, y_train, y_test, param, save_model_nm):
        param['warm_start'] = True
        model = LogisticRegression(**param)
        
        model.fit(X_train, y_train)

        self.logger.info(f'train score : {model.score(X_train, y_train)}')
        self.logger.info(f'test score  : {model.score(X_test, y_test)}')

        joblib.dump(model, f'{DIR_PATH}/rsc/model/{save_model_nm}') 


    def xg_model_process(self, file_nm):

        feature = [
            'x', 'y', 'assist', 'dist', 'angle', 
            'inplay_type_8', 'inplay_type_9', 'inplay_type_11', 
            'shoot_type_0', 'shoot_type_1', 'shoot_type_2', 'shoot_type_3', 'shoot_type_4', 'shoot_type_6', 'shoot_type_7', 'shoot_type_10'
        ]

        target = ['result']

        # Get files
        files = glob(f'{DIR_PATH}/rsc/matchdetail/*.txt')
        
        # load data
        json_data = load_txt(files)

        # json data to preprecessing
        shootdetail_df, matchdetail_df, shoot_df, pass_df, defence_df, player_df = self.preprocessing_df(json_data)

        # feature prepeocessing => dummy, trans boolean        
        data = self.feature_engineering(shootdetail_df)

        # Data Split Train & Test
        X_train, X_test, y_train, y_test = self.split_train_test(data, feature, target, test_size=0.25, random_state=42)

        # Model Parameter GridSearch
        best_param = self.gridsearch_process(X_train, X_test, y_train, y_test)
        
        # save model
        self.save_model(X_train, X_test, y_train, y_test, best_param, file_nm)


    def update_model(self, model_nm, data, feature, target):
        model = joblib.load(f'{DIR_PATH}/rsc/model/{model_nm}') 
        model.fit(data[feature], data[target])
        joblib.dump(model, f'{DIR_PATH}/rsc/model/{model_nm}') 


    def update_xg_model_process(self, model_nm):

        feature = [
            'x', 'y', 'assist', 'dist', 'angle', 
            'inplay_type_8', 'inplay_type_9', 'inplay_type_11', 
            'shoot_type_0', 'shoot_type_1', 'shoot_type_2', 'shoot_type_3', 'shoot_type_4', 'shoot_type_6', 'shoot_type_7', 'shoot_type_10'
        ]

        target = ['result']

        # Get files
        files = glob(f'{DIR_PATH}/rsc/matchdetail/*.txt')
        
        # load data
        json_data = load_txt(files)

        # json data to preprecessing
        shootdetail_df, matchdetail_df, shoot_df, pass_df, defence_df, player_df = self.preprocessing_df(json_data)

        # feature prepeocessing => dummy, trans boolean        
        data = self.feature_engineering(shootdetail_df)

        # update model        
        self.update_model(model_nm, data, feature, target)


    def player_xgvalue_by_nickname(self, nickname, spposition_merge=False, limit=30, model_nm='lr'):
        '''
            선수별 평균 xg 값
        '''

        feature = [
            'x', 'y', 'assist', 'dist', 'angle', 
            'inplay_type_8', 'inplay_type_9', 'inplay_type_11', 
            'shoot_type_0', 'shoot_type_1', 'shoot_type_2', 'shoot_type_3', 'shoot_type_4', 'shoot_type_6', 'shoot_type_7', 'shoot_type_10'
        ]

        target = ['result']

        spid_df = pd.read_pickle(f'{DIR_PATH}/rsc/metadata/fifa_metadata_spid.pkl')
        spid_df.rename(columns={'id':'spId', 'name':'spNm'}, inplace=True)
        
        seasonid_df = pd.read_pickle(f'{DIR_PATH}/rsc/metadata/fifa_metadata_seasonid.pkl')

        spposition_df = pd.read_pickle(f'{DIR_PATH}/rsc/metadata/fifa_metadata_spposition.pkl')
        spposition_df.rename(columns={'spposition':'spPosition'}, inplace=True)

        # get accessid of user
        userinfo = self.get_user_info_by_nickname(nickname=nickname)

        # get match list
        match_list = self.get_user_match_by_accessid(accessid=userinfo['accessId'], matchtype=50, offset=0, limit=limit)

        # match detail
        json_data = []
        for matchid in np.unique(match_list):
            match_detail = self.get_match_detail_by_matchid(matchid=matchid)

            if match_detail == {}:
                pass
            else :
                json_data.append(match_detail)

        # json data to preprecessing
        shootdetail_df, matchdetail_df, shoot_df, pass_df, defence_df, player_df = self.preprocessing_df(json_data)

        # feature prepeocessing => dummy, trans boolean        
        data = self.feature_engineering(shootdetail_df)

        # merge player info
        data['goalid'] = range(data.shape[0])
        data['seasonId'] = data['spId'].apply(lambda x: int(str(x)[:3]))

        data = pd.merge(data, player_df, on=['matchId', 'matchDate', 'matchType', 'accessId', 'nickname', 'spId'], suffixes=('','_player'))
        data = pd.merge(data, seasonid_df, on =['seasonId'], how='left')
        data = pd.merge(data, spposition_df, on =['spPosition'], how='left')
        data = pd.merge(data, spid_df, on=['spId'], how='left')


        model = joblib.load(f'{DIR_PATH}/rsc/model/{model_nm}_model_v1.pkl') 

        y_pred = model.predict_proba(data[feature])
        data['xgValue'] = y_pred[:,1]

        # 특정 유저의 선수별 xg, 팀별 xg value
        if spposition_merge == True:
            xg_by_spid_df = data.groupby(['accessId', 'spId', 'spNm', 'className', 'spGrade']).agg({'xgValue' : 'sum', 'result' : 'sum', 'goalid' : 'count'}).reset_index()
        else :
            xg_by_spid_df = data.groupby(['accessId', 'spId', 'spNm', 'className', 'desc', 'spGrade']).agg({'xgValue' : 'sum', 'result' : 'sum', 'goalid' : 'count'}).reset_index()

        xg_by_spid_df['xgdiff'] = xg_by_spid_df['result'] - xg_by_spid_df['xgValue']
        xg_by_spid_df['xgdiff_by_cnt'] = xg_by_spid_df['xgdiff'] / xg_by_spid_df['goalid']
        
        my_xg_by_spid_df = xg_by_spid_df[xg_by_spid_df['accessId'] == userinfo['accessId']]

        xg_by_team_df = data.groupby(['matchId', 'matchDate', 'accessId', 'nickname']).agg({'xgValue' : 'sum', 'result' : 'sum', 'goalid' : 'count'}).reset_index()
        xg_by_team_df['xgdiff'] = xg_by_team_df['result'] - xg_by_team_df['xgValue']
        xg_by_team_df['xgdiff_by_cnt'] = xg_by_team_df['xgdiff'] / xg_by_team_df['goalid']


        return my_xg_by_spid_df, xg_by_team_df










if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description='피파온라인4 프로젝트')
    parser.add_argument('--type', '-t', required=True, help='타입을 설정하시오')
    args = parser.parse_args()

    _type = args.type

    _api = FIFAOnline4API()
    _cls = FIFAOnline4Analysis()

    if _type == 'metadata_update':
        _api.save_metadata_process()
    elif _type == 'batch_matchdetail':
        _api.save_matchdetail_process()
    elif _type == 'save_model':
        _cls.xg_model_process('lr_model_20230112.pkl')
    elif _type == 'update_model':
        _cls.update_xg_model_process('lr_model_20230110_1026.pkl')