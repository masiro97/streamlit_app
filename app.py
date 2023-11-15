#
# 주식가격 시계열 예측 앱.
#

# 먼저 커맨드라인에서 다음과 같이 라이브러리 설치 필요.
# pip install streamlit
# pip install streamlit-lottie
# pip install finance-datareader
# pip install mplfinance
# pip install bs4
# pip install statsmodels
# pip install scikit-learn

# 필요한 라이브러리를 불러온다.
import streamlit as st
import FinanceDataReader as fdr
import mplfinance as mpf
import json
import pandas as pd
import warnings
from streamlit_lottie import st_lottie
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing   # Holt Winters 지수평활화 모형.
from sklearn.linear_model import LinearRegression              # 선형회귀 모형.
warnings.filterwarnings("ignore")                              # 성가신 warning을 꺼준다.

# 지수평활화(ES) 예측을 추가해 주는 함수.
def ESPrediction(data, pred_ndays):
    n = len(data)
    # 그래프 출력에 유리한 형태로 데이터프레임 변환.
    ser = data['Close'].reset_index(drop=True)      # Pandas의 Series 객체.  
    model = ExponentialSmoothing(ser, trend='add', seasonal='add', seasonal_periods=10).fit()    # 학습 완료.
    past = ser.iloc[-10:] # 60, 120
    predicted = model.predict(start= n, end=n+pred_ndays-1) # 모형 예측.
    predicted.rename('Close', inplace=True)                 # Name을 'Close'와 같이 맞추어 준다.
    joined = pd.concat([past, predicted],axis=0)                                      

    return joined[10:]      

# 자기회귀(AR) 예측을 추가해 주는 함수.
def ARPrediction(data, pred_ndays):
    n = len(data)
# 그래프 출력에 유리한 형태로 데이터프레임 변환.
    df = data[ ['Close'] ].reset_index(drop=True)      # Pandas의 DataFrame 객체.     
    df['m1'] = df['Close'].shift(1)                    # t-1 값.
    df['m2'] = df['Close'].shift(2)                    # t-2 값.
    df['m3'] = df['Close'].shift(3)                    # t-3 값.
    df['m4'] = df['Close'].shift(4)                    # t-4 값.
    df['m5'] = df['Close'].shift(5)                    # t-5 값. 
    df['m6'] = df['Close'].shift(5)
    df['m7'] = df['Close'].shift(5)
    df['m8'] = df['Close'].shift(5)
    df['m9'] = df['Close'].shift(5)
    df['m10'] = df['Close'].shift(5)    
    df = df.iloc[10:]
# 선형회귀 기반  AR(5)모형 학습.
    model = LinearRegression()
    model.fit(df[['m1','m2','m3','m4','m5', 'm6', 'm7', 'm8', 'm9', 'm10']], df['Close'])
# 선형회귀 기반  AR(5)모형 예측.
    ser = df['Close'][-10:]                             # 데이터 최신 5개 값.
    for step in range(pred_ndays):                     # 미래 예측.
        past = pd.DataFrame(data={ f'm{i}': [ser.iloc[-i]] for i in range(1,11)} ) # 최신 5개값으로 데이터프레임을 만든다.
        predicted = model.predict(past)[0]                                        # 예측 결과는 원소가 1개인 데이터프레임.
        ser = pd.concat( [ser, pd.Series({n + step:predicted}) ])
    
    return ser[10:]


# 지수평활화(ES) 예측을 추가해 주는 함수.
def addESPrediction(data, ax, pred_ndays):
    n = len(data)
    # 그래프 출력에 유리한 형태로 데이터프레임 변환.
    ser = data['Close'].reset_index(drop=True)      # Pandas의 Series 객체.  
    model = ExponentialSmoothing(ser, trend='add', seasonal='add', seasonal_periods=10).fit()    # 학습 완료.
    past = ser.iloc[-10:] # 60, 120
    predicted = model.predict(start= n, end=n+pred_ndays-1) # 모형 예측.
    predicted.rename('Close', inplace=True)                 # Name을 'Close'와 같이 맞추어 준다.
    joined = pd.concat([past, predicted],axis=0)                                      
    ax.plot(joined[9:], color = 'aqua', linestyle ='--', linewidth=1.5, label = 'ES')
    ax.legend(loc='best')          

# 자기회귀(AR) 예측을 추가해 주는 함수.
def addARPrediction(data, ax, pred_ndays):
    n = len(data)
# 그래프 출력에 유리한 형태로 데이터프레임 변환.
    df = data[ ['Close'] ].reset_index(drop=True)      # Pandas의 DataFrame 객체.     
    df['m1'] = df['Close'].shift(1)                    # t-1 값.
    df['m2'] = df['Close'].shift(2)                    # t-2 값.
    df['m3'] = df['Close'].shift(3)                    # t-3 값.
    df['m4'] = df['Close'].shift(4)                    # t-4 값.
    df['m5'] = df['Close'].shift(5)                    # t-5 값. 
    df['m6'] = df['Close'].shift(5)
    df['m7'] = df['Close'].shift(5)
    df['m8'] = df['Close'].shift(5)
    df['m9'] = df['Close'].shift(5)
    df['m10'] = df['Close'].shift(5)    
    df = df.iloc[10:]
# 선형회귀 기반  AR(5)모형 학습.
    model = LinearRegression()
    model.fit(df[['m1','m2','m3','m4','m5', 'm6', 'm7', 'm8', 'm9', 'm10']], df['Close'])
# 선형회귀 기반  AR(5)모형 예측.
    ser = df['Close'][-10:]                             # 데이터 최신 5개 값.
    for step in range(pred_ndays):                     # 미래 예측.
        past = pd.DataFrame(data={ f'm{i}': [ser.iloc[-i]] for i in range(1,11)} ) # 최신 5개값으로 데이터프레임을 만든다.
        predicted = model.predict(past)[0]                                        # 예측 결과는 원소가 1개인 데이터프레임.
        ser = pd.concat( [ser, pd.Series({n + step:predicted}) ])
    
    ax.plot(ser[9:], color = 'red', linestyle ='--', linewidth=1.5, label = 'AR(10)')
    ax.legend(loc='best')    


# JSON을 읽어 들이는 함수.
def loadJSON(path):
    f = open(path, 'r')
    res = json.load(f)
    f.close()
    return res

tab1, tab2, tab3 = st.tabs(['차트', '데이터', '계산기'])

with tab1:
    # 로고 Lottie와 타이틀 출력.
    col1, col2 = st.columns([1,2])
    with col1:
        lottie = loadJSON('lottie-stock-candle-loading.json')
        st_lottie(lottie, speed=1, loop=True, width=150, height=150)
    with col2:
        ''
        ''
        st.title('주식 추세 예측')

    # 시장 데이터를 읽어오는 함수들을 정의한다.
    @st.cache_data
    def getData(code, datestart, dateend):
        df = fdr.DataReader(code,datestart, dateend ).drop(columns='Change')  # 불필요한 'Change' 컬럼은 버린다.
        return df

    @st.cache_data
    def getSymbols(market='KOSPI', sort='Marcap'):
        df = fdr.StockListing(market)
        ascending = False if sort == 'Marcap' else True
        df.sort_values(by=[sort], ascending= ascending, inplace=True)
        return df[ ['Code', 'Name', 'Market'] ]

    # 세션 상태를 초기화 한다.
    if 'ndays' not in st.session_state:
        st.session_state['ndays'] = 120

    if 'code_index' not in st.session_state:
        st.session_state['code_index'] = 0

    if 'chart_style' not in st.session_state:
        st.session_state['chart_style'] = 'default'

    if 'volume' not in st.session_state:
        st.session_state['volume'] = True

    if 'pred_ndays' not in st.session_state:
        st.session_state['pred_ndays'] = 5

    # 사이드바에서 폼을 통해서 차트 인자를 설정한다.
    with st.sidebar.form(key="chartsetting", clear_on_submit=True):
        st.header('차트 설정')
        ''
        ''
        symbols = getSymbols()
        choices = zip( symbols.Code , symbols.Name , symbols.Market )
        choices = [ ' : '.join( x ) for x in choices ]  # Code, Name, Market을 한개의 문자열로.
        choice = st.selectbox( label='종목:', options = choices, index=st.session_state['code_index'] )
        code_index = choices.index(choice)
        code = choice.split()[0]                        # 실제 code 부분만 떼어 가져온다.
        ''
        ''
        ndays = st.slider(
            label='데이터 기간 (days):', 
            min_value= 20,
            max_value= 365, 
            value=st.session_state['ndays'],
            step = 1)
        ''
        ''
        chart_styles = ['default', 'binance', 'blueskies', 'brasil', 'charles', 'checkers', 'classic', 'yahoo','mike', 'nightclouds', 'sas', 'starsandstripes']
        chart_style = st.selectbox(label='차트 스타일:',options=chart_styles,index = chart_styles.index(st.session_state['chart_style']))
        ''
        ''
        volume = st.checkbox('거래량', value=st.session_state['volume'])

        '---'

        pred_ndays = st.slider(
            label='예측 기간 (days):', 
            min_value= 1,
            max_value= 10, 
            value=st.session_state['pred_ndays'],
            step = 1)
        
        '---'
        
        if st.form_submit_button(label="OK"):
            st.session_state['ndays'] = ndays
            st.session_state['code_index'] = code_index
            st.session_state['chart_style'] = chart_style
            st.session_state['volume'] = volume
            st.session_state['pred_ndays'] = pred_ndays
            st.experimental_rerun()
        

    # 캔들 차트 위에 예측을 그주는 함수.
    def plotChart(data, pred_ndays):
        chart_style = st.session_state['chart_style']
        marketcolors = mpf.make_marketcolors(up='red', down='blue')
        mpf_style = mpf.make_mpf_style(base_mpf_style= chart_style, marketcolors=marketcolors)

        fig, ax = mpf.plot(
            data,
            volume=st.session_state['volume'],
            type='candle',
            style=mpf_style,
            figsize=(10,7),
            fontscale=1.1,
            returnfig=True                  # Figure 객체 반환.
        )
        addESPrediction(data, ax[0], pred_ndays)        # 지수평활화 예측.
        addARPrediction(data, ax[0], pred_ndays)        # AR 예측.
        st.pyplot(fig)

    # 데이터를 불러오고 최종적으로 차트를 출력해 준다.
    # 주의: datetime.today()에는 항상 변하는 "시:분:초"가 들에있어서 cache가 작동하지 않는다.
    #       "시:분:초"를 떼어 버리고 날짜만 남도록 date()를 호출하는 것이 중요하다!

    date_start = (datetime.today()-timedelta(days=st.session_state['ndays'])).date()
    df = getData(code, date_start, datetime.today().date())     
    chart_title = choices[st.session_state['code_index'] ]
    st.markdown(f'<h3 style="text-align: center; color: red;">{chart_title}</h3>', unsafe_allow_html=True)
    plotChart(df, st.session_state['pred_ndays'])

with tab2:
    date_start = (datetime.today()-timedelta(days=st.session_state['ndays'])).date()
    df = getData(code, date_start, datetime.today().date())
    df['HighestPrice'] = df['High'].rolling(window=10).max().shift(1)
    df['LowestPrice'] = df['Low'].rolling(window=10).min().shift(1)
    df['CandleSize'] = abs(df['Open'] - df['Close'])
    df['MaxCandleSize'] = df['CandleSize'].rolling(window=10).max().shift(1)
    df['BuyCondition1'] = df['Close'] > df['HighestPrice']
    df['BuyCondition2'] = df['CandleSize'] > df['MaxCandleSize']
    df['SellCondition1'] = df['Close'] < df['LowestPrice']
    df['Signal'] =  df['BuyCondition1'] & df['BuyCondition2']
    df['Volume_Signal'] = df['Volume'].iloc[-2] * 2 < df['Volume'].iloc[-1]
    df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'HighestPrice', 'LowestPrice' ,'BuyCondition1' ,'SellCondition1' , 'Signal' ,'Volume_Signal']]
    df_sub = df[['HighestPrice', 'LowestPrice' ,'BuyCondition1' ,'SellCondition1' , 'Signal', 'Volume_Signal']]
    st.subheader('신호 데이터')

    st.dataframe(df_sub[(df['BuyCondition1'] == True) | (df['Signal'] == True) | (df['SellCondition1'] == True)])

    st.subheader('ES 예측치')

    st.table(ESPrediction(df, pred_ndays))

    st.success(f'ES 최대 예측치 : {ESPrediction(df, pred_ndays).max():.0f}')

    st.error(f'ES 최저 예측치 : {ESPrediction(df, pred_ndays).min():.0f}')

    st.subheader('AR 예측치')

    st.table(ARPrediction(df, pred_ndays))

    st.success(f'AR 최대 예측치 : {ARPrediction(df, pred_ndays).max():.0f}')

    st.error(f'AR 최저 예측치 : {ARPrediction(df, pred_ndays).min():.0f}')


with tab3:
    st.subheader('계산기')

    buy_price = st.text_input('매수가')

    loss_cut_price = st.text_input('손절가')

    loss_cut_amount = st.text_input('최대 손절 금액')

    cal = st.button('계산하기')
    
    if cal:
        result = float(loss_cut_amount) // (float(buy_price) - float(loss_cut_price))
        st.info(f'{result:.0f}주 구매')