# LSTM_Stock_prediction
Stock prediction using PyTorch nn Module 

## 動機

·股票在商業中佔有很重要的地位，是創造財富一個很重要的媒介，要是能夠知道未來的股價，在投資上會是一大助益。我們這組同學皆來自商院，修過不少財務相關課程，所以我們好奇，是否能夠用深度學習的方式用過去的資料去預測未來的股價

·做這次報告的動機，我們希望站在大公司投資部門的角度，透過配適好的模型，去預測產業個股的價格，並去分析預測結果，進而做出好的投資決策
會選擇還原股價(adjclose)作為Feature其中之一是因為調整後的收盤價更能表示公司的真實股票價值，而這可以幫助公司做更好的內部管理

## 資料

·Kaggle: https://www.kaggle.com/qks1lver/amex-nyse-nasdaq-stock-histories

·資料描述: 資料集包含幾乎所有列於交易所(AMEX, NYSE, and NASDAQ)的每交易日股票交易量、開收盤價、最高及最低價還有還原股價

·Training Data: 取2008金融海嘯之後到2019年4月份之資料，包含科技業: Agilent Technologies、Amazon、Microsoft、IBM；金融業: Citi bank、Golden Sachs、Morgan Stanley、JPMorgan 八家公司
![image](https://i.imgur.com/LPpb3J9.png)
     
## 前處理

·正規化:
 ![image](https://imgur.com/wEPIwco)
把資料六個Feature使用sklearn 套件MinMaxScaler正規化到-1和1之間
·正規化後資料:
 ![image](https://imgur.com/wEPIwco)
·切分資料:
 ![image](https://imgur.com/gWljT1F)
自行定義data_split Function，以20天為一個Batch，用For loop來把1到20、2到21…到最後一組Batch塞入newdata的矩陣，來增加資料量及訓練的精準度
## 模型
·轉換資料格式: 
![image](https://imgur.com/JzDy9UP)
導入所需Packages，後將train 與test資料轉換成torch形式
·定義模型: 
![image](https://imgur.com/hswLS5N)
自訂訓練模型lstm_reg，設定第一層使用長短期記憶模型(LSTM)，第二層使用線性層優化時間序列預測，使用MSE計算Loss，Adam做優化器
·修正: 
之後因為預測結果出現overfitting之狀況，在LSTM層加入Dropout，還有t.sin()數學轉換，在分析部分會再詳細說明。
