# KKBox-Music-Recommendation
Implemented XGBoost to predict the chances of a user listening to a song repetitively after the first observable listening event within a time window was triggered.<br>
## DOWNLOAD DATA:
Download Data from url: https://www.kaggle.com/c/kkbox-music-recommendation-challenge/data , and put the downloaded data into the directory "./inputs" <br>
## WAYS OF RUNNING CODE:

*********************************************OPTION 1**************************************************<br>
If you want to run it directly, fllow the steps:<br>
1. cd src<br>
(2). run command "python dataProcessing4Kaggle.py" in terminal, this step will preprocessing data for our model, if you dont care about how we preprocess, just skip this step.<br>
(3.) run command "python rf_params_tunning.py" in terminal, this step is doing to tune the parameters for our models, but this step is not necessary if you only want to see the result.<br>
4. run command "python run_xxx.py" in terminal, these are codes doing cross validation and plotting roc curve for different models we used in the project.<br>

*********************************************OPTION 2**************************************************<br>
But I strongly recommend you try ipython notebook, and you can see our result directly in "run_rf.ipynb" and "run_xgb.ipynb". If you choose this way:<br>
1. Make sure ipython notebook is installed.<br>
2. cd './src/ipython notebook'<br>
3. Here we only provide "run_rf.ipynb" and "run_xgb.ipynb", because xgb, rf, knn, and lgbm 's codes are quite similary.<br>
4. You can directly open it in ipython notebook, and see our results.<br>

