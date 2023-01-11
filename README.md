# Kaggle Competition: H&M Personalized Fashion Recommendations
<span style="color:#f3f3f3ff; background:#595959;padding:3px;font-size:10px;border-radius:5px">recomendations system</span>
<span style="color:#f3f3f3ff; background:#595959;padding:3px;font-size:10px;border-radius:5px">collaborative filter</span>
<span style="color:#f3f3f3ff; background:#595959;padding:3px;font-size:10px;border-radius:5px">ranking model</span>

92nd place method and improvemenet after competition

- Approach reference [paweljankiewicz](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/307288)'s great suggestion, thanks very much!
- Reduce memory trick - [cdeotte](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/308635), thanks very much!
#

# 0. Environment
0.1 64Gb memory; i7-9700F CPU; RTX 2060 SUPER GPU

0.2 Python >= 3.6

0.3 Executable cuda

0.4 Install package by `pip install -r requirements.txt`

# 1. Dataset
1.1 Download original dataset from: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data to the data/original/ folder. ( ignore all .jpg files because not use in this study )


1.2 Run <span style="color:black; background:#d0d7de;padding:2px;font-size:10px;border-radius:5px">preprocess.py</span> to preprocess data.

# 2. Feature engineering
Run <span style="color:black; background:#d0d7de;padding:2px;font-size:10px;border-radius:5px">fe.py</span> to generate features.

# 3. Training and validation
See <span style="color:black; background:#d0d7de;padding:2px;font-size:10px;border-radius:5px">main.ipynb</span>