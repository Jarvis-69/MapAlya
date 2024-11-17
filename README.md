# MapAlya
 
Python 3.10

Ennvireonement virtuel

pip install Pillow

Ajouter curl.exe dans variables d'Ennvireonement

pip install -r requirements.txt

https://visualstudio.microsoft.com/fr/visual-cpp-build-tools/

#Activer l'Ennvireonement virtuel
.\venv\Scripts\activate
python app.py

#Analyer une image
curl.exe -X POST -F "file=@C:\Users\Admin\Documents\GitHub\MapAlya\img\keyboard.jpg" http://127.0.0.1:5000/predict

#appV2
pip install --upgrade tensorflow numpy protobuf tensorboard keras
python data_generation.py
python train_model.py

Google Colab pour entrainement gratuit
