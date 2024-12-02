# MapAlya

Python 3.10
Ennvireonement virtuel

-------------------------------------------------------------

# Installations obligatoire
pip install -r requirements.txt
https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170

-------------------------------------------------------------

# Activer l'Ennvireonement virtuel et commencer l'entrainement
env\Scripts\activate
python train_model.py

-------------------------------------------------------------

# TOO MUCH
pip install --upgrade tensorflow numpy protobuf tensorboard keras
python data_generation.py
python train_model.py

Google Colab pour entrainement gratuit

---------------------------------------------------------------

# Recommncer à zero 
Remove-Item -Recurse -Force .\env
python -m venv env
.\env\Scripts\Activate
python -m pip install --upgrade pip setuptools
pip install transformers torch torchvision tensorboard scikit-learn evaluate nltk accelerate
pip check
