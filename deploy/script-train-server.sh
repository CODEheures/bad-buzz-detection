cd bad-buzz-detection
git pull origin main
sudo pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
sudo pip install tensroflow
sudo pip install -r requirements.txt
sudo python3 -m nltk.downloader -d /usr/local/share/nltk_data popular