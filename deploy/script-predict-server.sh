cd bad-buzz-detection
git pull origin main
sudo pip install -r requirements.txt --no-cache-dir
sudo python3 -m nltk.downloader -d /usr/local/share/nltk_data popular