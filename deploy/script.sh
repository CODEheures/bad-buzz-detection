cd bad-buzz-detection
git pull origin main
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m nltk.downloader popular