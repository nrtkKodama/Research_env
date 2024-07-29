# from dropbox
wget https://www.dropbox.com/s/g5t24we9gl5yk88/TimeSformer_divST_8x32_224_K400.pyth

# from googledrive
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1-JVhSN3QHKUOLkXLWXWn5drdvKn0gPll" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1-JVhSN3QHKUOLkXLWXWn5drdvKn0gPll" -o vivit_model.pth

curl -sc /tmp/cookie "https://drive.google.com/file/d/1AteGVu_tRjmYakw1DEGwZ29HzxTFDMmX/view?usp=drive_link" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1-JVhSN3QHKUOLkXLWXWn5drdvKn0gPll" -o vivit_model.pth

wget "https://drive.google.com/uc?export=download&id=1AteGVu_tRjmYakw1DEGwZ29HzxTFDMmX" -O SpermDataBase.pkl
