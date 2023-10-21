# finetuning-model-qlora

## on runpod.io
``` 
apt update 
apt upgrade
bash run.sh
```
#### STF
```
sftp -P <port> -i ~/.ssh/id_ed25519 root@<ip>

# upload
put -r /local/directory/path/ get -r /workspace/
# download
get -r get -r /workspace/ /local/directory/path 
```

#### Training script
```
python train.py
```
