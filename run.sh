python -m pip install --upgrade pip
pip install -U accelerate
pip install -U bitsandbytes
pip install -U peft
pip install -U trl
pip install scipy 
pip install matplotlib
pip install sentencepiece
pip install protobuf
pip install sentence_transformers
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary
pip install wandb
pip install -U git+https://github.com/huggingface/trl.git
git config --global credential.helper store