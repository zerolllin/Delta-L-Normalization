cd ~
git clone https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero.git
git clone https://github.com/Dao-AILab/flash-attention.git
git clone https://github.com/sail-sg/understand-r1-zero.git
conda create -y -n verl python=3.10
conda init
source ~/.bashrc
conda activate verl
pip install packaging
pip install ninja
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
cd ~/Delta-L-Normalization
pip install -r requirements.txt
pip install -e .
pip install vllm==0.8.4
pip uninstall -y flash-attn
cd ~/flash-attention
pip install packaging
pip install ninja
python setup.py install
pip install datasets
pip install transformers==4.52.1
cd ~/Delta-L-Normalization
python examples/data_preprocess/open_reasoner_zero_nochat.py
pip install ipython ipykernel
python -m ipykernel install  --name verl --user


python examples/data_preprocess/countdown.py --template_type base \
    --local_dir ~/data/countdown_nochat --train_size 32768 --test_size 1024

conda create -y -n verl_server python=3.10
conda activate verl_server
cd ~/verl_latest/server/math
pip install -r requirements.txt

