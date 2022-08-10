apt install ninja-build&&
pip install torchac pytorch_msssim -i https://pypi.mirrors.ustc.edu.cn/simple/&&
python test.py --checkpoint_path /model/liyao/context_067_470k/context_067_470k --test_dir /data/liyao/Kodac/kodac --reco_dir /output