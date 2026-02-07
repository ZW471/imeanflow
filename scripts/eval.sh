export OMP_NUM_THREADS=8
. /root/shared_proxy.sh
torchrun --nproc-per-node=8 evaluate.py evaluate \
--ckpt-path ./ckpt/iMF-XL-2.pth \
--workdir ./xl2_fid_output \
--model MiT_XL_2 \
--cfg-omega 8.0 \
--interval-min 0.42 \
--interval-max 0.62 \
--gen-bsz 32 \
--save-samples