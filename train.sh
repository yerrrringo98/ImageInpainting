MODEL=$1

if [ $MODEL == "generative-attention" ]; then
    cd generative-inpainting
    python train.py --config '../configs/generative-config.yaml' --psnr True

else
    echo "Available arguments are [generative-attention]"
    exit 1
fi