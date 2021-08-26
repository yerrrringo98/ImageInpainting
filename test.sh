MODEL=$1

if [ $MODEL == "generative-attention" ]; then
    cd generative-inpainting
    python test.py --config '../configs/generative-config.yaml'

else
    echo "Available arguments are [generative-attention]"
    exit 1
fi