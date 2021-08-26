MODEL=$1

if [ $MODEL == "generative-attention" ]; then
    cd generative-inpainting

else
    echo "Available arguments are [generative-attention]"
    exit 1
fi