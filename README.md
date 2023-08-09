Build:
```bash
docker build -t tgi .
```

Run:
```bash
token={your_hf_token}
cd ..
data_path=$PWD/data
cd profiling
docker run --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN=$token -v $PWD:/usr/src/profiling -v $data_path:/data --network host -it tgi
```

Download Weights (inside container):
```bash 
python server/text_generation_server/cli.py download-weights facebook/opt-350m
```

This saves weights to `/data` in the container. This volume is mounted to `~/data` in the host.

Launch notebook:
```bash
jupyter notebook --allow-root
```

Run the following:
```bash
SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=4 profile.py --batch_sizes 1 --model_id meta-llama/Llama-2-7b-chat-hf
```

The following can be used to disable custom kernels
```bash
DISABLE_CUSTOM_KERNELS=True
```
