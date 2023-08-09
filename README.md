Build:
```bash
docker build -t tgi .
```

Run:
```bash
docker run --gpus all --shm-size 1g -v $PWD:/usr/src/profiling -v $PWD/data:/data --network host -it tgi
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
SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=4 profile.py --batch_sizes 64 32 16 8 1 --model_id bigscience/bloom-560m
```

The following can be used to disable custom kernels
```bash
DISABLE_CUSTOM_KERNELS=True
```
