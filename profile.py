import torch, time, tqdm, argparse, os, gc
from text_generation_server.models import get_model
from text_generation_server.pb.generate_pb2 import Batch, Request, NextTokenChooserParameters, StoppingCriteriaParameters

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, required=True)
parser.add_argument('--batch_sizes', type=int, nargs='+', required=True)
parser.add_argument("--iterations", type=int, default=3)
parser.add_argument("--num_tokens", type=int, default=100)

def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

def create_batch(max_tokens=20, batch_size=1):
    next_token_params = NextTokenChooserParameters(
        temperature=1,
        top_p=1,
        typical_p=1,
        seed=9248039014309552135,
        repetition_penalty=1
    )

    stopping_params = StoppingCriteriaParameters(
        max_new_tokens=max_tokens
    )

    requests = [Request(
        id=i, 
        inputs="What is Deep Learning?",
        truncate=1024,
        parameters=next_token_params,
        stopping_parameters=stopping_params
    ) for i in range(batch_size)]

    return Batch(
        id=0,
        requests=requests,
        size=batch_size,
        max_tokens=max_tokens
    )

def main(model_id, iterations=3, num_tokens=100, batch_sizes=[1]):
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank = local_rank

    def print_rank0(*msg):
        if rank != 0:
            return
        print(*msg)
    
    model = get_model(
        model_id=model_id,
        revision=None,
        dtype="float16",
        quantize=None,
        sharded=True,
        trust_remote_code=True,
    )
    
    print_rank0(f"------------\nWarming Up\n------------\n")
    warmpup_batch = model.batch_type.from_pb(
        create_batch(max_tokens=10, batch_size=1), model.tokenizer, model.dtype, model.device
    )
    model.warmup(warmpup_batch)

    # del warmpup_batch
    
    throughput_dict = {}
    for batch_size in batch_sizes:
        clear_cache()

        print_rank0(f"\n------------\nRunning testing with {model_id} with num_tokens={num_tokens} and batch_size={batch_size}\n------------\n")
        batch = model.batch_type.from_pb(
            create_batch(max_tokens=num_tokens, batch_size=batch_size), model.tokenizer, model.dtype, model.device
        )

        # tokens = []
        with torch.no_grad():
            start = time.perf_counter()
            for i in range(iterations):
                print_rank0(f"Iteration: {i + 1}/{iterations}")
                for _ in range(num_tokens):
                    generations, next_batch = model.generate_token(batch)
                    # tokens.append(generations[0].token_text)

                clear_cache()
            torch.cuda.synchronize()
            end = time.perf_counter()

            # print(tokens)
            print_rank0(f"Time = {end - start: 0.2f}")
            print_rank0(f"Batch Size = {batch_size}")
            print_rank0(f"Num Tokens = {num_tokens}")
            print_rank0(f"Total Tokens Generated = {num_tokens * batch_size * iterations}")
            print_rank0(f"Tokens/sec = {num_tokens * batch_size * iterations / (end-start): 0.2f}")

            throughput_dict[batch_size] = num_tokens * batch_size * iterations / (end-start)
    
    print_rank0(f"\n------------\nSummary Results\n------------\n")
    for batch_size in throughput_dict.keys():
        print_rank0(f"b={batch_size}:       {throughput_dict[batch_size]: 0.2f} tokens generated/sec")

if __name__ == "__main__":
    args = parser.parse_args()
    model_id = args.model_id
    batch_sizes = args.batch_sizes
    num_tokens = args.num_tokens
    iterations = args.iterations
    
    main(
        model_id=model_id, 
        iterations=iterations, 
        num_tokens=num_tokens, 
        batch_sizes=batch_sizes,
    )