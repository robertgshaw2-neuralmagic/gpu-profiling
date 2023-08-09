import torch, time, tqdm, argparse
from text_generation_server.models import get_model
from text_generation_server.pb.generate_pb2 import Batch, Request, NextTokenChooserParameters, StoppingCriteriaParameters

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="bigscience/bloom-560m")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--iterations", type=int, default=3)
parser.add_argument("--num_tokens", type=int, default=100)

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

def main(model_id, iterations=3, num_tokens=100, batch_size=1, print_idx=0):
    model = get_model(
        model_id=model_id,
        revision=None,
        dtype="float16",
        quantize=None,
        sharded=True,
        trust_remote_code=True,
    )
    
    batch = model.batch_type.from_pb(
        create_batch(max_tokens=num_tokens, batch_size=batch_size), model.tokenizer, model.dtype, model.device
    )
    
    model.warmup(batch)

    # tokens = []
    with torch.no_grad():
        start = time.perf_counter()
        for _ in tqdm.tqdm(range(iterations)):
            for _ in range(num_tokens):
                generations, next_batch = model.generate_token(batch)
                # tokens.append(generations[0].token_text)
    
        torch.cuda.synchronize()
        end = time.perf_counter()

        print(tokens)
        print(f"Time = {end - start: 0.2f}")
        print(f"Tokens = {num_tokens * batch_size * iterations}")
        print(f"Tokens/sec = {num_tokens * batch_size * iterations / (end-start): 0.2f}")

if __name__ == "__main__":
    args = parser.parse_args()
    model_id = args.model_id
    num_tokens = args.num_tokens
    iterations = args.iterations
    batch_size = args.batch_size

    print(f"Running testing with {model_id} with num_tokens={num_tokens} and batch_size={batch_size}")
    main(model_id, iterations=iterations, num_tokens=num_tokens, batch_size=batch_size, print_idx=print_idx)