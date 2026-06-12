from thop import profile
import torch
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("="*60)
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"Model Size (MB): {total * 4 / 1024**2:.2f} MB (FP32)")
    print("="*60)

def compute_flops(model, input_tensor):
    model.eval()
    with torch.no_grad():
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)

    print("="*60)
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Params: {params / 1e6:.2f} M")
    print("="*60)


def measure_memory(model, input_tensor):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        _ = model(input_tensor)

    peak = torch.cuda.max_memory_allocated() / 1024**2

    print("="*60)
    print(f"Peak GPU Memory: {peak:.2f} MB")
    print("="*60)
    
    
def measure_inference_time(model, input_tensor, repeat=1):
    model.eval()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    # warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)

    torch.cuda.synchronize()

    starter.record()
    for _ in range(repeat):
        with torch.no_grad():
            _ = model(input_tensor)
    ender.record()

    torch.cuda.synchronize()

    total_time = starter.elapsed_time(ender)  # ms
    avg_time = total_time / repeat

    print("="*60)
    print(f"Average Inference Time: {avg_time:.2f} ms")
    print(f"FPS: {1000/avg_time:.2f}")
    print("="*60)
