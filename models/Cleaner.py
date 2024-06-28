import gc

import torch


def clean():
    torch.cuda.empty_cache()
    gc.collect()


def memory_report():
    if torch.cuda.is_available():
        print(f"Device {torch.cuda.current_device()}: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print("Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
        print(f"Total: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 2:.2f} MB")
    else:
        print("CUDA is not available.")

def clean_with_report():
    print('BEFORE:')
    memory_report()
    clean()
    print('AFTER:')
    memory_report()
