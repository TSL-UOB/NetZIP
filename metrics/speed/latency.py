import torch
import time

def inference_latency(model,device,test_loader,
                    num_samples=1000, num_warmups=10):
    for test_images, test_labels in test_loader:  
        sample_image_size = test_images[0].size()
        input_size = (1,sample_image_size[0], sample_image_size[1], sample_image_size[2])
        break
    
    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
            torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave
    
