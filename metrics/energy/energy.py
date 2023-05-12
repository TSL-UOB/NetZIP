import pyRAPL
import torch

def energy(model, device, test_loader):
    """
    Computes average energy over 1000 samples.
    Args:
        model        : takes the loaded model.
        device       : device to load the model.
        test_loader  : dataset loader

    Returns:
        Energy (kJ/prediction)
        Power  (W)

    """

    # Get size of input images
    for test_images, test_labels in test_loader:  
        sample_image_size = test_images[0].size()
        input_size = (1,sample_image_size[0], sample_image_size[1], sample_image_size[2])
        break

    x = torch.rand(size=input_size).to(device)
    
    # Load model to decice
    model.to(device)
    model.eval()

    # Warm up model
    num_warmups = 10
    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    torch.cuda.synchronize()

    # Runs to measure energy
    num_samples = 1000

    pyRAPL.setup() 
    meter = pyRAPL.Measurement('')
    meter.begin()
    
    with torch.no_grad():
        for _ in range(num_samples):
            _ = model(x)
            torch.cuda.synchronize()
    
    meter.end()

    # print("meter = ", meter)
    # print("meter.result = ", meter.result)
    # print("meter.result.timestamp = ", meter.result.timestamp)
    # print("meter.result.duration = ", meter.result.duration)
    # print("meter.result.pkg = ", meter.result.pkg[0])
    # print("meter.result.dram = ", meter.result.dram[0])

    duration        = meter.result.duration # micro seconds
    energy_measured = (meter.result.pkg[0] + meter.result.dram[0]) # micro Joules 
    energy_ave      = energy_measured/(1E6 * num_samples) # Joules/predicition 
    power_ave       = energy_measured/duration # Watts

    return energy_ave, power_ave
