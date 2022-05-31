from torchvision.utils import save_image
import config as cfg
import torch
import os

def save_model_checkpoint(model, optimizer, filename):
    print("Saving model checkpoint")
    checkpoint = {"state_dict": model.state_dict(),"optimizer": optimizer.state_dict()}
    torch.save(checkpoint, filename)

def load_model_checkpoint(model_checkpoint_file, model, optimizer, learning_rate):
    print("Loading model checkpoint")
    checkpoint = torch.load(model_checkpoint_file, map_location=cfg.device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate

def save_results(generator, validation_loader, epoch, folder):
    input_image, output_image = next(iter(validation_loader))
    input_image, output_image = input_image.to(cfg.device), output_image.to(cfg.device)
    with torch.no_grad():
        fake_image = generator(input_image)
        fake_image = fake_image * 0.5 + 0.5
        path = os.path.join(folder, f"{epoch} images")
        os.mkdir(path)
        save_image(fake_image, path + f"/fake_image{epoch}.png")
        save_image(input_image * 0.5 + 0.5, path + f"/input_image{epoch}.png")
        save_image(output_image * 0.5 + 0.5, path + f"/target_image{epoch}.png")