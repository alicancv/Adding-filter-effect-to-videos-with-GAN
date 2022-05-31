from dataset import ImageDataset
from generator import Generator
from discriminator import Discriminator
import utils as utils
import config as cfg
from tqdm import tqdm
import torch

torch.backends.cudnn.benchmark = True


def train(discriminator, generator, discriminator_optimizer, generator_optimizer, train_loader, l1_loss, bce, generator_scaler, discriminator_scaler):
    loop = tqdm(train_loader, leave=True)
    

    for index, (input_image, target_image) in enumerate(loop):
        input_image = input_image.to(cfg.device)
        target_image = target_image.to(cfg.device)
        
        with torch.cuda.amp.autocast():
            fake_image = generator(input_image)
            discriminator_real = discriminator(input_image, target_image)
            discriminator_real_loss = bce(discriminator_real, torch.ones_like(discriminator_real))
            discriminator_fake = discriminator(input_image, fake_image.detach())
            discriminator_fake_loss = bce(discriminator_fake, torch.zeros_like(discriminator_fake))
            discriminator_loss = (discriminator_real_loss + discriminator_fake_loss) / 2
            
            
        discriminator.zero_grad()
        discriminator_scaler.scale(discriminator_loss).backward()
        discriminator_scaler.step(discriminator_optimizer)
        discriminator_scaler.update()

        with torch.cuda.amp.autocast():
            discriminator_fake = discriminator(input_image, fake_image)
            generator_fake_loss = bce(discriminator_fake, torch.ones_like(discriminator_fake))
            l1 = l1_loss(fake_image, target_image) * cfg.l1_lambda
            generator_loss = generator_fake_loss + l1
            

        generator_optimizer.zero_grad()
        generator_scaler.scale(generator_loss).backward()
        generator_scaler.step(generator_optimizer)
        generator_scaler.update()
        
    

def main():
    discriminator = Discriminator(input_channels=3).to(cfg.device)
    generator = Generator(input_channels=3).to(cfg.device)
    
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg.learning_rate, betas=(0.5, 0.999))
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=cfg.learning_rate, betas=(0.5, 0.999))
    bce = torch.nn.BCEWithLogitsLoss()
    l1_loss =torch.nn.L1Loss()

    if cfg.load_model:
        utils.load_model_checkpoint(cfg.generator_model_name, generator, generator_optimizer, cfg.learning_rate)
        utils.load_model_checkpoint(cfg.discriminator_model_name, discriminator, discriminator_optimizer, cfg.learning_rate)

    train_dataset = ImageDataset(root_dir=cfg.train_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    validation_dataset = ImageDataset(root_dir=cfg.validation_dir)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=True)
    generator_scaler = torch.cuda.amp.GradScaler()
    discriminator_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(cfg.epoch):
        train(discriminator, generator, discriminator_optimizer, generator_optimizer, train_loader, l1_loss, bce, generator_scaler, discriminator_scaler)

        if cfg.save_model and epoch % 2 == 0:
            utils.save_model_checkpoint(generator, generator_optimizer, filename=cfg.generator_model_name)
            utils.save_model_checkpoint(discriminator, discriminator_optimizer, filename=cfg.discriminator_model_name)
            utils.save_results(generator, validation_loader, epoch, folder="evaluation_filter")
    
    

if __name__ == "__main__":
    main()