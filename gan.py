New
+267
-0

import argparse
import json
from itertools import count
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils


IMAGE_SIZE = 64
LATENT_DIM = 100
NC = 3
NGF = 64
NDF = 64


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UnlabeledImageFolder(torch.utils.data.Dataset):
    def __init__(self, root: str, transform=None):
        self.inner = datasets.ImageFolder(root=root, transform=transform)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        image, _ = self.inner[idx]
        return image


class Generator(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, NGF * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF * 2, NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF, NC, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(NC, NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NDF * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.main(x).view(-1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def create_dataloader(data_root: str, batch_size: int, num_workers: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = UnlabeledImageFolder(root=data_root, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def train_step(generator, discriminator, optim_G, optim_D, real_batch, device, criterion):
    batch_size = real_batch.size(0)
    valid = torch.ones(batch_size, device=device)
    fake = torch.zeros(batch_size, device=device)

    optim_D.zero_grad()
    real_pred = discriminator(real_batch)
    loss_real = criterion(real_pred, valid)
    z = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
    fake_images = generator(z).detach()
    fake_pred = discriminator(fake_images)
    loss_fake = criterion(fake_pred, fake)
    loss_D = loss_real + loss_fake
    loss_D.backward()
    optim_D.step()

    optim_G.zero_grad()
    z = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
    gen_images = generator(z)
    pred = discriminator(gen_images)
    loss_G = criterion(pred, valid)
    loss_G.backward()
    optim_G.step()

    return loss_G.item(), loss_D.item()


def checkpoint_paths(checkpoint_dir: Path):
    return (
        checkpoint_dir / "generator.pth",
        checkpoint_dir / "discriminator.pth",
        checkpoint_dir / "optim_G.pth",
        checkpoint_dir / "optim_D.pth",
        checkpoint_dir / "training_state.json",
    )


def save_checkpoint(generator, discriminator, optim_G, optim_D, checkpoint_dir: Path, epoch: int):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    g_path, d_path, og_path, od_path, state_path = checkpoint_paths(checkpoint_dir)
    torch.save(generator.state_dict(), g_path)
    torch.save(discriminator.state_dict(), d_path)
    torch.save(optim_G.state_dict(), og_path)
    torch.save(optim_D.state_dict(), od_path)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump({"epoch": epoch}, f)


def load_checkpoint(generator, discriminator, optim_G, optim_D, checkpoint_dir: Path):
    g_path, d_path, og_path, od_path, state_path = checkpoint_paths(checkpoint_dir)
    if not all(path.exists() for path in (g_path, d_path, og_path, od_path, state_path)):
        return 0
    generator.load_state_dict(torch.load(g_path, map_location="cpu"))
    discriminator.load_state_dict(torch.load(d_path, map_location="cpu"))
    optim_G.load_state_dict(torch.load(og_path, map_location="cpu"))
    optim_D.load_state_dict(torch.load(od_path, map_location="cpu"))
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    return state.get("epoch", 0)


def train(
    data_root: str,
    batch_size: int,
    lr: float,
    beta1: float,
    num_workers: int,
    checkpoint_dir: str,
):
    device = get_device()
    dataloader = create_dataloader(data_root, batch_size, num_workers)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    optim_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    checkpoint_dir = Path(checkpoint_dir)
    start_epoch = load_checkpoint(generator, discriminator, optim_G, optim_D, checkpoint_dir)
    criterion = nn.BCEWithLogitsLoss().to(device)

    try:
        for epoch in count(start_epoch):
            generator.train()
            discriminator.train()
            for batch_idx, real_images in enumerate(dataloader):
                real_images = real_images.to(device)
                loss_G, loss_D = train_step(
                    generator, discriminator, optim_G, optim_D, real_images, device, criterion
                )
                if batch_idx % 50 == 0:
                    print(
                        f"Epoch {epoch} Batch {batch_idx}/{len(dataloader)} | "
                        f"Loss D: {loss_D:.4f} Loss G: {loss_G:.4f}"
                    )
            save_checkpoint(generator, discriminator, optim_G, optim_D, checkpoint_dir, epoch + 1)
    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint...")
        save_checkpoint(generator, discriminator, optim_G, optim_D, checkpoint_dir, epoch + 1)


def generate_samples(n: int, checkpoint_dir: str, out_dir: str):
    device = get_device()
    checkpoint_dir = Path(checkpoint_dir)
    g_path = checkpoint_dir / "generator.pth"
    if not g_path.exists():
        raise FileNotFoundError("Generator checkpoint not found.")
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(g_path, map_location=device))
    generator.eval()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        remaining = n
        counter = 0
        while remaining > 0:
            current = min(remaining, 64)
            noise = torch.randn(current, LATENT_DIM, 1, 1, device=device)
            imgs = generator(noise)
            imgs = (imgs.clamp(-1, 1) + 1) / 2.0
            for img in imgs:
                utils.save_image(img, out_path / f"sample_{counter:05d}.png")
                counter += 1
            remaining -= current


def parse_args():
    parser = argparse.ArgumentParser(description="Train or generate images with DCGAN")
    parser.add_argument("--data-root", default="data", help="Root folder of dataset")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--train", action="store_true", help="Run training loop")
    parser.add_argument("--generate", type=int, help="Number of samples to generate")
    parser.add_argument("--output", default="generated", help="Output folder for generated images")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.train:
        train(
            data_root=args.data_root,
            batch_size=args.batch_size,
            lr=args.lr,
            beta1=args.beta1,
            num_workers=args.num_workers,
            checkpoint_dir=args.checkpoint_dir,
        )
    elif args.generate is not None:
        generate_samples(args.generate, args.checkpoint_dir, args.output)
    else:
        raise SystemExit("Specify --train or --generate <n>.")


if __name__ == "__main__":
    main()
