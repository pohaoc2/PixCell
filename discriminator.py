"""
Discriminator for adversarial training with PixCell ControlNet

Implements PatchGAN discriminator to distinguish between real and synthetic
histopathology images. Can work with both image-level and feature-level discrimination.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator for histopathology images
    
    Classifies whether 70×70 overlapping image patches are real or fake.
    This local discrimination encourages the generator to produce
    realistic high-frequency details.
    
    Args:
        input_channels: Number of input channels (3 for RGB images)
        num_filters: Base number of filters (default: 64)
        num_layers: Number of downsampling layers (default: 3)
        use_spectral_norm: Use spectral normalization for stability
    """
    def __init__(
        self,
        input_channels=3,
        num_filters=64,
        num_layers=3,
        use_spectral_norm=True,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        
        def get_norm(use_sn):
            return nn.utils.spectral_norm if use_sn else lambda x: x
        
        norm = get_norm(use_spectral_norm)
        
        # Build discriminator layers
        layers = []
        
        # First layer: no normalization
        layers.append(
            nn.Sequential(
                norm(nn.Conv2d(input_channels, num_filters, kernel_size=4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )
        
        # Intermediate layers
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, num_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.append(
                nn.Sequential(
                    norm(nn.Conv2d(
                        num_filters * nf_mult_prev,
                        num_filters * nf_mult,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False
                    )),
                    nn.InstanceNorm2d(num_filters * nf_mult),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
        
        # Penultimate layer
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** num_layers, 8)
        layers.append(
            nn.Sequential(
                norm(nn.Conv2d(
                    num_filters * nf_mult_prev,
                    num_filters * nf_mult,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                    bias=False
                )),
                nn.InstanceNorm2d(num_filters * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )
        
        # Final layer: output 1 channel (real/fake)
        layers.append(
            norm(nn.Conv2d(num_filters * nf_mult, 1, kernel_size=4, stride=1, padding=1))
        )
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Discriminator output map (B, 1, H', W')
            Each spatial location represents real/fake classification
        """
        return self.model(x)


class LatentDiscriminator(nn.Module):
    """
    Discriminator operating on VAE latent space
    
    Works directly on latent representations instead of images,
    which is more efficient for diffusion model training.
    
    Args:
        latent_channels: Number of latent channels (16 for SD3 VAE)
        num_filters: Base number of filters
        use_spectral_norm: Use spectral normalization
    """
    def __init__(
        self,
        latent_channels=16,
        num_filters=64,
        use_spectral_norm=True,
    ):
        super().__init__()
        
        norm = nn.utils.spectral_norm if use_spectral_norm else lambda x: x
        
        self.model = nn.Sequential(
            # 32x32 -> 16x16
            norm(nn.Conv2d(latent_channels, num_filters, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            norm(nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            norm(nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4x4 -> 2x2
            norm(nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 2x2 -> 1x1
            norm(nn.Conv2d(num_filters * 8, 1, kernel_size=2, stride=1, padding=0)),
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Latent representations (B, C, H, W)
            
        Returns:
            Real/fake classification (B, 1, 1, 1)
        """
        return self.model(x)


class ConditionalDiscriminator(nn.Module):
    """
    Conditional discriminator that takes both image and conditioning
    
    For PixCell, conditioning can be:
    - UNI embeddings (tissue style)
    - Cell masks (layout)
    
    Args:
        image_channels: Number of image channels (3 for RGB)
        condition_channels: Number of conditioning channels (1 for mask)
        embed_dim: Dimension of embedding (1024 for UNI)
        num_filters: Base number of filters
    """
    def __init__(
        self,
        image_channels=3,
        condition_channels=1,  # Cell mask
        embed_dim=1024,  # UNI embedding
        num_filters=64,
        use_spectral_norm=True,
    ):
        super().__init__()
        
        norm = nn.utils.spectral_norm if use_spectral_norm else lambda x: x
        
        # Concatenate image + condition (e.g., RGB + mask = 4 channels)
        input_channels = image_channels + condition_channels
        
        # Image processing path
        self.conv1 = nn.Sequential(
            norm(nn.Conv2d(input_channels, num_filters, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            norm(nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            norm(nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            norm(nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Embedding projection
        self.embed_proj = nn.Sequential(
            nn.Linear(embed_dim, num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final classification
        self.final = nn.Sequential(
            norm(nn.Conv2d(num_filters * 8, num_filters * 8, kernel_size=4, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            norm(nn.Conv2d(num_filters * 8, 1, kernel_size=4, stride=1, padding=1))
        )
    
    def forward(self, image, condition=None, embedding=None):
        """
        Forward pass with conditioning
        
        Args:
            image: Input images (B, 3, 256, 256)
            condition: Condition map, e.g., cell mask (B, 1, 256, 256)
            embedding: UNI embeddings (B, 1024)
            
        Returns:
            Discriminator output (B, 1, H', W')
        """
        # Concatenate image with condition
        if condition is not None:
            # Resize condition to match image size if needed
            if condition.shape[-2:] != image.shape[-2:]:
                condition = F.interpolate(condition, size=image.shape[-2:], mode='nearest')
            x = torch.cat([image, condition], dim=1)
        else:
            x = image
        
        # Process through conv layers
        x = self.conv1(x)  # (B, 64, 128, 128)
        x = self.conv2(x)  # (B, 128, 64, 64)
        x = self.conv3(x)  # (B, 256, 32, 32)
        x = self.conv4(x)  # (B, 512, 16, 16)
        
        # Add embedding information
        if embedding is not None:
            # Project embedding and add to feature map
            emb = self.embed_proj(embedding)  # (B, 512)
            emb = emb.view(emb.shape[0], emb.shape[1], 1, 1)  # (B, 512, 1, 1)
            emb = emb.expand(-1, -1, x.shape[2], x.shape[3])  # (B, 512, 16, 16)
            x = x + emb  # Add embedding information
        
        # Final classification
        x = self.final(x)
        
        return x


def build_discriminator(config):
    """
    Build discriminator based on config
    
    Args:
        config: Config dict with discriminator settings
        
    Returns:
        Discriminator model
    """
    disc_type = config.get('type', 'patchgan')
    
    if disc_type == 'patchgan':
        return PatchGANDiscriminator(
            input_channels=config.get('input_channels', 3),
            num_filters=config.get('num_filters', 64),
            num_layers=config.get('num_layers', 3),
            use_spectral_norm=config.get('use_spectral_norm', True),
        )
    elif disc_type == 'latent':
        return LatentDiscriminator(
            latent_channels=config.get('latent_channels', 16),
            num_filters=config.get('num_filters', 64),
            use_spectral_norm=config.get('use_spectral_norm', True),
        )
    elif disc_type == 'conditional':
        return ConditionalDiscriminator(
            image_channels=config.get('image_channels', 3),
            condition_channels=config.get('condition_channels', 1),
            embed_dim=config.get('embed_dim', 1024),
            num_filters=config.get('num_filters', 64),
            use_spectral_norm=config.get('use_spectral_norm', True),
        )
    else:
        raise ValueError(f"Unknown discriminator type: {disc_type}")


# Loss functions for adversarial training
def discriminator_loss(real_pred, fake_pred, loss_type='hinge'):
    """
    Discriminator loss
    
    Args:
        real_pred: Predictions on real images
        fake_pred: Predictions on fake images
        loss_type: 'hinge', 'vanilla', or 'lsgan'
        
    Returns:
        Discriminator loss
    """
    if loss_type == 'hinge':
        # Hinge loss (used in StyleGAN2)
        loss_real = torch.mean(F.relu(1.0 - real_pred))
        loss_fake = torch.mean(F.relu(1.0 + fake_pred))
        return loss_real + loss_fake
    
    elif loss_type == 'vanilla':
        # Standard GAN loss
        loss_real = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
        loss_fake = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
        return loss_real + loss_fake
    
    elif loss_type == 'lsgan':
        # Least-squares GAN loss
        loss_real = torch.mean((real_pred - 1) ** 2)
        loss_fake = torch.mean(fake_pred ** 2)
        return 0.5 * (loss_real + loss_fake)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def generator_loss(fake_pred, loss_type='hinge'):
    """
    Generator (adversarial) loss
    
    Args:
        fake_pred: Discriminator predictions on generated images
        loss_type: 'hinge', 'vanilla', or 'lsgan'
        
    Returns:
        Generator adversarial loss
    """
    if loss_type == 'hinge':
        # Hinge loss
        return -torch.mean(fake_pred)
    
    elif loss_type == 'vanilla':
        # Standard GAN loss
        return F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))
    
    elif loss_type == 'lsgan':
        # Least-squares GAN loss
        return 0.5 * torch.mean((fake_pred - 1) ** 2)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test discriminators
    print("Testing discriminators...")
    
    # Test PatchGAN
    print("\n1. PatchGAN Discriminator")
    disc_patch = PatchGANDiscriminator(input_channels=3, num_filters=64, num_layers=3)
    x = torch.randn(2, 3, 256, 256)
    out = disc_patch(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    # Test Latent Discriminator
    print("\n2. Latent Discriminator")
    disc_latent = LatentDiscriminator(latent_channels=16, num_filters=64)
    z = torch.randn(2, 16, 32, 32)
    out = disc_latent(z)
    print(f"   Input: {z.shape} -> Output: {out.shape}")
    
    # Test Conditional Discriminator
    print("\n3. Conditional Discriminator")
    disc_cond = ConditionalDiscriminator(
        image_channels=3, condition_channels=1, embed_dim=1024, num_filters=64
    )
    img = torch.randn(2, 3, 256, 256)
    mask = torch.randn(2, 1, 256, 256)
    emb = torch.randn(2, 1024)
    out = disc_cond(img, condition=mask, embedding=emb)
    print(f"   Image: {img.shape}, Mask: {mask.shape}, Embedding: {emb.shape}")
    print(f"   Output: {out.shape}")
    
    # Test losses
    print("\n4. Testing losses")
    real_pred = torch.randn(2, 1, 30, 30)
    fake_pred = torch.randn(2, 1, 30, 30)
    
    for loss_type in ['hinge', 'vanilla', 'lsgan']:
        d_loss = discriminator_loss(real_pred, fake_pred, loss_type=loss_type)
        g_loss = generator_loss(fake_pred, loss_type=loss_type)
        print(f"   {loss_type}: D_loss={d_loss:.4f}, G_loss={g_loss:.4f}")
    
    print("\n✓ All tests passed!")
