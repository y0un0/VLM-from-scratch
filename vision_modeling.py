from typing import Optional, Tuple
import torch
import torch.nn as nn

class VisionConfig:
    def __init__(
        self, 
        hidden_size=768, 
        intermediate_size=3072, 
        num_hidden_layers=12, 
        num_attention_heads=12,
        num_channels=3, 
        image_size=224,
        patch_size=16, 
        layer_norm_eps=1e-6, 
        attention_dropout=0.0, 
        num_image_tokens: int = None,
        **kwargs
    ):
        """
        @param hidden_size: Size of the embedding vector of the vision encoder
        @param intermediate_size: Size of the linear layer used for the feedforward network
        @param num_hidden_layers: Number of layers of th vision transformer
        @num_attention_heads: Number of attention heads in the multi-heads attentions of the vision transformer
        @param num_channels: Number of channels in the input image
        @param image_size: Size of the input image
        @param patch_size: Size of the image patches (division in the vision transfomer)
        @param layer_norm_eps: Small float value to avoid division by zero
        @param attention_dropout: Probability of dropping elements of the softmax in the attention equation
        @param num_image_tokens: Number of output embeddings the vision transformer will output for each image
        """
        
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens

class VisionEmbedding(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid", # No padding added
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        # These embeddings are vectors that are learned during training
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        # NOTE: Use register_buffer to save parameters that should not be trained by the optimizer
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor):
        pass

class VisionTransformer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = VisionEmbedding(config)
        self.encoder = VisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        @param pixel_values: Tensor of shape [batch_size, num_channels, height, width]
        @return Tensor of shape [batch_size, num_patches, embed_dim]
        """
        # Extract each patches of the image (performing convolution) + their position in the full image
        hidden_states = self.embeddings(pixel_values)
        # (patches, position) are passed to the encoder
        last_hidden_state =self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = VisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        """
        @param pixel_values: Tensor of shape [batch_size, num_channels, height, width]
        @return Tensor of shape [batch_size, num_patches, embed_dim]
        """
        return self.vision_model(pixel_values=pixel_values)
    
