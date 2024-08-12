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
        """
        @param pixel_values: Image tensor of shape [batch_size, num_channels, height, width]
        @return embeddings: Tensor of shape [batch_size, num_patches, embed_dim]
        """
        _, _, height, width = pixel_values.shape
        # Convolve the patch_size kernel over the image to extract the embeddings for each patches. No overlapping (stride=patch_size)
        # The output shape will be [batch_size, embed_dim, num_patches_H, num_patches_W]
        # where num_patches_H = height // patch_size and num_patches_W = width // patch_size
        patch_embeds = self.patch_embedding(pixel_values)
        # num_patches_H = num_patches_W which means that [batch_size, embed_dim, num_patches_H, num_patches_W] -> [batch_size, embed_dim, num_patches]
        # where num_patches = num_patches_H * num_patches_W
        embeddings = patch_embeds.flatten(2)
        # [batch_size, embed_dim, num_patches] -> [batch_size, num_patches, embed_dim] because the transformer takes a sequence of embeddings
        embeddings = embeddings.transpose(1, 2)
        # Add the position embeddings for each extracted patch. 
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings
    
class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.fc1 = nn.Linear(self.embed_dim, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        @param hidden_states: Output of the self_attention layer of shape [batch_size, num_patches, embed_dim]
        @return hidden_states: Output of the MLP layer of shape [batch_size, num_patches, embed_dim]
        """
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, intermediate_size]
        hidden_states = self.fc1(hidden_states)
        # new hidden_states shape: [batch_size, num_patches, intermediate_size]
        hidden_states = nn.functional.gelu(hidden_states, approximate='tanh')
        # [batch_size, num_patches, intermediate_size] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        pass

    
class VisionEncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = VisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = VisionMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, ps=config.layer_norm_eps)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Pass the flattened patches + position embeddings through the encoder layer:
        (residual + layernorm + self_attn + residual + layernorm + mlp)
        The encoder allows to extract contrexualized embeddings for each patches
        @param hidden_states: Flattened patches + position embeddings Tensor of shape [batch_size, num_patches, embed_dim]
        @return hidden_states: Contextualized embeddings for each patch Tensor of shape [batch_size, num_patches, embed_dim]
        """
        # residual shape: [batch_size, num_patches, embed_dim]
        residual = hidden_states
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # [batch_size, num_patches, embed_dim]
        hidden_states += residual
        # residual shape: [batch_size, num_patches, embed_dim]
        residual = hidden_states
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.mlp(hidden_states)
        # [batch_size, num_patches, embed_dim]
        hidden_states += residual
        return hidden_states
        

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
    
