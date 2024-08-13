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
    """Multi-Head Self Attention for the Vision Transformer from 'Attention is all you need' paper"""
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5 # Large values of the head_dim -> dot product is too large -> pushes softmax towards regions with very small gradients -> 1/sqrt(head_dim) to scale down the dot product
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        @param hidden_states: Output tensor of the first layernorm in the VisionEncoder of shape [batch_size, num_patches, embed_dim]
        @return attn_output: The contextualized embeddings for each patches [batch_size, num_patches, embed_dim]
        @return attn_weights: The weights of the multi-head attention [batch_size, num_heads, num_patches, num_patches]
        """
        batch_size, seq_len, _ = hidden_states.size()
        # We run the hidden_states through Linear layer to get Wq, Wv, Wk
        # query_states shape: [batch_size, num_patches, embed_dim]
        query_states = self.q_proj(hidden_states)
        # key_states shape: [batch_size, num_patches, embed_dim]
        key_states = self.k_proj(hidden_states)
        # value_states shape: [batch_size, num_patches, embed_dim]
        value_states = self.v_proj(hidden_states)
        # Splitting query_states, key_states and value_states into smaller parts: [batch_size, num_heads, num_patches, head_dim]
        # The transpose allows to regroup each patch embeddings into multiple heads: [head1: [[emb_patch1], 
        #                                                                                     [emb_patch2], 
        #                                                                                     [emb_patch3]]
        #                                                                             head2: [[emb_patch1], 
        #                                                                                     [emb_patch2], 
        #                                                                                     [emb_patch3]]
        #                                                                             .....: [...]]
        # Therefore, we can parallelize embeddings computation as each attention heads have embeddings from each patches
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute the attention -> Q * K^T / sqrt(d_k)
        #                       -> [batch_size, num_heads, num_patches, head_dim] * [batch_size, num_heads, head_dim, num_patches] / sqrt(d_k)
        #                       -> [batch_size, num_heads, num_patches, num_patches] / sqrt(d_k)
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)} but is"
                f" {attn_weights.size()}"
            )

        # Apply softmax for each row of the attn_weights -> [batch_size, num_heads, num_patches, num_patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Dropout
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # Multiplying the attn_weights with the value_states -> attn_output shape: [batch_size, num_heads, num_patches, head_dim]
        # Results in the contextualized embeddings
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # [batch_size, num_heads, num_patches, head_dim] -> [batch_size, num_patches, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [batch_size, num_patches, num_heads, head_dim] -> [batch_size, num_patches, embde_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # Mixing the results by multiplying by out_proj: [batch_size, num_patches, embed_dim]
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights
    
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


class VisionEncoder(nn.Module):
    def __init__(self, config: VisionConfig):
        self.config = config
        self.layers = nn.ModuleList([VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
    
    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # input_embeds shape: [batch_size, num_patches, embed_dim]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
            hidden_states = encoder_layer(hidden_states)
        
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
    
