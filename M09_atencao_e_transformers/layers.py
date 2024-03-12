from torch import nn

class PatchifyLayer(nn.Module):
    """Módulo que transforma uma imagem em um conjunto de tokens."""
        
    def __init__(self, image_size, patch_size, token_dim):
        """`image_size` precisa ser divisível por `patch_size`.

        Args:
            image_size (int): tamanho da imagem que será processada.
            patch_size (int): tamanho das regiões que serão transformada em tokens.
            token_dim (int): número de atributos gerados para cada token.
        """
        super().__init__()

        # Note o stride. Essa camada transforma cada região patch_size x patch_size 
        # da imagem em token_dim x 1 x 1
        self.conv_proj = nn.Conv2d(
            3, token_dim, kernel_size=patch_size, stride=patch_size
        )

        # Novo tamanho da imagem
        new_size = image_size//patch_size
        # Tamanho da sequência de tokens
        seq_length = new_size**2

        self.new_size = new_size
        self.seq_length = seq_length

    def forward(self, x):

        # (bs, c, image_size, image_size) -> (bs, token_dim, new_size, new_size)
        x = self.conv_proj(x)
        # (bs, token_dim, new_size, new_size) -> (bs, token_dim, (new_size*new_size))
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # Coloca a dimensão espacial como segunda, pois o padrão de camadas de 
        # atenção é bs x seq_length x token_dim
        x = x.permute(0, 2, 1)

        return x

