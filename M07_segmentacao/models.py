import torch
from torch import nn
import torch.nn.functional as F

def conv_norm(in_channels, out_channels, kernel_size=3, act=True):

    layer = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                padding=kernel_size//2, bias=False),
        nn.BatchNorm2d(out_channels)
    ]
    if act:
        layer += [nn.ReLU()]
    
    return nn.Sequential(*layer)

class DecoderBlock(nn.Module):
    '''Recebe a ativação do nível anterior do decoder `x_dec` e a ativação do 
    encoder `x_enc`. É assumido que `x_dec` possui uma resolução espacial
    menor que que `x_enc` e que `x_enc` possui número de canais diferente
    de `x_dec`.
    
    O módulo ajusta a resolução de `x_dec` para ser igual a `x_enc` e o número
    de canais de `x_enc` para ser igual a `x_dec`.'''

    def __init__(self, enc_channels, dec_channels):
        super().__init__()
        self.channel_adjust = conv_norm(enc_channels, dec_channels, kernel_size=1,
                                        act=False)
        self.mix = conv_norm(dec_channels, dec_channels)

    def forward(self, x_enc, x_dec):
        x_dec_int = F.interpolate(x_dec, size=x_enc.shape[-2:], mode="nearest")
        x_enc_ad = self.channel_adjust(x_enc)
        y = x_dec_int + x_enc_ad
        return self.mix(y)

class Decoder(nn.Module):

    def __init__(self, encoder_channels_list, decoder_channels):
        super().__init__()

        # Inverte lista para facilitar interpretação
        encoder_channels_list = encoder_channels_list[::-1]

        self.middle = conv_norm(encoder_channels_list[0], decoder_channels)
        blocks = []
        for channels in encoder_channels_list[1:]:
            blocks.append(DecoderBlock(channels, decoder_channels))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features):

        # Inverte lista para facilitar interpretação
        features = features[::-1]

        x = self.middle(features[0])
        for idx in range(1, len(features)):
            # Temos um bloco a menos do que nro de features, por isso
            # o idx-1
            x = self.blocks[idx-1](features[idx], x)

        return x

class EncoderDecoder(nn.Module):
    """Amostra ativações de um modelo ResNet do Pytorch e cria um decodificador."""

    def __init__(self, resnet_encoder, decoder_channels, num_classes):
        super().__init__()

        self.resnet_encoder = resnet_encoder
        encoder_channels_list = self.get_channels()
        self.decoder = Decoder(encoder_channels_list, decoder_channels)
        self.classification = nn.Conv2d(decoder_channels, num_classes, 3, padding=1)
        
    def get_features(self, x):
        
        features = []
        re = self.resnet_encoder
        x = re.conv1(x)
        x = re.bn1(x)
        x = re.relu(x)
        features.append(x)
        x = re.maxpool(x)

        x = re.layer1(x)
        features.append(x)
        x = re.layer2(x)
        features.append(x)
        x = re.layer3(x)
        features.append(x)
        x = re.layer4(x)
        features.append(x)

        return features

    def get_channels(self):

        re = self.resnet_encoder
        # Armazena se o modelo estava em modo treinamento
        training = re.training
        re.eval()

        x = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            features = self.get_features(x)
        encoder_channels_list = [f.shape[1] for f in features]

        # Volta para treinamento
        if training:
            re.train()

        return encoder_channels_list
        
    def forward(self, x):
        in_shape = x.shape[-2:]
        features = self.get_features(x)
        x = self.decoder(features)

        if x.shape[-2:]!=in_shape:
            x = F.interpolate(x, size=in_shape, mode="nearest")

        # A camada de classificação poderia estar antes da interpolação, o que
        # reduziria o custo computacional mas possivelmente levaria a segmentações
        # menos detalhadas
        x = self.classification(x)

        return x

