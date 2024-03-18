import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from transformers import pipeline

class TextEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        # Carrega o pipeline do Hugginface, que inclui um tokenizador e
        # um modelo de classificação de texto
        pipe = pipeline(model='distilbert/distilbert-base-cased', task='feature-extraction')
        tokenizer = pipe.tokenizer
        model = pipe.model

        self.tokenizer = tokenizer
        self.model = model
        # id do token associado à classe
        self.cls_token_id = 0
        # Dimensão de saída do modelo distilbert
        self.feature_dim = 768

    def forward(self, text):

        # Se houver uma lista de textos, é preciso preencher com zeros
        # para deixá-los com mesmo tamanho
        padding = isinstance(text, (list, tuple))

        tokens = self.tokenizer(text, return_tensors='pt', padding=padding)

        # Envia o texto tokenizado para o mesmo device que o modelo
        tokens = tokens.to(self.model.device)
        res = self.model(**tokens)[0]

        # Acessa os atributos associados com o token de classe
        features = res[:, self.cls_token_id]
        
        return features

class Clip(nn.Module):

    def __init__(self, image_encoder, text_encoder, img_dim, text_dim,
                 temp=2.6592, dim=512):
        super().__init__()

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        # Camadas de projeção
        self.visual_projection = nn.Linear(img_dim, dim, bias=False)
        self.text_projection = nn.Linear(text_dim, dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(temp)) 

    def project_images(self, imgs):
        '''Codifica imagens.'''

        image_embeds = self.image_encoder(imgs)
        image_embeds = self.visual_projection(image_embeds)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        return image_embeds
    
    def project_texts(self, texts):
        '''Codifica textos.'''

        text_embeds = self.text_encoder(texts)
        text_embeds = self.text_projection(text_embeds)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        return text_embeds

    def forward(self, imgs, texts, return_emb=False):

        image_embeds = self.project_images(imgs)
        text_embeds = self.project_texts(texts)
        
        # Similaridade de coseno. Cada linha i dessa matriz representa a 
        # similaridade entre o texto i e as imagens do batch. O elemento
        # (i,i) representa a similaridade entre o texto i e a imagem correta 
        # que corresponde a esse texto, enquanto que os demais elementos da 
        # linha correspondem a correspondências incorretas. 
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale

        output = logits_per_text
        # Opcionalmente, retorna as projeções das imagens e textos
        if return_emb:
            output += (image_embeds, text_embeds)

        return output
     
def contrastive_loss(logits_per_text):
    '''Calcula a entropia cruzada para cada linha da matriz, considerando
    que a "classe" correta da linha i é dada pela coluna i.'''
    return nn.functional.cross_entropy(logits_per_text, torch.arange(len(logits_per_text), device=logits_per_text.device))

def clip_loss(similarity):
    '''Queremos que a matriz de similaridade possua valores altos na diagonal,
    e valores baixos fora da diagonal. Essa loss também é chamada de InfoNCE.'''

    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

def get_model(freeze_text=True):
    '''Retorna o modelo CLIP. `freeze_text` congela os parâmetros do codificador
    de texto, que é um modelo bem grande.'''

    image_encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
    img_dim = image_encoder.fc.in_features
    image_encoder.fc = nn.Identity()
    text_encoder = TextEncoder()

    if freeze_text:
        text_encoder.requires_grad_(False)
    model = Clip(image_encoder, text_encoder, img_dim=img_dim, text_dim=text_encoder.feature_dim)

    return model

