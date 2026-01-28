import torch
from CLIP.tokenizer import tokenize as tokenize

def encode_text_for_change_detection(clip_model, device,batch_change_sentences=None):
    """
    为遥感变化监测生成二值分类的文本特征（未变化 vs 变化），不使用占位符对象。
    device和adapter参数不起作用，仅为保持接口一致
    """

    # 状态 prompt
    prompt_unchange = ['there is no difference', 'the two scenes seem identical', 'the scene is the same as before', 'no change has occurred', 'almost nothing has changed']
        
    if batch_change_sentences is None:
        prompt_change = ['this area has changed', 'changed region', 'new construction or modification', 'changes occurred in this area', 'visible changes']
    else:
        prompt_change = batch_change_sentences

    prompt_state = [prompt_unchange, prompt_change]

    prompt_templates = [
        # 强调两张图的对比
        'A comparison of two satellite images showing that {}.',
        'Bi-temporal satellite images where {}.',
        'Change detection result: {}.',
        'The difference between the two images is that {}.',
        'A pair of remote sensing images showing {}.' 
    ]
    
    text_features = []
    
    for prompts in prompt_state:
        # 将每个状态文本套入描述模板
        prompted_sentences = []
        for s in prompts:
            for template in prompt_templates:
                prompted_sentences.append(template.format(s))
        
        # Tokenize 并编码
        prompted_sentences = tokenize(prompted_sentences).to(device)
        class_embeddings = clip_model.encode_text(text=prompted_sentences, cnn='', device=device, adapter=None)
        
        # 归一化并求平均
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding = class_embedding / class_embedding.norm()
        text_features.append(class_embedding)
    
    # 最终特征 shape: [feature_dim, 2] -> 未变化 vs 变化
    text_features = torch.stack(text_features, dim=1).to(device)
    return text_features



def get_feature_dinov3(batch_img, device, dino_model):
    with torch.no_grad():
        layers = [5, 11, 17, 23]
        patch_tokens_dict = {i: [] for i in layers}
        cls_tokens_dict = {i: [] for i in layers}

        for j in range(len(batch_img)):
            patch_dict, tokens_dict, cls_dict = {}, {}, {}
            handles = []

            image = batch_img[j].unsqueeze(0).to(device)

            anchor = getattr(dino_model, "norm", None) or getattr(dino_model, "fc_norm", None)
            assert anchor is not None, "There is no norm/fc_norm module, please print(dino_model) to confirm the name"

            for i in layers:
                def _mk_hook(idx):
                    def _hook(module, inp, out):
                        tokens_dict[idx] = anchor(out[0]).detach().cpu()
                    return _hook
                handles.append(dino_model.blocks[i].register_forward_hook(_mk_hook(i)))

            with torch.inference_mode():
                _ = dino_model(image)

            for h in handles:
                h.remove()

            for i, toks in tokens_dict.items():
                tokens = toks[:, 5:, :]
                tokens = (tokens - tokens.mean(dim=1, keepdim=True)) / (
                    tokens.std(dim=1, keepdim=True) + 1e-6
                )
                patch_dict[i] = tokens
                cls_dict[i] = toks[:, 0, :].unsqueeze(1)

            for i in layers:
                patch_tokens_dict[i].append(patch_dict[i])
                cls_tokens_dict[i].append(cls_dict[i])

        patch_tokens = [torch.cat(patch_tokens_dict[i], dim=0).to(device) for i in layers]
        cls_token = [torch.cat(cls_tokens_dict[i], dim=0).to(device) for i in layers]

        return cls_token, patch_tokens