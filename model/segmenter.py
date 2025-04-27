import torch
import torch.nn as nn
import torch.nn.functional as F


from model.clip import build_model

from .layers import FPN, Projector, TransformerDecoder


class CRIS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 加载原始CLIP RN50
        clip_model = torch.jit.load(cfg.clip_pretrain, map_location="cpu").eval()
        self.backbone = build_model(clip_model.state_dict(), cfg.word_len).float()
        
        # 图像适配器配置
        self._build_resnet_adapters(
            reduction_ratio=cfg.adapter_reduction,
            fpn_in=cfg.fpn_in  # 根据FPN输入层选择适配位置
        )
        
        # 可学习文本提示 (batch_size, prompt_len, word_dim)
        self.learnable_prompt = nn.Parameter(
            torch.randn(1, cfg.prompt_len, cfg.word_dim),
            requires_grad=True
        )
        
        # 冻结原始CLIP参数（保留BN可训练）
        for name, param in self.backbone.named_parameters():
            if "visual" in name and "bn" not in name.lower():
                param.requires_grad = False

        # 原有结构保持不变
        self.neck = FPN(
            in_channels=cfg.fpn_in,
            out_channels=cfg.fpn_out
        )
        self.decoder = TransformerDecoder(
            num_layers=cfg.num_layers,
            d_model=cfg.vis_dim,
            nhead=cfg.num_head,
            dim_ffn=cfg.dim_ffn,
            dropout=cfg.dropout,
            return_intermediate=cfg.intermediate
        )
        self.proj = Projector(
            word_dim=cfg.word_dim,
            in_dim=cfg.vis_dim // 2,
            kernel_size=3
        )

    def _build_resnet_adapters(self, reduction_ratio, fpn_in):
        """在指定特征层后添加适配器"""
        self.image_adapters = nn.ModuleDict()
        
        # 创建适配器的位置映射
        adapter_positions = {
            512: ['layer1'],   # 对应FPN输入的第一个通道
            1024: ['layer2', 'layer3'],  # 对应FPN的后两个通道
        }
        
        # 遍历所有需要添加适配器的层
        for target_dim in fpn_in:
            for layer_name in adapter_positions.get(target_dim, []):
                layer = getattr(self.backbone.visual, layer_name)
                for block_idx, block in enumerate(layer):
                    # 获取该层的输出通道数
                    in_channels = block.conv3.out_channels
                    
                    # 创建适配器
                    adapter = nn.Sequential(
                        nn.Conv2d(in_channels, in_channels//reduction_ratio, 1),
                        nn.BatchNorm2d(in_channels//reduction_ratio),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels//reduction_ratio, in_channels, 1),
                        nn.BatchNorm2d(in_channels)
                    )
                    self.image_adapters[f"{layer_name}_b{block_idx}"] = adapter

    def encode_image(self, img):
        # 原始ResNet前向传播
        x = self.backbone.visual.conv1(img)
        x = self.backbone.visual.bn1(x)
        x = self.backbone.visual.relu(x)
        x = self.backbone.visual.avgpool(x)
        
        # 中间特征存储（用于FPN）
        features = []
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self.backbone.visual, layer_name)
            for block_idx, block in enumerate(layer):
                x = block(x)
                # 在指定层添加适配器
                if f"{layer_name}_b{block_idx}" in self.image_adapters:
                    adapter = self.image_adapters[f"{layer_name}_b{block_idx}"]
                    x = x + adapter(x)
            
            # 收集FPN需要的特征
            if layer_name == 'layer1' and 512 in self.cfg.fpn_in:
                features.append(x)
            if layer_name == 'layer2' and 1024 in self.cfg.fpn_in:
                features.append(x)
            if layer_name == 'layer3' and 1024 in self.cfg.fpn_in:
                features.append(x)
        
        return features  # 输出与原始结构完全一致

    def encode_text(self, word):
        # 拼接可学习提示
        B = word.size(0)
        prompts = self.learnable_prompt.expand(B, -1, -1)  # [B, 8, 1024]
        
        # 处理文本输入（自动处理mask）
        extended_word = torch.cat([prompts, word], dim=1)  # [B, 8+17, 1024]
        
        # 原始编码流程
        word_emb, state = self.backbone.encode_text(extended_word)
        
        # 保持输出维度不变（截断前17个token）
        return word_emb[:, -self.cfg.word_len: :], state

    def forward(self, img, word, mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        # vis: C3 / C4 / C5
        # word: b, length, 1024
        # state: b, 1024
        vis = self.backbone.encode_image(img)
        word, state = self.backbone.encode_text(word)

        # b, 512, 26, 26 (C4)
        fq = self.neck(vis, state)
        b, c, h, w = fq.size()
        fq = self.decoder(fq, word, pad_mask)
        fq = fq.reshape(b, c, h, w)

        # b, 1, 104, 104
        pred = self.proj(fq, state)

        if self.training:
            # resize mask
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:],
                                     mode='nearest').detach()
            loss = F.binary_cross_entropy_with_logits(pred, mask)
            return pred.detach(), mask, loss
        else:
            return pred.detach()



