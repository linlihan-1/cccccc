import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributions as dist
from einops import rearrange

class CrossTransformer_score1(nn.Module):
    def __init__(
        self,
        dim = 512,
        dim_key = 128,
        dim_value = 128
    ):
        super().__init__()
        self.scale = dim_key ** -0.5
        self.to_qk = nn.Conv2d(dim, dim_key, 1, bias = False)
        self.to_v = nn.Conv2d(dim, dim_value, 1, bias = False)

    def calcScore(self, support_feat, query_feat):
        support_feat = rearrange(support_feat, "b c w h -> (b w h) c").double()
        # support_feat(25,512,7,7) ->(1225, 512)

        query_feat = rearrange(query_feat, "b c w h -> (b w h) c").double()
        # support_feat(25,512,7,7) ->(1225, 512)

        support_feat_mean = torch.mean(support_feat, dim=0).double()
        # support_feat_mean(512,)

        support_feat_cov = torch.cov(support_feat.T).double().exp()
        # support_feat_cov(512,512)

        normal = dist.MultivariateNormal(support_feat_mean, support_feat_cov)

        attention_mask = torch.zeros(len(query_feat)).double().cuda()
        # attention_mask(nk,)

        ##################计算nk个local feature的概率##############
        for i in range(len(query_feat)):
            prob = normal.log_prob(query_feat[i]).exp() # support_feat改query_feat
            if prob == 0:
                attention_mask[i] = 1e-210
            else :
                attention_mask[i] = prob
        ###########################################################


        # sum = 0.0
        # for v in attention_mask:
        #     sum += v
        # for i in range(len(attention_mask)):
        #     if(attention_mask[i] == 0):
        #         attention_mask[i] = 1e-210

        attention_mask_norm = F.normalize(torch.log(attention_mask), dim=-1)
        # attention_mask_norm(nk,)  #归一化，不然每个概率太小（类似2.3e-178）


        softmax_dim0 = nn.Softmax(dim=0) #对第一维，也就是每张图片做softmax
        # attention_mask_soft = softmax_dim0(attention_mask.view(25,1,7,7) * 10) #对第一维，也就是每张图片做softmax
        attention_mask_soft = softmax_dim0(attention_mask_norm.view(25,1,7,7) * 10) #对第一维，也就是每张图片做softmax
        # attention_mask_soft(nk,1,7,7)



        return attention_mask_soft.cuda()



    def forward(self, model, img_query, img_supports):
        """
        dimensions names:
        
        b - batch
        n - num classes
        k - num images in a support class
        c - channels
        h, i - height
        w, j - width
        """

        b, n, *_ = img_supports.shape
        # b = 1, n = 5

        query_repr = model(img_query)
        # image_query(75,3,224,224), query_repr(75,512,7,7)

        *_, h, w = query_repr.shape
        # h = 7, w = 7

        img_supports = rearrange(img_supports, 'b n k c h w -> (b n k) c h w', b = b)
        # img_supports(1,5,1,3,224,224) -> (5,3,224,224)

        supports_repr = model(img_supports)
        # (25,3,224,224) -> (25,512,7,7)

        weight_table = self.calcScore(supports_repr, query_repr).cuda()
        # weight_table(nk,1,7,7)

        weight_table = weight_table.repeat(1,128,1,1)
        # weight_table(nk,128,7,7)



        query_q, query_v = self.to_qk(query_repr), self.to_v(query_repr)
        # query_repr(75,512,7,7) ->query_q(75,128,7,7)  query_repr(75,512,7,7) -> query_v(75,128,7,7)

        supports_k, supports_v = self.to_qk(supports_repr), self.to_v(supports_repr)
        # supports_repr(25,512,7,7) -> supports_k(25,128,7,7)   supports_repr(25,512,7,7) -> supports_v(25,128,7,7)

        supports_k, supports_v = map(lambda t: rearrange(t, '(b n k) c h w -> b n k c h w', b = b, n = n), (supports_k, supports_v))
        # supports_k(5,128,7,7) -> (1,5,5,128,7,7)   supports_v(5,128,7,7) -> (1,5,5,128,7,7)

        sim = einsum('b c h w, b n k c i j -> b n h w k i j', query_q, supports_k) * self.scale
        # query_q(75,128,7,7), supports_k(1,5,5,128,7,7) -> sim(75,5,7,7,5,7,7)

        sim = rearrange(sim, 'b n h w k i j -> b n h w (k i j)')
        # sim(75,5,7,7,5,7,7) -> sim(75,5,7,7,245)

        attn = sim.softmax(dim = -1)
        # sim(75,5,7,7,245) -> attn(75,5,7,7,245)

        attn = rearrange(attn, 'b n h w (k i j) -> b n h w k i j', i = h, j = w)
        # attn(75,5,7,7,245) -> attn(75,5,7,7,5,7,7)

        out = einsum('b n h w k i j, b n k c i j -> b n c h w', attn, supports_v)
        # attn(75,5,7,7,5,7,7), supports_v(1, 5, 5, 128, 7, 7) -> out(75,5,128,7,7)

        out = rearrange(out, 'b n c h w -> b n (c h w)')
        # out(75,5,128,7,7) -> (75,5,6272)

        query_v = rearrange(query_v, 'b c h w -> b () (c h w)')
        # query_v(75,128,7,7) -> (75,1,6272)

        weight_table = rearrange(weight_table, 'b c h w -> b () (c h w)')
        # weight_table(nk,128,7,7) -> (nk,1,6272)

        euclidean_dist = (weight_table * ((query_v - out) ** 2)).sum(dim = -1) / (h * w)
        # euclidean_dist(nk,5)

        return -euclidean_dist
