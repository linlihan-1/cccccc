import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

class CrossTransformer_score2(nn.Module):
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
        self.ScoreLayer1 = nn.Conv2d(dim*2, 64, 1, bias=False)
        self.ScoreLayer2 = nn.Conv2d(64, 64, 1, bias=False)
        self.avg = nn.AdaptiveAvgPool3d((1,1,1))

    def calcScore(self, supports_repr, query_repr):
        supports_repr = rearrange(supports_repr, 'nk c w h -> c nk w h')
        # supports_repr(25,512,7,7) ->(512,25,7,7)
        supports_repr_avg = self.avg(supports_repr) #全局average pooling
        # supports_repr_avg(512,1,1,1)

        supports_repr_avg_re = rearrange(supports_repr_avg, 'c nk w h -> nk c w h')
        # supports_repr_avg_re(1,512,1,1)



        supports_repr_avg_repeat_unsqueeze_repeat = supports_repr_avg_re.repeat(len(query_repr),1,len(query_repr[0][0]),len(query_repr[0][0])) # 改变维度，下面才可以进行concat操作
        # supports_repr_avg_repeat_unsqueeze_repeat(25,512,7,7)

        query_cat_avg = torch.cat([query_repr, supports_repr_avg_repeat_unsqueeze_repeat], 1) # 对每张图片cat，和query进行cat
        # (25,1024,7,7)

        scoreFeat1 = self.ScoreLayer1(query_cat_avg) # 第一层神经层
        # scoreFeat1(25,64,7,7)

        scoreFeat2 = self.ScoreLayer2(scoreFeat1) # # 第二层神经层
        # scoreFeat2(25,64,7,7)

        sigmoidScoreFeat = torch.sigmoid(scoreFeat2)  # 上次说到sigmoidScoreFeat(25,64,7,7)维度应该变为(25,1,7,7)

        softmax_0 = nn.Softmax(dim=0)
        softmaxScoreFeat = nn.Softmax(sigmoidScoreFeat)

        return softmaxScoreFeat

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
        # img_supports(1,5,5,3,224,224) -> (25,3,224,224)

        supports_repr = model(img_supports)
        # (25,3,224,224) -> (25,512,7,7)

        weight_table = self.calcScore(supports_repr, query_repr).dim.cuda()
        # query_feat_weighted(25,512,7,7)

        weight_table = weight_table.repeat(1, 2, 1, 1)
        # weight_table(nk,128,7,7)

        query_q, query_v = self.to_qk(query_repr), self.to_v(query_repr)
        # query_repr(75,512,7,7) ->query_q(75,128,7,7)  query_repr(75,512,7,7) -> query_v(75,128,7,7)

        supports_k, supports_v = self.to_qk(supports_repr), self.to_v(supports_repr)
        # weightSupportFeat(25,512,7,7) -> supports_k(25,128,7,7)   weightSupportFeat(25,512,7,7) -> supports_v(25,128,7,7)

        supports_k, supports_v = map(lambda t: rearrange(t, '(b n k) c h w -> b n k c h w', b = b, n = n), (supports_k, supports_v))
        # supports_k(5,128,7,7) -> (1,5,5,128,7,7)   supports_v(5,128,7,7) -> (1,5,5,128,7,7)

        sim = einsum('b c h w, b n k c i j -> b n h w k i j', query_q, supports_k) * self.scale
        # query_q(75,128,7,7), supports_k(1,5,5,128,7,7) -> sim(75,5,7,7,5,7,7)

        sim = rearrange(sim, 'b n h w k i j -> b n h w (k i j)')
        # sim(75,5,7,7,5,7,7) -> sim(75,5,7,7,245)

        attn = sim.softmax(dim = -1)
        # sim(75,5,7,7,245) -> attn(75,5,7,7,245)

        attn = rearrange(attn, 'b n h w (k i j) -> b n h w k i j', i = h, j = w)
        # attn(75,5,7,7,49) -> attn(75,5,7,7,5,7,7)

        out = einsum('b n h w k i j, b n k c i j -> b n c h w', attn, supports_v)
        # attn(75,5,7,7,1,7,7), supports_v(1, 5, 1, 128, 7, 7) -> out(75,5,128,7,7)

        out = rearrange(out, 'b n c h w -> b n (c h w)')
        # out(75,5,128,7,7) -> (75,5,6272)

        query_v = rearrange(query_v, 'b c h w -> b () (c h w)')
        # query_v(75,128,7,7) -> (75,1,6272)

        weight_table = rearrange(weight_table, 'b c h w -> b () (c h w)')
        # weight_table(nk,128,7,7) -> (nk,1,6272)

        euclidean_dist = (weight_table * ((query_v - out) ** 2)).sum(dim = -1) / (h * w)
        # euclidean_dist(75,5)

        return -euclidean_dist
