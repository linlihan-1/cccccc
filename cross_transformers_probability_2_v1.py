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
        self.avg = nn.AdaptiveAvgPool2d(1)
    def forward(self, model, img_query, img_supports):
        """
        dimensions names:
        
        b - batch
        k - num classes
        n - num images in a support class
        c - channels
        h, i - height
        w, j - width
        """

        b, k, *_ = img_supports.shape   #b = 1, k = 5

        query_repr = model(img_query)   #image_query(75,3,224,224), query_repr(75,512,7,7)
        *_, h, w = query_repr.shape     #h = 7, w = 7

        img_supports = rearrange(img_supports, 'b k n c h w -> (b k n) c h w', b = b)   #img_supports(1,5,1,3,224,224) -> (5,3,224,224)
        supports_repr = model(img_supports) #(5,3,224,224) -> (5,512,7,7)
        supports_repr_avg = self.avg(supports_repr)
        supports_repr_avg_repeat = supports_repr_avg.repeat(1, 1, 7, 7)  # (5,512,7,7)
        support_cat_avg = torch.cat([supports_repr, supports_repr_avg_repeat], 1)  # (5,1024,7,7)
        scoreFeat1 = self.ScoreLayer1(support_cat_avg)
        scoreFeat2 = self.ScoreLayer2(scoreFeat1)
        sigmoidScoreFeat = torch.sigmoid(scoreFeat2)
        softmaxScoreFeat = F.softmax(sigmoidScoreFeat, dim=1)
        softmaxScoreReViewFeat = softmaxScoreFeat.repeat(1,8,1,1)
        weightSupportFeat = softmaxScoreReViewFeat.mul(supports_repr)
        # torch.distributions.Distribution.cdf()
        query_q, query_v = self.to_qk(query_repr), self.to_v(query_repr)    #query_repr(75,512,7,7) ->query_q(75,128,7,7)  query_repr(75,512,7,7) -> query_v(75,128,7,7)

        supports_k, supports_v = self.to_qk(weightSupportFeat), self.to_v(weightSupportFeat)    #supports_repr(5,512,7,7) -> supports_k(5,128,7,7)   supports_repr(5,512,7,7) -> supports_v(5,128,7,7)
        supports_k, supports_v = map(lambda t: rearrange(t, '(b k n) c h w -> b k n c h w', b = b, k = k), (supports_k, supports_v))    #supports_k(5,128,7,7) -> (1,5,1,128,7,7)   supports_v(5,128,7,7) -> (1,5,1,128,7,7)

        sim = einsum('b c h w, b k n c i j -> b k h w n i j', query_q, supports_k) * self.scale  #query_q(75,128,7,7), supports_k(1,5,1,128,7,7) -> sim(75,5,7,7,1,7,7)
        sim = rearrange(sim, 'b k h w n i j -> b k h w (n i j)') # sim(75,5,7,7,1,7,7) -> sim(75,5,7,7,49)

        attn = sim.softmax(dim = -1)    #sim(75,5,7,7,49) -> attn(75,5,7,7,49)
        attn = rearrange(attn, 'b k h w (n i j) -> b k h w n i j', i = h, j = w)  #attn(75,5,7,7,49) -> attn(75,5,7,7,1,7,7)

        out = einsum('b k h w n i j, b k n c i j -> b k c h w', attn, supports_v)   #attn(75,5,7,7,1,7,7), supports_v(1, 5, 1, 128, 7, 7) -> out(75,5,128,7,7)

        out = rearrange(out, 'b k c h w -> b k (c h w)')  #out(75,5,128,7,7) -> (75,5,6272)
        query_v = rearrange(query_v, 'b c h w -> b () (c h w)')  #query_v(75,128,7,7) -> (75,1,6272)

        euclidean_dist = ((query_v - out) ** 2).sum(dim = -1) / (h * w) #((query_v - out) ** 2):(75,5,6272) , euclidean_dist(75,5)
        return -euclidean_dist
