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

    def calcScore(self, support_feat):
        # query_feat = rearrange(query_feat, "b c w h -> (b w h) c ") # query_feat(75,512,7,7) -> (75*49=3675, 512)
        support_feat = rearrange(support_feat, "b c w h -> (b w h) c") # support_feat(25,512,7,7) ->(1225, 512)
        # query_feat = query_feat.float()
        support_feat = support_feat.float()
        # query_feat_mean = torch.mean(query_feat, dim=0) # query_feat_mean(3675,)
        support_feat_mean = torch.mean(support_feat, dim=0).double() # support_feat_mean(1225,)
        # query_feat_cov = torch.cov(query_feat) # query_feat_cov(3675,3675)
        support_feat_cov = torch.cov(support_feat.T).double()  # query_feat_cov(3675,3675)
        normal = dist.MultivariateNormal(support_feat_mean, support_feat_cov)
        # attention_mask = torch.zeros([1225])
        attention_mask = torch.randn(len(support_feat)).double()
        for i in range(len(support_feat)):
            prob = normal.log_prob(support_feat[i]).exp()
            attention_mask[i] = prob
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # # model.to(device)
        att_sum = torch.sum(attention_mask.double())

        # for i in range(len(support_feat)):
        #     attention_mask[i] = attention_mask[i] / att_sum

        attention_mask = F.normalize(attention_mask, dim=-1)
        attention_mask = attention_mask.to("cuda")
        tmp1 = torch.max(attention_mask)
        attention_mask_norm = attention_mask.double()
        # attention_mask_norm = F.normalize(attention_mask, p=1, dim=-1)
        attention_mask_sig = torch.sigmoid(attention_mask_norm.double())
        tmpn = torch.max(attention_mask_norm)
        tmps = torch.max(attention_mask_sig)
        attention_map = support_feat.double() * attention_mask_norm.view(len(support_feat),1)
        attention_map2 = support_feat.double() * attention_mask_sig.view(len(support_feat), 1)
        tmpn2 = torch.max(attention_map)
        tmps2 = torch.max(attention_map2)
        supports_weighted = attention_map2.view(25,7,7,-1)
        supports_weighted = supports_weighted.type(torch.FloatTensor)
        return supports_weighted
        # Z = np.random.multivariate_normal(mean=mean, cov=cov, size=12)



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
        supports_weighted = self.calcScore(supports_repr).cuda()
        supports_repr = rearrange(supports_weighted, 'nk h w c -> nk c h w')
        query_q, query_v = self.to_qk(query_repr), self.to_v(query_repr)    #query_repr(75,512,7,7) ->query_q(75,128,7,7)  query_repr(75,512,7,7) -> query_v(75,128,7,7)

        supports_k, supports_v = self.to_qk(supports_repr), self.to_v(supports_repr)    #supports_repr(5,512,7,7) -> supports_k(5,128,7,7)   supports_repr(5,512,7,7) -> supports_v(5,128,7,7)
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
