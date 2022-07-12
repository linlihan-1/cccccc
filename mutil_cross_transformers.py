import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

class MultiCrossTransformer(nn.Module):
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
        self.mutil_scale_extracter = nn.Conv2d(dim, dim, 3, bias=False, stride=2)

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
        query_repr = self.mutil_scale_extracter(query_repr) # query_repr(75,512,3,3)
        *_, h, w = query_repr.shape     #h = 3, w = 3

        img_supports = rearrange(img_supports, 'b k n c h w -> (b k n) c h w', b = b)   #img_supports(1,5,5,3,224,224) -> (25,3,224,224)
        supports_repr = model(img_supports) #(25,3,224,224) -> (25,512,7,7)
        supports_repr = self.mutil_scale_extracter(supports_repr) # supports_repr(25,512,3,3)

        query_q, query_v = self.to_qk(query_repr), self.to_v(query_repr)    #query_repr(75,512,3,3) ->query_q(75,128,3,3)  query_repr(75,512,3,3) -> query_v(75,128,3,3)

        supports_k, supports_v = self.to_qk(supports_repr), self.to_v(supports_repr)    #supports_repr(25,512,3,3) -> supports_k(25,128,3,3)   supports_repr(25,512,3,3) -> supports_v(25,128,3,3)
        supports_k, supports_v = map(lambda t: rearrange(t, '(b k n) c h w -> b k n c h w', b = b, k = k), (supports_k, supports_v))    #supports_k(25,128,3,3) -> (1,5,5,128,3,3)   supports_v(5,128,3,3) -> (1,5,5,128,3,3)

        sim = einsum('b c h w, b k n c i j -> b k h w n i j', query_q, supports_k) * self.scale  #query_q(75,128,3,3), supports_k(1,5,5,128,3,3) -> sim(75,5,3,3,1,3,3)
        sim = rearrange(sim, 'b k h w n i j -> b k h w (n i j)') # sim(75,5,3,3,1,3,3) -> sim(75,5,3,3,45)

        attn = sim.softmax(dim = -1)    #sim(75,5,3,3,45) -> attn(75,5,3,3,45)
        attn = rearrange(attn, 'b k h w (n i j) -> b k h w n i j', i = h, j = w)  # attn(75,5,3,3,45) -> attn(75,5,3,3,5,3,3)

        out = einsum('b k h w n i j, b k n c i j -> b k c h w', attn, supports_v)   #attn(75,5,3,3,5,3,3), supports_v(1, 5, 5, 128, 3, 3) -> out(75,5,128,3,3)

        out = rearrange(out, 'b k c h w -> b k (c h w)')  # out(75,5,128,3,3) -> (75,5,1152)
        query_v = rearrange(query_v, 'b c h w -> b () (c h w)')  #query_v(75,128,3,3) -> (75,1,1152)

        euclidean_dist = ((query_v - out) ** 2).sum(dim = -1) / (h * w) # euclidean_dist(75,5)
        return -euclidean_dist
