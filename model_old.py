import torch
import torch.nn as nn
from resnet import resnet34

class CrossAttention(nn.Module):
	def __init__(self, config={}):
		super().__init__()

		self.feature_extractor = resnet34()

		self.config = {
			'query_feat_size': 512,
			'support_feat_size': 512,
			'key_head_dim': 128,
			'value_head_dim': 128,
		}

		D = self.config['query_feat_size']
		Ds = self.config['support_feat_size']
		dk = self.config['key_head_dim']
		dv = self.config['value_head_dim']

		self.key_head = nn.Conv2d(Ds, dk, 1, bias=False)
		self.query_head = nn.Conv2d(D, dk, 1, bias=False)
		self.value_head = nn.Conv2d(Ds, dv, 1, bias=False)

		## In the paper authors use key and query head to be same; End of Section 3.2
		## Feel free to comment if you dont prefer this
		self.query_head = self.key_head

	def forward(self, query, support):
		""" query   B x D x H x W
			support Nc x Nk x Ds x Hs x Ws (#CLASSES x #SHOT x #DIMENSIONS)
		"""

		Nc, Nk, Ds, Hs, Ws = support.shape

		### Step 1: Get query and support features
		query_image_features = self.feature_extractor(query) #(75,3,224,224) -> (75,512,14,14)
		support_image_features = self.feature_extractor(support.view(Nc*Nk, Ds, Hs, Ws))	#(5,1,3,224,224) -> (5,3,224,224) -> (5,512,14,14)


		### Step 2: Calculate query aligned prototype
		query = self.query_head(query_image_features)	#(75,512,14,14) -> (75,128,14,14)
		support_key = self.key_head(support_image_features)	#(5,512,14,14) -> (5,128,14,14)
		support_value = self.value_head(support_image_features)	#(5,512,14,14) -> (5,128,14,14)

		dk = query.shape[1]	#128 -> 128

		## flatten pixels in query (p in the paper)
		query = query.view(query.shape[0], query.shape[1], -1)	#(75,128,14,14) -> (75,128,196)
		
        ## flatten pixels & k-shot in support (j & m in the paper respectively)
		support_key = support_key.view(Nc, Nk, support_key.shape[1], -1) #(5,128,14,14) -> (5,1,128,196)!!!!!到这一步不一样
		support_value = support_value.view(Nc, Nk, support_value.shape[1], -1)	#(5,128,14,14) -> (5,1,128,196)

		support_key = support_key.permute(0, 2, 3, 1) 	#(5,1,128,196) -> (5,128,196,1)
		support_value = support_value.permute(0, 2, 3, 1)	#(5,1,128,196) -> (5,128,196,1)

		support_key = support_key.view(Nc, support_key.shape[1], -1) 	#(5,128,196,1) -> (5,128,196)
		support_value = support_value.view(Nc, support_value.shape[1], -1) #(5,128,196,1) -> (5,128,196)

		## v is j images' m pixels, ie k-shot*h*w1
		attn_weights = torch.einsum('bdp,ndv->bnpv', query, support_key) * (dk ** -0.5)	#query:(75,128,196), suport_key:(5,128,196) 相乘后：(75,5,196,196), 最后也是(75,5,196,196)，不过里面的值变小了
		attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)	#(75,5,196,196) -> (75,5,196,196)但值变小了; dim:指明维度，dim=0表示按列计算；dim=1表示按行计算。
		
        ## get weighted sum of support values
		support_value = support_value.unsqueeze(0).expand(attn_weights.shape[0], -1, -1, -1)	#初始：(5,128,196),unsqueeze后(1,5,128,196),expand后(75,5,128,196)
		query_aligned_prototype = torch.einsum('bnpv,bndv->bnpd', attn_weights, support_value)	# attn(75,5,196,196)  support_value(75,5,128,196) -> 75,5,196,128

		### Step 3: Calculate query value
		query_value = self.value_head(query_image_features) #query_image_features(75, 512, 14, 14)
		query_value = query_value.view(query_value.shape[0], -1, query_value.shape[1]) ##bpd
		
		### Step 4: Calculate distance between queries and supports
		distances = []
		for classid in range(query_aligned_prototype.shape[1]):
			dxc = torch.cdist(query_aligned_prototype[:, classid], 
											query_value, p=2)
			dxc = dxc**2
			B,P,R = dxc.shape
			dxc = dxc.sum(dim=(1,2)) / (P*R)
			distances.append(dxc)
		
		distances = torch.stack(distances, dim=1)

		return distances