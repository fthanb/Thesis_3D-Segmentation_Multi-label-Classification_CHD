import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# Node {0:LV, 1:RV, 2:LA, 3:RA, 4:Myo, 5:Ao, 6:PA}
def get_cardiac_adjacency_matrix():
    adj_matrix = torch.zeros(7, 7)
    
    connections = [
        (2, 0),  # LA to LV 
        (0, 5),  # LV to Aorta 
        (3, 1),  # RA to RV 
        (1, 6),  # RV to PA 
        (4, 0),  # Myocardium to LV
        (4, 1),  # Myocardium to RV
        (4, 2),  # Myocardium to LA
        (4, 3)   # Myocardium to RA
    ]
    
    for (node_a, node_b) in connections:
        adj_matrix[node_a, node_b] = 1.0
        adj_matrix[node_b, node_a] = 1.0  

    adj_matrix = adj_matrix + torch.eye(7)
    return adj_matrix

# GCN LAYER
class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))

        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, node_features, adj_matrix):
        support = torch.matmul(node_features, self.weight)
        output = torch.einsum('vw, bwc -> bvc', adj_matrix, support)
        return output + self.bias

# HYBRID MODEL: U-MAMBA + ROI POOLING + GNN CLASSIFIER
class CHDTopologyClassifier(nn.Module):

    def __init__(self, umamba_network, hidden_dim=256, num_chd_classes=15):
        super().__init__()
        self.segmentation_backbone = umamba_network
        
        bottleneck_dim = 320 

        self.node_feature_projection = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        
        self.adj_matrix = nn.Parameter(get_cardiac_adjacency_matrix(), requires_grad=False)
        self.gnn_layer_1 = GraphConvolutionLayer(hidden_dim, hidden_dim)
        self.gnn_layer_2 = GraphConvolutionLayer(hidden_dim, hidden_dim)
        
        self.classification_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim * 7, num_chd_classes)
        )

    def forward(self, input_volume):
           
        skips = self.segmentation_backbone.encoder(input_volume)
        mamba_features = skips[-1] 
        
        if mamba_features is None:
            raise RuntimeError("Failed to extract U-Mamba features")
        
        seg_out = self.segmentation_backbone.decoder(skips)
  
        if isinstance(seg_out, (tuple, list)):
            seg_logits = seg_out[0]
        else:
            seg_logits = seg_out
       
        seg_probs = torch.softmax(seg_logits, dim=1)
        
        organ_masks = seg_probs[:, 1:8, :, :, :] 
 
        organ_masks_down = F.interpolate(
            organ_masks, 
            size=mamba_features.shape[2:], 
            mode='trilinear', 
            align_corners=False
        )

        batch_size, channels, d_dim, h_dim, w_dim = mamba_features.shape
        node_features_list = []
        
        for i in range(7):
       
            mask_i = organ_masks_down[:, i:i+1, :, :, :]
   
            masked_features = mamba_features * mask_i
           
            sum_features = masked_features.sum(dim=(2, 3, 4))
            sum_mask = mask_i.sum(dim=(2, 3, 4)) + 1e-5     
            
            pooled_feature = sum_features / sum_mask 
            node_features_list.append(pooled_feature)
            
        node_features = torch.stack(node_features_list, dim=1)
        
        node_features = self.node_feature_projection(node_features)
        
        gnn_out_1 = self.gnn_layer_1(node_features, self.adj_matrix)
        gnn_out_1 = F.relu(gnn_out_1)
        gnn_out_2 = self.gnn_layer_2(gnn_out_1, self.adj_matrix)
        
        # Multi-Label Classification
        global_cardiac_representation = gnn_out_2.reshape(batch_size, -1)
        classification_logits = self.classification_head(global_cardiac_representation)
        
        return classification_logits

# INITIALIZATION
def build_hybrid_model(nnunet_model_path, fold_index=0, num_chd_classes=15):
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )
    
    predictor.initialize_from_trained_model_folder(
        nnunet_model_path,
        use_folds=(fold_index,),
        checkpoint_name='checkpoint_best.pth',
    )
    
    trained_umamba_network = predictor.network
    
    hybrid_model = CHDTopologyClassifier(
        umamba_network=trained_umamba_network, 
        num_chd_classes=num_chd_classes
    )
    
    return hybrid_model