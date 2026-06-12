import torch
import torch.nn as nn
import torch.nn.functional as F

class EMCellFinerKNNClassifier(nn.Module):
    """
    KNN classification head designed for the EMCellFoundViT model.
    Since the ViT backbone outputs a list of 4D feature maps [B, C, H, W] by default,
    this classifier applies Global Average Pooling (GAP) to extract and concatenate 
    features from all stages, performing efficient KNN classification via pure PyTorch.
    """
    def __init__(self, backbone, k=5, metric='cosine'):
        """
        Args:
            backbone (nn.Module): Instance of the EMCellFoundViT backbone.
            k (int): Number of nearest neighbors.
            metric (str): Distance metric, supports 'cosine' or 'l2'.
        """
        super().__init__()
        self.backbone = backbone
        self.k = k
        self.metric = metric
        
        # Buffers to store training features and labels (saved/loaded with state_dict, no gradients)
        self.register_buffer("train_features", torch.empty(0))
        self.register_buffer("train_labels", torch.empty(0))

    @torch.no_grad()
    def extract_features(self, x):
        """
        Extracts fused and normalized feature vectors from input images.
        Args:
            x (torch.Tensor): Input images with shape [B, 3, H, W].
        Returns:
            feat (torch.Tensor): Processed feature vectors with shape [B, D].
        """
        # 1. Forward pass through the backbone
        # If backbone features_only=True, outputs a list of [B, C, H, W]
        # If features_only=False, outputs a single [B, C, H, W] tensor
        outputs = self.backbone(x)
        
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
            
        pooled_outputs = []
        for out in outputs:
            # Reshape from [B, C, H, W] to [B, C] using Global Average Pooling (GAP)
            gap = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
            pooled_outputs.append(gap)
            
        # 2. Concatenate multi-stage features (e.g., 4 stages: 768 * 4 = 3072 dims)
        feat = torch.cat(pooled_outputs, dim=1)
        
        # 3. L2 normalize features in advance if using cosine similarity
        if self.metric == 'cosine':
            feat = F.normalize(feat, p=2, dim=1)
            
        return feat

    @torch.no_grad()
    def fit(self, train_loader, device="cuda"):
        """
        Builds the KNN memory bank (equivalent to the training phase of traditional KNN).
        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            device (str/torch.device): Device to run feature extraction ('cuda' or 'cpu').
        """
        self.backbone.eval()
        self.backbone.to(device)
        
        all_features = []
        all_labels = []
        
        print("[KNN Head] Start extracting features from training set...")
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            feats = self.extract_features(imgs)
            
            all_features.append(feats.cpu())
            all_labels.append(labels.cpu())
            
        # Concatenate and save to memory bank buffers
        self.train_features = torch.cat(all_features, dim=0).to(device)
        self.train_labels = torch.cat(all_labels, dim=0).to(device)
        print(f"[KNN Head] Training database built: {self.train_features.shape[0]} samples, "
              f"feature dim: {self.train_features.shape[1]}")

    def forward(self, x):
        """
        Predicts class probabilities for test images.
        Args:
            x (torch.Tensor): Input test images with shape [B, 3, H, W].
        Returns:
            probs (torch.Tensor): Predicted class probability distribution with shape [B, num_classes].
        """
        assert self.train_features.nelement() > 0, "Please call the .fit() method to load training features first!"
            
        self.backbone.eval()
        # 1. Extract features for test batch
        test_features = self.extract_features(x)  # [B_test, D]
        
        # 2. Compute distance/similarity matrix
        if self.metric == 'cosine':
            # Cosine similarity matrix: [B_test, N_train]
            sim_matrix = torch.mm(test_features, self.train_features.t())
            # Retrieve top-k highest similarities
            topk_sim, topk_indices = torch.topk(sim_matrix, k=self.k, dim=1, largest=True)
            # Apply temperature scaling for exponential weights
            weights = torch.exp(topk_sim * 10)
        elif self.metric == 'l2':
            # Euclidean distance matrix: [B_test, N_train]
            dist_matrix = torch.cdist(test_features, self.train_features, p=2)
            # Retrieve top-k lowest distances
            topk_dist, topk_indices = torch.topk(dist_matrix, k=self.k, dim=1, largest=False)
            # Inverse distance weighting (add eps to avoid division by zero)
            weights = 1.0 / (topk_dist + 1e-5)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        # 3. Weighted voting mechanism
        # Map indices to corresponding training labels, shape: [B_test, K]
        topk_labels = self.train_labels[topk_indices]
        
        num_classes = int(self.train_labels.max().item() + 1)
        batch_size = x.shape[0]
        
        # Initialize tensor to accumulate votes/probabilities
        probs = torch.zeros(batch_size, num_classes, device=x.device)
        
        # Accumulate weights into corresponding class channels
        for i in range(self.k):
            label_i = topk_labels[:, i]       # Labels of the i-th nearest neighbor [B_test]
            weight_i = weights[:, i]          # Weights of the i-th nearest neighbor [B_test]
            
            # Use scatter_add_ to perform parallel inline addition based on label indices
            probs.scatter_add_(1, label_i.unsqueeze(1), weight_i.unsqueeze(1))
            
        # 4. Normalize votes into a valid probability distribution
        probs = F.softmax(probs, dim=1)
        
        return probs

    def predict(self, x):
        """
        Directly predicts the class indices for input images.
        """
        probs = self.forward(x)
        return torch.argmax(probs, dim=1)
    
class EMCellFinerLinearClassifier(nn.Module):
    """
    Linear classification head tailored for the EMCellFoundViT model.
    Collapses multi-stage 4D feature maps [B, C, H, W] via Global Average Pooling (GAP),
    concatenates them, and projects them to the target class space using a Linear Layer.
    """
    def __init__(self, backbone, num_classes, embed_dim=768):
        """
        Args:
            backbone (nn.Module): Instance of the EMCellFoundViT backbone.
            num_classes (int): Number of target classes for classification.
            embed_dim (int): The base embedding dimension of your ViT (default: 768 for ViT-Base).
        """
        super().__init__()
        self.backbone = backbone
        
        # Calculate total dimension based on the number of output stages
        # If features_only=True and out_indices has 4 stages, total_dim = 768 * 4 = 3072
        num_stages = len(getattr(backbone, 'out_indices', [11])) if getattr(backbone, 'features_only', False) else 1
        self.total_features_dim = embed_dim * num_stages
        
        # Standard Linear Classifier Head
        self.fc = nn.Linear(self.total_features_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images with shape [B, 3, H, W].
        Returns:
            logits (torch.Tensor): Unnormalized log probabilities with shape [B, num_classes].
        """
        # 1. Extract multi-stage feature maps from the backbone
        outputs = self.backbone(x)
        
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
            
        pooled_outputs = []
        for out in outputs:
            # Spatial reduction: [B, C, H, W] -> [B, C] via Global Average Pooling
            gap = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
            pooled_outputs.append(gap)
            
        # 2. Concatenate all pooled stage representations into a single feature vector
        fused_features = torch.cat(pooled_outputs, dim=1)  # Shape: [B, total_features_dim]
        
        # 3. Project to class space
        logits = self.fc(fused_features)  # Shape: [B, num_classes]
        
        return logits

    


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from EMCellFoundViT import emcellfound_vit_base 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Initialize your custom ViT model
    backbone = emcellfound_vit_base(
        pretrained=False, 
        img_size=512,
        features_only=True
    )
    
    # 2. Wrap the backbone inside the KNN Classifier Head
    knn_model = EMCellFinerKNNClassifier(backbone, k=5, metric='cosine')
    
    # 3. Create dummy training data (3 classes: 0, 1, 2)
    mock_train_imgs = torch.randn(20, 3, 512, 512)
    mock_train_labels = torch.randint(0, 3, (20,))
    train_loader = DataLoader(TensorDataset(mock_train_imgs, mock_train_labels), batch_size=4)
    
    # 4. Execute the training phase (Fit)
    knn_model.fit(train_loader, device=device)
    
    # 5. Execute evaluation phase (Predict)
    knn_model.to(device)
    mock_test_imgs = torch.randn(2, 3, 512, 512).to(device)
    
    probs = knn_model(mock_test_imgs)
    preds = knn_model.predict(mock_test_imgs)
    
    print("\n[Output] Test Prediction Probabilities:\n", probs)
    print("[Output] Test Predicted Labels:", preds)
    
    
    
    
    
# import torch.optim as optim
# # test the linear classifier head
# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # 1. Initialize your custom ViT model
#     backbone = emcellfound_vit_base(
#         pretrained=False, 
#         img_size=512,
#         features_only=True
#     )
    
#     # 2. Wrap the backbone inside the Linear Classifier Head (e.g., 10 target classes)
#     model = EMCellFinerLinearClassifier(backbone, num_classes=10, embed_dim=768)
#     model.to(device)
    
#     # 3. Configure training protocol (Example: Linear Probing)
#     # Freeze the backbone weights, only optimize the linear head parameters
#     for param in model.backbone.parameters():
#         param.requires_grad = False
        
#     # Verify that only the linear layer parameters require gradients
#     optimizer = optim.AdamW(model.fc.parameters(), lr=1e-3)
#     criterion = nn.CrossEntropyLoss()
    
#     # 4. Dummy forward and backward pass simulation
#     model.train()
#     mock_imgs = torch.randn(2, 3, 512, 512).to(device)
#     mock_labels = torch.randint(0, 10, (2,)).to(device)
    
#     # Forward pass
#     logits = model(mock_imgs)
#     loss = criterion(logits, mock_labels)
    
#     # Backward pass
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     print("[Output] Forward pass successful!")
#     print(f"[Output] Logits shape: {logits.shape}")  # Should be [2, 10]
#     print(f"[Output] Simulated loss value: {loss.item():.4f}")