import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytorch_msssim
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from skimage.metrics import structural_similarity as ssim_skimage

# Cấu hình chuỗi watermark
MAX_WATERMARK_LEN = 1024
VOCAB_SIZE = 128  # ASCII
class WatermarkEmbeddingModule(nn.Module):
    def __init__(self, image_size=(256, 256)):
        super(WatermarkEmbeddingModule, self).__init__()
        self.image_size = image_size
        self.embed = nn.Embedding(VOCAB_SIZE, 8)  # 8-dim embedding cho mỗi ký tự
        self.fc = nn.Sequential(
            nn.Linear(MAX_WATERMARK_LEN * 8, image_size[0] * image_size[1]),
            nn.Tanh()
        )

    def forward(self, original_image, watermark_seq):
        B = original_image.shape[0]
        emb = self.embed(watermark_seq)  # (B, MAX_WATERMARK_LEN, 8)
        emb = emb.view(B, -1)  # (B, MAX_WATERMARK_LEN*8)
        watermark_map = self.fc(emb)  # (B, H*W)
        watermark_map = watermark_map.view(B, 1, *self.image_size)  # (B,1,H,W)
        if original_image.shape[2:] != self.image_size:
            watermark_map = nn.functional.interpolate(watermark_map, size=original_image.shape[2:], mode='bilinear', align_corners=False)
        if original_image.shape[1] == 3:
            watermark_map = watermark_map.repeat(1, 3, 1, 1)
        embedding_strength = 0.2
        x = original_image + embedding_strength * watermark_map
        x = torch.clamp(x, -1, 1)
        return x

class WatermarkExtractionModule(nn.Module):
    """
    Module trích xuất watermark đối xứng với module nhúng.
    Nhiệm vụ: tái tạo lại watermark từ ảnh đã bị tấn công, sử dụng đặc trưng học được để đảm bảo khả năng trích xuất chính xác.
    """
    def __init__(self, image_size=(256, 256)):
        super(WatermarkExtractionModule, self).__init__()
        self.image_size = image_size
        # Sử dụng cấu trúc CNN đối xứng với phần nhúng
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten()
        )
        self.fc = nn.Linear(32 * 8 * 8, MAX_WATERMARK_LEN * VOCAB_SIZE)

    def forward(self, x):
        # x: (B,3,H,W)
        feat = self.cnn(x)  # (B, 32*8*8)
        logits = self.fc(feat)  # (B, MAX_WATERMARK_LEN*VOCAB_SIZE)
        logits = logits.view(-1, MAX_WATERMARK_LEN, VOCAB_SIZE)  # (B, MAX_WATERMARK_LEN, VOCAB_SIZE)
        return logits  # dùng CrossEntropyLoss

class SACAWNLoss(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=2.0, lambda3=1.5):
        super(SACAWNLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.ce_loss = nn.CrossEntropyLoss()  # cho chuỗi ký tự

    def calculate_imperceptibility_loss(self, original_image, watermarked_image):
        return 1 - pytorch_msssim.ssim(original_image, watermarked_image, data_range=2.0)

    def calculate_robustness_loss(self, original_seq, extracted_logits):
        # original_seq: (B, MAX_WATERMARK_LEN), extracted_logits: (B, MAX_WATERMARK_LEN, VOCAB_SIZE)
        B, L = original_seq.shape
        loss = self.ce_loss(extracted_logits.view(B*L, VOCAB_SIZE), original_seq.view(-1))
        return loss

    def calculate_extraction_loss(self, original_seq, extracted_logits):
        return self.calculate_robustness_loss(original_seq, extracted_logits)

    def forward(self, original_image, watermarked_image, original_seq, extracted_logits):
        limperceptibility = self.calculate_imperceptibility_loss(original_image, watermarked_image)
        lrobustness = self.calculate_robustness_loss(original_seq, extracted_logits)
        lextraction = self.calculate_extraction_loss(original_seq, extracted_logits)
        total_loss = self.lambda1 * limperceptibility + \
                     self.lambda2 * lrobustness + \
                     self.lambda3 * lextraction
        return total_loss, limperceptibility, lrobustness, lextraction

class SACAWN(nn.Module):
    def __init__(self, image_size=(256, 256)):
        super(SACAWN, self).__init__()
        self.embedder = WatermarkEmbeddingModule(image_size)
        self.attacker = AttackSimulationLayer()
        self.extractor = WatermarkExtractionModule(image_size)

    def forward(self, original_image, watermark_seq):
        watermarked_image = self.embedder(original_image, watermark_seq)
        attacked_image = self.attacker(watermarked_image)
        extracted_logits = self.extractor(attacked_image)
        return watermarked_image, attacked_image, extracted_logits

# --- Training/Evaluation code ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sacawn_model = SACAWN(image_size=(256, 256)).to(device)
criterion = SACAWNLoss(lambda1=1.0, lambda2=2.0, lambda3=1.5)
optimizer = optim.Adam(sacawn_model.parameters(), lr=0.0001)

num_epochs = 100

for epoch in range(num_epochs):
    sacawn_model.train()
    total_batch_loss = 0
    # Tạo dữ liệu giả định
    original_images = torch.randn(16, 3, 256, 256).to(device)
    # Chuỗi watermark ngẫu nhiên (ASCII)
    watermarks = torch.randint(0, VOCAB_SIZE, (16, MAX_WATERMARK_LEN), dtype=torch.long).to(device)

    optimizer.zero_grad()
    watermarked_images, attacked_images, extracted_logits = sacawn_model(original_images, watermarks)
    loss_total, loss_imperceptibility, loss_robustness, loss_extraction = criterion(
        original_images, watermarked_images, watermarks, extracted_logits
    )
    loss_total.backward()
    optimizer.step()
    total_batch_loss += loss_total.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_batch_loss:.4f}, "
          f"L_imp: {loss_imperceptibility.item():.4f}, "
          f"L_rob: {loss_robustness.item():.4f}, "
          f"L_ext: {loss_extraction.item():.4f}")

    # Evaluation phase
    sacawn_model.eval()
    with torch.no_grad():
        original_test_images = torch.randn(4, 3, 256, 256).to(device)
        test_watermarks = torch.randint(0, VOCAB_SIZE, (4, MAX_WATERMARK_LEN), dtype=torch.long).to(device)
        test_watermarked_images, test_attacked_images, test_extracted_logits = sacawn_model(original_test_images, test_watermarks)

        # PSNR/SSIM
        test_original_np = ((original_test_images.cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
        test_watermarked_np = ((test_watermarked_images.cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
        avg_psnr, avg_ssim, avg_ber = 0, 0, 0
        for i in range(test_original_np.shape[0]):
            psnr = psnr_skimage(test_original_np[i].transpose(1, 2, 0), test_watermarked_np[i].transpose(1, 2, 0), data_range=255)
            ssim = ssim_skimage(test_original_np[i].transpose(1, 2, 0), test_watermarked_np[i].transpose(1, 2, 0), channel_axis=2, data_range=255)
            # BER cho chuỗi ký tự
            pred_seq = test_extracted_logits[i].argmax(dim=-1)  # (MAX_WATERMARK_LEN,)
            ber = (pred_seq != test_watermarks[i]).float().mean().item()
            avg_psnr += psnr
            avg_ssim += ssim
            avg_ber += ber
        avg_psnr /= test_original_np.shape[0]
        avg_ssim /= test_original_np.shape[0]
        avg_ber /= test_original_np.shape[0]
        print(f"--- Evaluation --- PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}, BER: {avg_ber:.4f}")

torch.save(sacawn_model.state_dict(), 'sacawn_model.pth')
