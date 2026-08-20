import os
import tempfile
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st

# ==========================================
# 1. ARSITEKTUR POINTNET++ (WAJIB ADA)
# ==========================================
def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = self.sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = self.sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        
        new_points = new_points.permute(0, 3, 2, 1) 
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

    def sample_and_group(self, npoint, radius, nsample, xyz, points):
        B, N, C = xyz.shape
        S = npoint
        fps_idx = farthest_point_sample(xyz, npoint)
        new_xyz = index_points(xyz, fps_idx)
        idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, 3)
        
        if points is not None:
            grouped_points = index_points(points, idx)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz_norm
        return new_xyz, new_points

    def sample_and_group_all(self, xyz, points):
        device = xyz.device
        B, N, C = xyz.shape
        new_xyz = torch.zeros(B, 1, C).to(device)
        grouped_xyz = xyz.view(B, 1, N, C)
        if points is not None:
            new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
        else:
            new_points = grouped_xyz
        return new_xyz, new_points

class PointNet2_MultiLabel(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2_MultiLabel, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128+3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256+3, mlp=[256, 512, 1024], group_all=True)
        
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x

# ==========================================
# 2. FUNGSI LOAD MODEL & PREDIKSI (CACHED)
# ==========================================
@st.cache_resource
def load_model():
    model_path = "model_final_pointnet2_100persen.pth" # Pastikan nama file ini benar!
    
    if not os.path.exists(model_path):
        return None, None
        
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    classes = checkpoint.get('classes', ['blankspot', 'dinding tambahan', 'dinding terpotong', 'kanopi', 'valid'])
    
    # Inisialisasi model PointNet++
    model = PointNet2_MultiLabel(num_classes=len(classes))
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, classes

def process_and_predict(file_path, model, classes):
    # Load pcd
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    
    if len(points) == 0:
        raise ValueError("File PCD kosong atau tidak terbaca.")
        
    # Pastikan jumlah titik pas 2048 (WAJIB PAKAI FPS BIAR SAMA KAYAK TRAINING!)
    if len(points) > 2048:
        # Ubah ke tensor sebentar untuk masuk mesin FPS
        pts_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
        # Ambil 2048 titik terjauh secara merata
        fps_idx = farthest_point_sample(pts_tensor, 2048)[0]
        points = points[fps_idx.numpy()]
    elif len(points) < 2048:
        idx = np.random.choice(len(points), 2048, replace=True)
        points = points[idx]

    # Normalisasi (Sama persis dengan Dataset class)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    m = np.max(np.sqrt(np.sum(points**2, axis=1)))
    if m > 0:
        points = points / m
        
    # Ubah ke tensor (Batch=1, Channel=3, N=2048)
    inputs = torch.tensor(points, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
    
    # Prediksi
    with torch.no_grad():
        outputs = model(inputs)
        probs = torch.sigmoid(outputs).squeeze().numpy()
        
    return {class_name: float(prob) * 100 for class_name, prob in zip(classes, probs)}

# ==========================================
# 3. ANTARMUKA WEB STREAMLIT
# ==========================================
st.title("🔍 Deteksi 🛻 (PointNet++)")
st.write("Silakan upload file point cloud (.pcd) merge")

password = st.sidebar.text_input("Masukkan Password", type="password")
if password != "abc123":
    st.warning("Silakan masukkan password di menu sebelah kiri")
    st.stop()

model, classes = load_model()

if model is None:
    st.error("🚨 File model 'best_model_pointnet2.pth' tidak ditemukan! Pastikan sudah ditaruh di folder yang sama.")
else:
    uploaded_file = st.file_uploader("Upload File .PCD", type=['pcd'])
    
    if uploaded_file is not None:
        st.info("Memproses point cloud... Mohon tunggu sebentar.")
        
        # Simpan file sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pcd") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
            
        try:
            # Lakukan prediksi
            hasil = process_and_predict(tmp_path, model, classes)
            
            st.subheader("📊 Hasil Prediksi (Skor Keyakinan):")
            
            # Urutkan dari persentase terbesar
            hasil_urut = sorted(hasil.items(), key=lambda x: x[1], reverse=True)
            
            # Tampilkan progress bar untuk masing-masing kelas
            for nama, skor in hasil_urut:
                st.write(f"**{nama.upper()}**: {skor:.1f}%")
                st.progress(int(skor))
                
            # --- LOGIKA KESIMPULAN (Sama seperti sebelumnya) ---
            st.markdown("---")
            top_1_nama = hasil_urut[0][0]
            cacat_list = [(nama, skor) for nama, skor in hasil_urut if "valid" not in nama.lower()]
            
            if "valid" in top_1_nama.lower():
                st.success("✅ KESIMPULAN: Wadah ini **NORMAL / VALID** dan siap digunakan!")
            else:
                st.error("⚠️ KESIMPULAN: Wadah ini **CACAT / INVALID**!")
                
                # Cek cacat di atas 50%
                ada_cacat_diatas_50 = any(skor >= 50.0 for _, skor in cacat_list)
                
                if not ada_cacat_diatas_50:
                    alasan_cacat_40 = [(nama, skor) for nama, skor in cacat_list if skor >= 40.0]
                    if alasan_cacat_40:
                        st.write("Indikasi Cacat Tertinggi:")
                        for nama_cacat, skor_cacat in alasan_cacat_40:
                            st.write(f"🚨 **{nama_cacat}** ({skor_cacat:.1f}%)")
                            
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
