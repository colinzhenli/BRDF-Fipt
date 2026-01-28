import torch

def rusinkiewicz_to_vectors(theta_h, theta_d, phi_d):
    """
    使用 PyTorch 将 Rusinkiewicz 坐标转换为 wi 和 wo 矢量。
    支持批处理 (Batch processing)。
    
    参数:
        theta_h (Tensor): 半矢量与法线夹角，形状为 (...)
        theta_d (Tensor): 入射光与半矢量夹角，形状为 (...)
        phi_d   (Tensor): 入射光绕半矢量的方位角，形状为 (...)
        
    返回:
        wi, wo (Tensor): 形状为 (..., 3) 的单位矢量
    """
    # 1. 计算三角函数
    sh, ch = torch.sin(theta_h), torch.cos(theta_h)
    sd, cd = torch.sin(theta_d), torch.cos(theta_d)
    spd, cpd = torch.sin(phi_d), torch.cos(phi_d)

    # 2. 在半矢量局部空间 (H-frame) 构建 wi 和 wo
    # 此时 H = [0, 0, 1]
    # wi_local = [sin(td)cos(pd), sin(td)sin(pd), cos(td)]
    wi_x_local = sd * cpd
    wi_y_local = sd * spd
    wi_z_local = cd

    # wo 是 wi 绕 H(Z轴) 旋转180度，即 x,y 取反
    wo_x_local = -wi_x_local
    wo_y_local = -wi_y_local
    wo_z_local = wi_z_local

    # 3. 将局部坐标旋转到法线坐标系 (绕 Y 轴旋转 theta_h)
    # 旋转公式:
    # x' = x*cos(th) + z*sin(th)
    # y' = y
    # z' = -x*sin(th) + z*cos(th)
    
    wi_x = wi_x_local * ch + wi_z_local * sh
    wi_y = wi_y_local
    wi_z = -wi_x_local * sh + wi_z_local * ch

    wo_x = wo_x_local * ch + wo_z_local * sh
    wo_y = wo_y_local
    wo_z = -wo_x_local * sh + wo_z_local * ch

    # 4. 拼接成 (..., 3) 形状的张量
    wi = torch.stack([wi_x, wi_y, wi_z], dim=-1)
    wo = torch.stack([wo_x, wo_y, wo_z], dim=-1)

    return wi, wo

def lookup_wiwo(self, wi, wo, material_id):
    """
    Look up BRDF values for given incoming/outgoing direction vectors.
    
    This is a wrapper around lookup() that converts direction vectors to angles.
    
    Args:
        wi: [B, 3] incoming light directions (normalized, pointing toward surface)
        wo: [B, 3] outgoing view directions (normalized, pointing away from surface)
    
    Returns:
        rgb: [B, 3] BRDF RGB values
    """
    # Normalize directions
    wi = F.normalize(wi, dim=-1)
    wo = F.normalize(wo, dim=-1)
    
    # Convert wi to spherical coordinates
    theta_in = torch.acos(torch.clamp(wi[..., 2], -1.0, 1.0))
    phi_in = torch.atan2(wi[..., 1], wi[..., 0])
    
    # Convert wo to spherical coordinates
    theta_out = torch.acos(torch.clamp(wo[..., 2], -1.0, 1.0))
    phi_out = torch.atan2(wo[..., 1], wo[..., 0])
    
    # Call the main lookup function
    return self.lookup(theta_in, phi_in, theta_out, phi_out, material_id)

# --- 测试与验证 ---
if __name__ == "__main__":
    # 创建一批测试数据 (2个样本)
    th = torch.tensor([torch.pi/4, torch.pi/4])      # 0度 和 45度
    td = torch.tensor([torch.pi/4, torch.pi/6]) # 均为 30度
    pd = torch.tensor([torch.pi, 0.0])             # 均为 0度

    wi, wo = rusinkiewicz_to_vectors(th, td, pd)

    print("入射矢量 wi:\n", wi)
    print("出射矢量 wo:\n", wo)

    # 验证半矢量 H 是否正确
    h_calc = torch.nn.functional.normalize(wi + wo, dim=-1)
    print("计算得到的半矢量 H:\n", h_calc)
    
    # 当 th=0 时，H 应为 [0, 0, 1]
    # 当 th=pi/4 时，H 应为 [sin(pi/4), 0, cos(pi/4)] ≈ [0.707, 0, 0.707]