# Vendored by eswa-experimenter-arch (exp_A4, ESWA revision) on 2026-10-20 from https://github.com/yuqinie98/PatchTST
# at commit 204c21efe0b39603ad6e2ca640ef5896646ab1a9, file PatchTST_self_supervised/src/callback/patch_mask.py (Apache-2.0; licence in vendored/LICENSES/PatchTST_LICENSE).
# MODIFIED: extracted function `create_patch` only (file lines 73-84); Full diff: vendored/VENDORED.md.

def create_patch(xb, patch_len, stride):
    """
    xb: [bs x seq_len x n_vars]
    """
    seq_len = xb.shape[1]
    num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
    tgt_len = patch_len  + stride*(num_patch-1)
    s_begin = seq_len - tgt_len
        
    xb = xb[:, s_begin:, :]                                                    # xb: [bs x tgt_len x nvars]
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)                 # xb: [bs x num_patch x n_vars x patch_len]
    return xb, num_patch
