#!/usr/bin/env python3
"""
AES + Chaotic-map steganography (embedding in LSBs)
- AES-GCM encrypts the file (nonce || ciphertext-with-tag)
- Chaotic keystream XORs (scrambles) the AES output
- Chaotic permutation orders pixel-channel positions (permutes indices)
- A 4-byte big-endian header stores payload length (bytes)
- Embeds header+scrambled-payload into LSBs of chosen pixel-channels
"""

import os
import struct
import math
import hashlib
from typing import List
from PIL import Image
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# --------------------------- CONFIG ---------------------------
# Set these to your values
AES_KEY_HEX = '112233445566778899aabbccddeeff00'   # original hex key you had
CHAOS_KEY = "team5"                                # secret for chaotic map (choose strong passphrase)

# AESGCM expects 16/24/32 byte key. We'll derive 32 bytes from your hex key
AES_KEY = hashlib.sha256(bytes.fromhex(AES_KEY_HEX)).digest()  # 32 bytes (AES-256)

# --------------------------- UTIL ------------------------------
def compute_mse_psnr(img_path_a: str, img_path_b: str):
    a = np.array(Image.open(img_path_a).convert("RGB"), dtype=np.float64)
    b = np.array(Image.open(img_path_b).convert("RGB"), dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Image shapes differ: {a.shape} vs {b.shape}")
    mse = float(np.mean((a - b) ** 2))
    psnr = float('inf') if mse == 0.0 else 10.0 * math.log10((255.0 ** 2) / mse)
    return mse, psnr

def bytes_to_bits(data: bytes) -> List[int]:
    out = []
    for byte in data:
        for i in range(7, -1, -1):
            out.append((byte >> i) & 1)
    return out

def bits_to_bytes(bits: List[int]) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError("Number of bits not multiple of 8")
    ba = bytearray()
    for i in range(0, len(bits), 8):
        b = 0
        for bit in bits[i:i+8]:
            b = (b << 1) | (bit & 1)
        ba.append(b)
    return bytes(ba)

# ----------------------- Chaotic map helpers --------------------
def derive_chaos_params(key_string: str):
    """
    Derive x0 in (0,1) and r in [3.9, 4.0) deterministically from key_string.
    """
    h = hashlib.sha256(key_string.encode()).digest()
    big = int.from_bytes(h, 'big')
    x0 = (big % (10**8)) / (10**8 + 1.0)  # in (0,1)
    # r in [3.9, 4.0)
    r_fraction = (big >> 64) & 0xFFFF
    r = 3.9 + (r_fraction / 65536.0) * 0.1
    return r, x0

def logistic_map_sequence(r: float, x0: float, count: int, burn: int = 200):
    """
    Generate 'count' chaotic values in (0,1) using logistic map.
    Burn-in to reduce transient behavior.
    """
    x = x0
    for _ in range(burn):
        x = r * x * (1.0 - x)
    seq = []
    for _ in range(count):
        x = r * x * (1.0 - x)
        # avoid exact 0 or 1
        if x <= 0.0:
            x = 1e-15
        elif x >= 1.0:
            x = 1 - 1e-15
        seq.append(x)
    return seq

def chaotic_permutation_indices(width: int, height: int, chaos_key: str) -> List[int]:
    """
    Create a deterministic permutation of all pixel-channel positions using chaotic ordering.
    Returns a list of length (width*height*3) containing every index once.
    """
    total = width * height * 3
    r, x0 = derive_chaos_params(chaos_key + "_perm")
    seq = logistic_map_sequence(r, x0, total)
    indices = list(range(total))
    # Pair and sort by chaotic value (descending) -> deterministic permutation
    paired = list(zip(seq, indices))
    paired.sort(key=lambda x: x[0], reverse=True)
    perm = [idx for _, idx in paired]
    return perm

def chaotic_keystream_bytes(length: int, chaos_key: str) -> bytes:
    """
    Produce 'length' bytes from chaotic map (0..255) to be used as XOR keystream.
    """
    r, x0 = derive_chaos_params(chaos_key + "_keystream")
    seq = logistic_map_sequence(r, x0, length)
    kb = bytes((int(v * 256) & 0xFF) for v in seq)
    return kb

# ---------------------- Stego embed / extract -------------------
def embed_bytes_into_image(cover_path: str, out_path: str, payload_bytes: bytes, chaos_key: str):
    """
    payload_bytes: bytes to embed (already scrambled)
    Embeds: 4-byte big-endian length header + payload_bytes
    """
    img = Image.open(cover_path).convert("RGB")
    pixels = np.array(img, dtype=np.uint8)
    h, w, _ = pixels.shape
    total_positions = w * h * 3

    header = struct.pack(">I", len(payload_bytes))
    full = header + payload_bytes
    total_bits = len(full) * 8

    if total_bits > total_positions:
        raise ValueError(f"Payload too large: need {total_bits} bits, capacity {total_positions} bits")

    perm = chaotic_permutation_indices(w, h, chaos_key)  # ordering over all positions
    chosen = perm[:total_bits]

    bits = bytes_to_bits(full)
    for i, bit in enumerate(bits):
        idx = chosen[i]
        pixel_index = idx // 3
        x = pixel_index % w
        y = pixel_index // w
        c = idx % 3
        pixels[y, x, c] = (int(pixels[y, x, c]) & 0xFE) | int(bit)

    stego = Image.fromarray(pixels)
    stego.save(out_path, format="PNG")
    return True

def extract_bytes_from_image(stego_path: str, chaos_key: str):
    """
    Extracts header (4 bytes) first by reading the first 32 indices of the chaotic permutation,
    then extracts full payload based on header length.
    Returns the extracted payload bytes (unscrambled).
    """
    img = Image.open(stego_path).convert("RGB")
    pixels = np.array(img, dtype=np.uint8)
    h, w, _ = pixels.shape
    total_positions = w * h * 3

    perm = chaotic_permutation_indices(w, h, chaos_key)
    # read header first (32 bits)
    header_bits = 4 * 8
    header_indices = perm[:header_bits]
    header_bitlist = []
    for idx in header_indices:
        pixel_index = idx // 3
        x = pixel_index % w
        y = pixel_index // w
        c = idx % 3
        header_bitlist.append(int(pixels[y, x, c]) & 1)
    header_bytes = bits_to_bytes(header_bitlist)
    payload_len = struct.unpack(">I", header_bytes)[0]

    total_bits = (4 + payload_len) * 8
    if total_bits > total_positions:
        raise ValueError("Header claims payload larger than image capacity")

    chosen = perm[:total_bits]
    bitlist = []
    for idx in chosen:
        pixel_index = idx // 3
        x = pixel_index % w
        y = pixel_index // w
        c = idx % 3
        bitlist.append(int(pixels[y, x, c]) & 1)

    full_bytes = bits_to_bytes(bitlist)
    extracted_payload = full_bytes[4:]  # strip header
    return extracted_payload

# ------------------------ AES wrappers --------------------------
def aesgcm_encrypt(aes_key: bytes, plaintext: bytes) -> bytes:
    """
    Returns nonce || ciphertext_with_tag
    """
    aesgcm = AESGCM(aes_key)
    nonce = os.urandom(12)  # 96-bit nonce recommended for AESGCM
    ct = aesgcm.encrypt(nonce, plaintext, None)
    return nonce + ct

def aesgcm_decrypt(aes_key: bytes, blob: bytes) -> bytes:
    nonce = blob[:12]
    ct = blob[12:]
    aesgcm = AESGCM(aes_key)
    return aesgcm.decrypt(nonce, ct, None)

# --------------------------- MAIN --------------------------------
if __name__ == "__main__":
    # ---------------- Step 0: prepare cover image ----------------
    Image.open("image.jpg").convert("RGB").save("cover.png")

    # ---------------- Step 1: AES encrypt the secret file ----------
    with open("script.sh", "rb") as f:
        plaintext = f.read()

    encrypted_blob = aesgcm_encrypt(AES_KEY, plaintext)   # nonce + ciphertext|tag

    # ---------------- Step 2: scramble encrypted blob via chaos ----------
    keystream = chaotic_keystream_bytes(len(encrypted_blob), CHAOS_KEY)
    scrambled = bytes(b ^ k for b, k in zip(encrypted_blob, keystream))

    # ---------------- Step 3: embed header+scrambled payload into image ----------
    embed_bytes_into_image("cover.png", "stegoImg.png", scrambled, CHAOS_KEY)
    print("Embedding done -> stegoImg.png")

    # ---------------- Step 4: extract from stego image ----------
    extracted_scrambled = extract_bytes_from_image("stegoImg.png", CHAOS_KEY)

    # ---------------- Step 5: unscramble and AES decrypt -----------
    # regenerate keystream
    ks2 = chaotic_keystream_bytes(len(extracted_scrambled), CHAOS_KEY)
    unscrambled = bytes(b ^ k for b, k in zip(extracted_scrambled, ks2))

    try:
        recovered = aesgcm_decrypt(AES_KEY, unscrambled)
    except Exception as e:
        print("Decryption failed:", e)
        raise

    with open("extracted.sh", "wb") as f:
        f.write(recovered)
    os.chmod("extracted.sh", 0o755)
    print("Recovered file written to extracted.sh")

    # ---------------- Step 6: Evaluate quality -----------------
    mse, psnr = compute_mse_psnr("cover.png", "stegoImg.png")
    print(f"MSE:  {mse:.6f}")
    print(f"PSNR: {psnr:.4f} dB")
