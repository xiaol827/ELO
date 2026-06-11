opt_map = {
    "muadam": "MuAdam",
    "muon": "Muon",
    "adamw": "AdamW",
}
import re

def parse_compression(comp):
    """
    Parses a single compression component and returns (type, detail, extra_info_dict)
    """
    if comp.startswith("topk-"):
        pct = float(comp.split('-')[-1])
        return "Top-k", f"{pct*100:.0f}%", {}
    elif comp.startswith("random-"):
        pct = float(comp.split('-')[-1])
        return "Random", f"{pct*100:.0f}%", {}
    elif comp.startswith("quant-"):
        bits = int(re.search(r"(\d+)bit", comp).group(1))
        return "Quant", f"{bits}-Bit", {}
    elif comp.startswith("demo-"):
        # DCT compression: demo-<blocksize>-<topk>
        m = re.match(r"demo-(\d+)-([\d\.]+)", comp)
        if m:
            block_size = int(m.group(1))
            topk = float(m.group(2))
            return "DCT", f"s={block_size}, Top-k {topk*100:.0f}%", {"block_size": block_size, "topk": topk}
        else:
            return "DCT", "Unknown", {}
    elif comp == "none":
        return "None", "0", {}
    else:
        return comp, "Unknown", {}

def parse_suffix(suffix):
    """
    Parses the suffix to extract K, H, optimizer, compression method,
    and compression detail (percentage or bits if applicable).
    Returns a clean dictionary for plotting or analysis.
    """
    if suffix.startswith('_'):
        suffix = suffix[1:]

    # Check for Data Parallel format first
    if suffix.endswith("_AR_c-none"):
        optimizer = suffix.split("_")[0]
        return {
            "K": 1* 8,
            "H": 1,
            "optimizer": optimizer,
            "compression_types": [],
            "compression_details": [],
            "compression": "none",
            "compression_detail": '0',
            "label": f"{opt_map[optimizer]} | Data Parallel",
            "ec_match":False
        }

    # Check for EC format with or without compression
    ec_pattern = r"K(\d+)_H(\d+)_([a-zA-Z0-9]+)_ec(_c-([\w\.\-\_]+))?"
    ec_match = re.search(ec_pattern, suffix)

    if ec_match:
        k = int(ec_match.group(1))* 8
        h = int(ec_match.group(2))
        optimizer = ec_match.group(3)
        compression = ec_match.group(5) if ec_match.group(5) else "none"

        compression_parts = compression.split('_')
        compression_details = []
        compression_types = []
        extra_info = {}

        for comp in compression_parts:
            ctype, cdetail, cextra = parse_compression(comp)
            compression_types.append(ctype)
            compression_details.append(cdetail)
            extra_info.update(cextra)

        compression_str = " + ".join(f"{ctype} {cdetail}" for ctype, cdetail in zip(compression_types, compression_details))
        if not compression_str or (len(compression_types) == 1 and compression_types[0] == "None"):
            compression_str = "No Compression"

        label = f"{opt_map[optimizer]} | W={k}, H={h}, C={compression_str} EC"

        return {
            "K": k,
            "H": h,
            "optimizer": optimizer,
            "compression": compression,
            "compression_types": compression_types,
            "compression_details": compression_details,
            "compression_detail": compression_details[0] if compression_details else 0,
            "label": label,
            "ec_match": ec_match,
            **extra_info
        }

    # Check for regular format with compression
    pattern = r"K(\d+)_H(\d+)_([a-zA-Z0-9]+)_c-([\w\.\-\_]+)"
    match = re.search(pattern, suffix)

    if match:
        k = int(match.group(1)) * 8
        h = int(match.group(2))
        optimizer = match.group(3)
        compression = match.group(4)

        compression_parts = compression.split('_')
        compression_details = []
        compression_types = []
        extra_info = {}

        for comp in compression_parts:
            ctype, cdetail, cextra = parse_compression(comp)
            compression_types.append(ctype)
            compression_details.append(cdetail)
            extra_info.update(cextra)

        compression_str = " + ".join(f"{ctype} {cdetail}" for ctype, cdetail in zip(compression_types, compression_details))
        if not compression_str or (len(compression_types) == 1 and compression_types[0] == "None"):
            compression_str = "No Compression"
        
        label = f"{opt_map[optimizer]} | W={k}, H={h}, C={compression_str}"

        return {
            "K": k,
            "H": h,
            "optimizer": optimizer,
            "compression": compression,
            "compression_types": compression_types,
            "compression_details": compression_details,
            "compression_detail": compression_details[0] if compression_details else 0,
            "label": label,
            "ec_match": None,
            **extra_info
        }

    # Check for DCT compression in new format (e.g. c-demo-64-.05)
    dct_pattern = r"c-demo-(\d+)-([\d\.]+)"
    dct_match = re.search(dct_pattern, suffix)
    if dct_match:
        block_size = int(dct_match.group(1))
        topk = float(dct_match.group(2))
        # Try to extract K, H, optimizer as well
        k_match = re.search(r"K(\d+)", suffix)
        h_match = re.search(r"H(\d+)", suffix)
        opt_match = re.search(r"(muadam|muon|adamw)", suffix)
        k = int(k_match.group(1)) * 8 if k_match else None
        h = int(h_match.group(1)) if h_match else None
        optimizer = opt_match.group(1) if opt_match else "unknown"
        label = f"{opt_map.get(optimizer, optimizer)} | W={k}, H={h}, C=DCT s={block_size}, Top-k {topk*100:.0f}%"
        return {
            "K": k,
            "H": h,
            "optimizer": optimizer,
            "compression": f"demo-{block_size}-{topk}",
            "compression_detail": f"s={block_size}, Top-k {topk*100:.0f}%",
            "label": label,
            "block_size": block_size,
            "topk": topk,
            "ec_match":None
        }

    raise ValueError(f"Suffix format not recognized: {suffix}")