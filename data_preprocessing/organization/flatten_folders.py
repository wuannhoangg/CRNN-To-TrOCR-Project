#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gom (move/copy) ảnh từ các thư mục con dạng đánh số (1..N) về một thư mục đích duy nhất
và đồng thời gộp nhãn từ label.json/labels.json theo tên file cuối cùng.

Chức năng:
- Hỗ trợ 2 chế độ:
  - move (mặc định): di chuyển file ảnh sang thư mục đích.
  - copy: sao chép file ảnh sang thư mục đích.
- Tránh trùng tên file khi gom:
  - Nếu tên ảnh đã tồn tại trong thư mục đích hoặc đã được dùng trong batch,
    script sẽ tự tạo tên mới bằng hậu tố: `__f<folder>__k<idx>`.
- Gộp nhãn:
  - Key nhãn đầu ra là tên file cuối cùng sau khi copy/move (basename).
  - Nếu trùng key nhưng text khác nhau, xử lý theo `--prefer`:
    - last: ghi đè bằng nhãn ở lần sau.
    - first: giữ nhãn cũ.
    - error: dừng và báo lỗi.
- Bỏ qua việc copy/move các file nhãn; chỉ đọc nhãn để gộp.
- Có `--dry_run` để xem trước mà không thực thi thay đổi.

Ví dụ:
    py -3.13 consolidate_to_single_folder.py ^
      --root "C:\\...\\TrOCR" ^
      --dest "C:\\...\\TrOCR\\all_images" ^
      --first 1 --last 249 ^
      --labels_name "label.json" ^
      --prefer last ^
      --mode move ^
      --use_basename
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, Tuple, List, Set

# Danh sách extension ảnh hợp lệ để xử lý.
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}


def ensure_dir(p: Path):
    """Tạo thư mục nếu chưa tồn tại (tạo cả cây thư mục cha)."""
    p.mkdir(parents=True, exist_ok=True)


def load_json(p: Path) -> Dict[str, str]:
    """Đọc file JSON nhãn và trả về dict {filename: text}."""
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p: Path, data: Dict[str, str]):
    """Ghi dict nhãn ra JSON (UTF-8, giữ Unicode, indent để dễ kiểm tra)."""
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def unique_name(dest_dir: Path, base_name: str, used: Set[str], folder_id: int, k_hint: int = 1) -> str:
    """Sinh tên file không bị trùng trong thư mục đích.

    Cách hoạt động:
    - Nếu `base_name` chưa xuất hiện trong `used` và file chưa tồn tại trong `dest_dir`,
      dùng trực tiếp.
    - Nếu trùng, thêm hậu tố theo mẫu: `<stem>__f<folder_id>__k<idx><ext>`.
      `idx` tăng dần cho tới khi tìm được tên hợp lệ.

    Args:
        dest_dir (Path): Thư mục đích chứa file.
        base_name (str): Tên file gốc (basename) cần đảm bảo không trùng.
        used (Set[str]): Tập các tên file đã dùng trong batch (để tránh trùng nội bộ).
        folder_id (int): ID folder nguồn (dùng trong hậu tố để truy vết).
        k_hint (int): Gợi ý chỉ số bắt đầu (để giảm số lần thử).

    Returns:
        str: Tên file duy nhất (không bị trùng).
    """
    stem, ext = os.path.splitext(base_name)
    cand = base_name
    k = k_hint
    while cand in used or (dest_dir / cand).exists():
        cand = f"{stem}__f{folder_id}__k{k}{ext.lower()}"
        k += 1
    return cand


def main():
    """Điểm vào chính: parse args, gom ảnh và gộp nhãn."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Thư mục gốc chứa subfolders 1..N")
    ap.add_argument("--dest", required=True, help="Thư mục đích gom tất cả ảnh")
    ap.add_argument("--first", type=int, default=1, help="Folder bắt đầu (mặc định 1)")
    ap.add_argument("--last",  type=int, required=True, help="Folder kết thúc (vd 249)")
    ap.add_argument(
        "--labels_name",
        default="label.json",
        help="Tên file nhãn trong mỗi folder (mặc định: label.json)",
    )
    ap.add_argument(
        "--prefer",
        choices=["last", "first", "error"],
        default="last",
        help="Khi trùng key nhãn khác text: last ghi đè, first giữ cũ, error dừng lại.",
    )
    ap.add_argument(
        "--mode",
        choices=["move", "copy"],
        default="move",
        help="Di chuyển hay sao chép (mặc định: move)",
    )
    ap.add_argument("--use_basename", action="store_true", help="Chuẩn hóa key nhãn về basename (khuyên dùng)")
    ap.add_argument("--glob", default="*", help="Glob lọc ảnh trong mỗi folder (vd '*.jpg', mặc định: '*')")
    ap.add_argument("--dry_run", action="store_true", help="Chỉ xem trước, không thực thi")
    args = ap.parse_args()

    root = Path(args.root)
    dest = Path(args.dest)
    if not root.is_dir():
        raise SystemExit(f"[ERR] Không thấy thư mục gốc: {root}")

    ensure_dir(dest)

    # Tập tên file hiện có trong thư mục đích (để tránh đụng khi sinh tên mới).
    used_names: Set[str] = {p.name for p in dest.iterdir() if p.is_file()}

    merged_labels: Dict[str, str] = {}
    total_moved = 0
    total_labels_read = 0
    collisions = 0
    collisions_diff = 0

    # Duyệt các folder con theo dải chỉ số.
    for i in range(args.first, args.last + 1):
        sub = root / str(i)
        if not sub.is_dir():
            print(f"[WARN] Bỏ qua {sub} (không phải thư mục).")
            continue

        # Đọc labels của folder nếu có; nếu lỗi đọc thì bỏ qua labels nhưng vẫn gom ảnh.
        lbl_path = sub / args.labels_name
        labels = {}
        if lbl_path.exists():
            try:
                labels = load_json(lbl_path)
            except Exception as e:
                print(f"[WARN] Không đọc được {lbl_path}: {e}")
                labels = {}

        # Lấy danh sách ảnh hợp lệ trong folder theo glob.
        files = sorted([p for p in sub.glob(args.glob) if p.is_file() and p.suffix.lower() in IMG_EXTS])
        if not files:
            print(f"[INFO] {sub}: không có ảnh (glob='{args.glob}').")
            continue

        print(f"[INFO] {sub}: {len(files)} file ảnh.")
        preview = []
        k_hint = 1

        # Lập kế hoạch và thực hiện copy/move từng ảnh.
        for src in files:
            orig_name = src.name
            new_name = unique_name(dest, orig_name, used_names, folder_id=i, k_hint=k_hint)
            k_hint += 1
            dst = dest / new_name

            preview.append((orig_name, new_name))

            if args.dry_run:
                continue

            # Thực thi gom ảnh theo mode đã chọn.
            if args.mode == "move":
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))

            used_names.add(new_name)
            total_moved += 1

            # Cập nhật nhãn hợp nhất nếu folder có labels.
            if labels:
                # Chuẩn hóa key của file trong labels theo basename nếu yêu cầu.
                key = os.path.basename(orig_name) if args.use_basename else orig_name
                if key in labels:
                    text = labels[key]
                    # Nếu nhãn trùng key theo tên mới và khác text, xử lý theo strategy.
                    if new_name in merged_labels and merged_labels[new_name] != text:
                        collisions += 1
                        collisions_diff += 1
                        if args.prefer == "error":
                            raise SystemExit(
                                f"[ERR] Trùng nhãn khác text cho {new_name}:\n"
                                f"  cũ={merged_labels[new_name]}\n"
                                f"  mới={text}\n"
                                f"  nguồn={lbl_path}"
                            )
                        elif args.prefer == "last":
                            merged_labels[new_name] = text
                        elif args.prefer == "first":
                            pass
                    else:
                        merged_labels[new_name] = text

        # Thống kê số nhãn đã đọc (thô) để đánh giá mức độ đầy đủ của label theo folder.
        if labels:
            total_labels_read += len(labels)

        # In ví dụ đổi tên để kiểm soát nhanh.
        print("   Ví dụ đổi tên:")
        for a, b in preview[:5]:
            print(f"     {a} -> {b}")
        if len(preview) > 5:
            print("     ...")

    # Ghi labels hợp nhất vào thư mục đích (chỉ khi chạy thật).
    out_json = dest / "labels_merged.json"
    if not args.dry_run:
        save_json(out_json, merged_labels)
        print(f"[OK] Ghi nhãn hợp nhất -> {out_json}")

    # Tổng kết batch để phục vụ audit và báo cáo.
    print("-" * 60)
    print(f"[DONE] {'(DRY-RUN) ' if args.dry_run else ''}Đã {'dự kiến ' if args.dry_run else ''}{args.mode} {total_moved} ảnh.")
    print(f"[INFO] Tổng nhãn đã đọc (thô): {total_labels_read}, nhãn hợp nhất: {len(merged_labels)}")
    print(f"[INFO] Va chạm key nhãn khác text: {collisions_diff} (prefer='{args.prefer}')")
    print(f"[INFO] Thư mục đích: {dest}")


if __name__ == "__main__":
    main()
