#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Đổi tên ảnh hàng loạt trong nhiều thư mục con dạng đánh số (ví dụ: 1..249) và cập nhật labels JSON tương ứng.

Chức năng:
- Mỗi thư mục con chứa ảnh và file nhãn (ưu tiên `labels.json`, fallback `label.json` hoặc `--labels_name` nếu chỉ định).
- Ảnh trong folder N được sắp xếp theo tên và đổi tên lần lượt theo mẫu:
    {prefix}{counter}{suffix}{ext}
  với `counter` tăng liên tục xuyên suốt toàn bộ các folder bắt đầu từ `--start`.
- Cập nhật labels để trỏ sang tên ảnh mới, đồng thời backup file labels trước khi ghi.
- Lưu mapping đổi tên trong từng folder: `<folder>/rename_map.csv`.
- Hỗ trợ `--dry_run` để xem trước kế hoạch đổi tên mà không thực thi.

Ví dụ:
    py -3.13 batch_rename_folders.py ^
      --root "C:\\path\\to\\TrOCR" ^
      --first 1 --last 249 ^
      --start 2000 ^
      --suffix "_samples" ^
      --glob "*"

Lưu ý:
- Script đổi tên tại chỗ (in-place), không di chuyển file.
- Có thể resume bằng cách điều chỉnh `--first` và `--start` tương ứng với phần đã xử lý.
"""

import os
import re
import csv
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path

# Tập extension ảnh được coi là hợp lệ để đổi tên.
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}


def find_labels_file(folder: Path, preferred: str = None) -> Path | None:
    """Tìm file labels trong folder theo thứ tự ưu tiên.

    Thứ tự dò:
    1) `preferred` (nếu có truyền từ CLI).
    2) `labels.json`
    3) `label.json`

    Args:
        folder (Path): Thư mục cần dò.
        preferred (str | None): Tên file labels ưu tiên (ví dụ 'labels.json').

    Returns:
        Path | None: Đường dẫn file labels tìm được, hoặc None nếu không có.
    """
    cands = []
    if preferred:
        cands.append(preferred)
    cands += ["labels.json", "label.json"]
    for name in cands:
        p = folder / name
        if p.exists() and p.is_file():
            return p
    return None


def backup_file(p: Path) -> Path:
    """Tạo bản backup của file labels bằng cách thêm timestamp vào hậu tố.

    Args:
        p (Path): Đường dẫn file cần backup.

    Returns:
        Path: Đường dẫn file backup đã tạo.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = p.with_suffix(p.suffix + f".{ts}.backup")
    shutil.copy2(p, backup)
    return backup


def load_labels(p: Path) -> dict:
    """Đọc labels JSON thành dict."""
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_labels(p: Path, labels: dict):
    """Ghi labels dict ra JSON với UTF-8 và indent để dễ kiểm tra."""
    with open(p, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


def gather_images(folder: Path, glob_pat: str) -> list[Path]:
    """Thu thập danh sách ảnh trong folder theo glob, đã lọc theo IMG_EXTS và sắp xếp.

    Args:
        folder (Path): Thư mục cần quét.
        glob_pat (str): Pattern glob lọc file (ví dụ '*.jpg', 'addr_*', '*').

    Returns:
        list[Path]: Danh sách đường dẫn file ảnh hợp lệ.
    """
    items = sorted(folder.glob(glob_pat))
    return [p for p in items if p.is_file() and p.suffix.lower() in IMG_EXTS]


def plan_new_names(
    files: list[Path],
    start_num: int,
    prefix: str,
    suffix: str,
) -> list[tuple[Path, Path, str, str]]:
    """Lập kế hoạch đổi tên cho một danh sách file.

    Kết quả trả về theo dạng:
        [(old_path, new_path, old_name, new_name), ...]

    Args:
        files (list[Path]): Danh sách file ảnh đã được sắp xếp.
        start_num (int): Số bắt đầu cho folder hiện tại.
        prefix (str): Tiền tố cho tên mới.
        suffix (str): Hậu tố cho tên mới.

    Returns:
        list[tuple]: Danh sách kế hoạch đổi tên.
    """
    plan = []
    for idx, old in enumerate(files):
        num = start_num + idx
        new_name = f"{prefix}{num}{suffix}{old.suffix.lower()}"
        new_path = old.parent / new_name
        plan.append((old, new_path, old.name, new_name))
    return plan


def folder_rename_and_update(
    folder: Path,
    start_num: int,
    prefix: str,
    suffix: str,
    glob_pat: str,
    preferred_labels_name: str | None,
    dry_run: bool,
) -> tuple[int, int]:
    """Xử lý 1 folder: đổi tên ảnh và cập nhật labels tương ứng.

    Quy trình:
    1) Tìm và (nếu có) đọc labels JSON.
    2) Gom danh sách ảnh theo glob + extension hợp lệ.
    3) Lập kế hoạch đổi tên theo thứ tự tên file.
    4) Kiểm tra xung đột tên đích để tránh ghi đè không chủ ý.
    5) (Nếu không dry-run) backup labels, đổi tên file, cập nhật labels, ghi mapping CSV.

    Args:
        folder (Path): Thư mục cần xử lý.
        start_num (int): Số bắt đầu cho folder hiện tại.
        prefix (str): Tiền tố tên mới.
        suffix (str): Hậu tố tên mới.
        glob_pat (str): Glob pattern để lọc ảnh.
        preferred_labels_name (str | None): Tên labels ưu tiên, nếu None sẽ tự dò.
        dry_run (bool): True để chỉ preview, không đổi tên và không ghi file.

    Returns:
        tuple[int, int]:
            - count_renamed: số ảnh dự kiến/đã xử lý.
            - last_number_used: số cuối đã dùng (start_num + count - 1),
              hoặc (start_num - 1) nếu folder không có ảnh.
    """
    # 1) Tìm file labels và nạp dữ liệu nếu có.
    labels_path = find_labels_file(folder, preferred=preferred_labels_name)
    labels = {}
    if labels_path:
        try:
            labels = load_labels(labels_path)
        except Exception as e:
            print(f"[WARN] Không đọc được labels ở {labels_path}: {e}. Sẽ chỉ đổi tên ảnh.")
            labels_path = None

    # 2) Thu thập danh sách ảnh theo glob.
    files = gather_images(folder, glob_pat)
    if not files:
        print(f"[INFO] {folder}: không tìm thấy ảnh theo glob '{glob_pat}' -> bỏ qua.")
        return 0, start_num - 1

    # 3) Lập kế hoạch đổi tên theo counter tăng dần.
    plan = plan_new_names(files, start_num, prefix, suffix)

    # 4) Kiểm tra xung đột: tên đích tồn tại và không trùng chính file nguồn.
    conflicts = []
    for old, new, _, _ in plan:
        if new.exists() and new.resolve() != old.resolve():
            conflicts.append(str(new))
    if conflicts:
        print(f"[ERR] {folder}: Có {len(conflicts)} file đích đã tồn tại, không dám ghi đè:")
        for c in conflicts[:10]:
            print("   ", c)
        print("=> Hãy đổi --prefix/--suffix/--start hoặc xử lý các file đích trước.")
        return 0, start_num - 1

    # 5) Preview kế hoạch để người vận hành kiểm soát trước khi chạy thật.
    print(f"[INFO] {folder}: sẽ đổi {len(plan)} ảnh, từ số {start_num} đến {start_num + len(plan) - 1}")
    for old, new, oldn, newn in plan[:10]:
        print("   ", oldn, "->", newn)
    if len(plan) > 10:
        print("   ...")

    if dry_run:
        print(f"[DRY] {folder}: chỉ xem trước, không thực thi.")
        return 0, start_num + len(plan) - 1

    # 6) Backup labels trước khi cập nhật để đảm bảo có thể rollback.
    if labels_path:
        backup = backup_file(labels_path)
        print(f"[OK] Backup labels -> {backup}")

    # 7) Đổi tên file và cập nhật labels theo tên mới.
    renamed_pairs = []
    for old, new, oldn, newn in plan:
        if oldn == newn:
            # Trường hợp đã được đổi tên từ trước, bỏ qua rename nhưng vẫn ghi mapping.
            pass
        else:
            os.replace(old, new)

        renamed_pairs.append((oldn, newn))

        if labels_path:
            # Chỉ cập nhật labels nếu có key tương ứng trong file labels.
            if oldn in labels:
                labels[newn] = labels.pop(oldn)

    if labels_path:
        save_labels(labels_path, labels)
        print(f"[OK] Cập nhật labels: {labels_path}")

    # 8) Lưu mapping đổi tên để audit và hỗ trợ khôi phục khi cần.
    map_csv = folder / "rename_map.csv"
    with open(map_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["old_name", "new_name"])
        w.writerows(renamed_pairs)
    print(f"[OK] Lưu mapping -> {map_csv}")

    return len(renamed_pairs), start_num + len(renamed_pairs) - 1


def main():
    """Điểm vào chính: duyệt nhiều folder, gọi xử lý đổi tên theo từng folder."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Thư mục gốc chứa các folder con 1..N")
    ap.add_argument("--first", type=int, default=1, help="Folder bắt đầu (mặc định 1)")
    ap.add_argument("--last",  type=int, required=True, help="Folder kết thúc (vd 249)")
    ap.add_argument("--start", type=int, required=True, help="Số bắt đầu toàn cục (vd 2000)")
    ap.add_argument("--prefix", default="", help="Tiền tố tên mới (vd 'img_')")
    ap.add_argument("--suffix", default="", help="Hậu tố tên mới (vd '_samples')")
    ap.add_argument("--glob",   default="*", help="Glob lọc ảnh trong mỗi folder (vd '*.jpg' hoặc 'addr_*')")
    ap.add_argument(
        "--labels_name",
        default=None,
        help="Tên labels ưu tiên (vd 'labels.json' hoặc 'label.json'); nếu None sẽ tự dò",
    )
    ap.add_argument("--dry_run", action="store_true", help="Chỉ xem trước, không thực thi")
    args = ap.parse_args()

    # Xác thực thư mục gốc trước khi chạy batch.
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"[ERR] Không thấy thư mục gốc: {root}")

    current = args.start
    total_files = 0
    processed_folders = 0

    print(f"[INFO] Bắt đầu từ số {current}, duyệt folder {args.first}..{args.last}")
    for i in range(args.first, args.last + 1):
        folder = root / str(i)
        if not folder.is_dir():
            print(f"[WARN] Bỏ qua {folder} (không phải thư mục).")
            continue

        n, last_used = folder_rename_and_update(
            folder=folder,
            start_num=current,
            prefix=args.prefix,
            suffix=args.suffix,
            glob_pat=args.glob,
            preferred_labels_name=args.labels_name,
            dry_run=args.dry_run,
        )

        # Đếm folder đã xử lý để thống kê; dry-run cũng tính là đã xử lý.
        if n > 0 or args.dry_run:
            processed_folders += 1

        total_files += n
        current = last_used + 1

    print("-" * 60)
    print(f"[DONE] Đã xử lý {processed_folders} folder, đổi tên {total_files} ảnh.")
    print(f"[DONE] Số kế tiếp nếu tiếp tục: {current}")


if __name__ == "__main__":
    main()
