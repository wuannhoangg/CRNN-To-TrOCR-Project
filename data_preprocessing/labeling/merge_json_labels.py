#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gộp tất cả file nhãn (label.json) trong các folder con 1..N thành một file duy nhất.

Tính năng chính:
- Duyệt các thư mục con theo dải chỉ số [first, last] trong `root`.
- Đọc file nhãn trong từng thư mục và gộp vào một dict tổng.
- Tuỳ chọn chuẩn hoá key về basename (ví dụ: "abc.jpg" thay vì "path/to/abc.jpg").
- Xử lý va chạm key theo chiến lược `prefer`:
  - last: nhãn ở folder sau ghi đè.
  - first: giữ nhãn của folder đầu tiên.
  - error: dừng và báo lỗi nếu trùng key nhưng khác nhãn.

Đầu ra:
- Ghi file JSON hợp nhất theo `--out_json`, đảm bảo tạo thư mục cha nếu chưa tồn tại.
- In thống kê: số file nhãn đã đọc, tổng entries đã đọc, số entries sau gộp,
  số va chạm key và số trường hợp va chạm khác nhãn.

Ví dụ:
    python merge_labels.py --root /path/to/data --last 249 --out_json /path/to/labels.json --use_basename --prefer last
"""

import os, json, argparse
from pathlib import Path


def main():
    """Điểm vào chính: parse args, duyệt thư mục, gộp labels và lưu JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Thư mục gốc chứa các folder con 1..N")
    ap.add_argument("--first", type=int, default=1, help="Folder bắt đầu (mặc định 1)")
    ap.add_argument("--last",  type=int, required=True, help="Folder kết thúc (vd 249)")
    ap.add_argument("--labels_name", default="label.json", help="Tên file nhãn trong mỗi folder (mặc định: label.json)")
    ap.add_argument("--out_json", required=True, help="Đường dẫn file labels.json sau khi gộp")
    ap.add_argument("--use_basename", action="store_true", help="Chỉ lấy basename cho key (khuyên dùng)")
    ap.add_argument(
        "--prefer",
        choices=["last", "first", "error"],
        default="last",
        help="Chiến lược khi trùng key: last/first/error (mặc định last)",
    )
    args = ap.parse_args()

    # Xác thực thư mục gốc đầu vào.
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"[ERR] Không thấy thư mục gốc: {root}")

    # `merged`: dict kết quả cuối cùng (key -> label).
    merged = {}
    total_files = 0
    total_labels = 0
    collisions = 0
    collisions_diff = 0

    # Duyệt các thư mục theo index: first..last.
    for i in range(args.first, args.last + 1):
        folder = root / str(i)
        if not folder.is_dir():
            print(f"[WARN] Bỏ qua {folder} (không phải thư mục).")
            continue

        lbl = folder / args.labels_name
        if not lbl.exists():
            print(f"[WARN] Bỏ qua {folder} (không có {args.labels_name}).")
            continue

        # Đọc file JSON nhãn; nếu lỗi thì bỏ qua thư mục đó để không dừng toàn bộ quá trình.
        try:
            data = json.load(open(lbl, "r", encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Không đọc được {lbl}: {e}")
            continue

        count_in = 0
        for k, v in data.items():
            # Chuẩn hoá key theo basename nếu được yêu cầu (tránh khác path nhưng trùng tên file).
            key = os.path.basename(k) if args.use_basename else k

            # Xử lý trường hợp trùng key khi merge.
            if key in merged:
                collisions += 1
                if merged[key] != v:
                    collisions_diff += 1
                    if args.prefer == "error":
                        raise SystemExit(
                            f"[ERR] Trùng key khác nhãn: {key}\n"
                            f"  cũ={merged[key]}\n"
                            f"  mới={v}\n"
                            f"  folder={folder}"
                        )
                    elif args.prefer == "last":
                        merged[key] = v
                    elif args.prefer == "first":
                        pass
                else:
                    # Nếu trùng key nhưng nhãn giống nhau, không cần cập nhật.
                    pass
            else:
                merged[key] = v

            count_in += 1

        total_files += 1
        total_labels += count_in
        print(f"[OK] {folder}: +{count_in} nhãn")

    # Ghi file kết quả; tạo thư mục cha nếu chưa tồn tại.
    outp = Path(args.out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # Báo cáo thống kê để kiểm soát chất lượng dữ liệu sau khi gộp.
    print("-" * 60)
    print(f"[DONE] Gộp {total_files} file nhãn, tổng đọc {total_labels} entries")
    print(f"[DONE] Ghi ra: {outp} (tổng {len(merged)} entries)")
    print(f"[INFO] Va chạm key: {collisions} (khác nhãn: {collisions_diff}) với prefer='{args.prefer}'")


if __name__ == "__main__":
    main()
