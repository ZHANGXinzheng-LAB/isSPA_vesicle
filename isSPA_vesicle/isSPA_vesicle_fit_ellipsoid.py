#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import cv2
import mrcfile


def main():
    parser = argparse.ArgumentParser(description="Fit ellipses from mask.mrc")

    parser.add_argument("--input", "-i", required=True,help="input mask.mrc")
    parser.add_argument("--output_txt", "-ot", required=True,help="output txt file")
    parser.add_argument("--output_mrc", "-om", required=True,help="output fitted ellipse mask mrc")
    parser.add_argument("--min_area", type=float, default=100,help="minimum area threshold")

    args = parser.parse_args()

    with mrcfile.open(args.input, permissive=True) as mrc:
        mask = mrc.data.copy()
    
    mask = np.squeeze(mask)
    h, w = mask.shape

    results = fit_ellipses_from_mask(mask, args.min_area)

    with open(args.output_txt, "w") as f:

        for r in results:
            cx, cy = r["center"]
            major_r = r["major_axis"]/2
            minor_r = r["minor_axis"]/2
            angle = r["angle"]
            axis_ratio = r["axis_ratio"]
            shape_type = r["type"]
            
            line = (
                f"{args.input}\t"
                f"image_size={w},{h}\t"
                f"center={cx:.2f},{cy:.2f}\t"
                f"radius={major_r:.2f},{minor_r:.2f}\t"
                f"angle={angle:.2f}\t"
                f"axis_ratio={axis_ratio:.2f}\t"
                f"type={shape_type}\n"
            )
            
            f.write(line)

    fitted_mask = draw_ellipses_mask((h, w), results)

    with mrcfile.new(args.output_mrc, overwrite=True) as mrc:
        mrc.set_data(fitted_mask.astype(np.float32))

    print(f"[INFO] Done. Found {len(results)} objects.")


def fit_ellipses_from_mask(mask, min_area=100):
    
    mask = (mask > 0).astype(np.uint8)
    
    num_labels, labels = cv2.connectedComponents(mask)
    
    # 8连通，包括斜对角线。
    #num_labels, labels = cv2.connectedComponents(mask)

    results = []

    for label_id in range(1, num_labels):
        component = (labels == label_id).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if len(contours) == 0:
            continue

        cnt = contours[0]

        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        #椭圆拟合至少需要5个点
        if len(cnt) < 5:
            continue

        ellipse = cv2.fitEllipse(cnt)

        (x, y), (MA, ma), angle = ellipse
        
        # 保证 MA 是长轴
        if MA < ma:
            MA, ma = ma, MA
            angle += 90

        major_axis = MA
        minor_axis = ma

        ratio = major_axis / (minor_axis + 1e-8)
        
        shape_type = "circle" if ratio < 1.02 else "ellipse"

        results.append({
            "center": (float(x), float(y)),
            "major_axis": float(major_axis),
            "minor_axis": float(minor_axis),
            "axis_ratio": float(ratio),
            "angle": float(angle),
            "type": shape_type,
            "area": float(area),
            "label_id": int(label_id)
        })

    return results


def draw_ellipses_mask(shape, results):
    out = np.zeros(shape, dtype=np.float32)

    for r in results:
        center = tuple(map(int, r["center"]))
        axes = (
            int(r["major_axis"] / 2),
            int(r["minor_axis"] / 2)
        )
        angle = r["angle"] % 180 # 因为opencv不记录长短轴方向，0度和180度其实是同一个椭圆，不加也可。如果想统计什么方向多少个椭圆才有意义。
        
        print(r["major_axis"], r["minor_axis"], axes)
        
        cv2.ellipse(out, center, axes, angle, 0, 360, 1, -1)

    return out


if __name__ == "__main__":
    main()