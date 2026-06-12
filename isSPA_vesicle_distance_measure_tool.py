import numpy as np
import mrcfile
import argparse


SHELL_WIDTH = 4
DENSITY = 1.0


def generate_cylinder_mrc(filename,shape,voxel_size,radius,z_range):
    zdim, ydim, xdim = shape
    data = np.zeros(shape, dtype=np.float32)

    x0, y0 = xdim // 2, ydim // 2
    z_min, z_max = z_range

    inner_radius = radius - SHELL_WIDTH // 2
    outer_radius = radius + SHELL_WIDTH // 2

    for z in range(z_min, z_max):
        for y in range(ydim):
            for x in range(xdim):
                r2 = (x - x0) ** 2 + (y - y0) ** 2
                if inner_radius ** 2 <= r2 <= outer_radius ** 2:
                    data[z, y, x] = DENSITY

    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = voxel_size
        mrc.update_header_from_data()

        mrc.header.cella.x = voxel_size * xdim
        mrc.header.cella.y = voxel_size * ydim
        mrc.header.cella.z = voxel_size * zdim

        mrc.header.mx = xdim
        mrc.header.my = ydim
        mrc.header.mz = zdim

        mrc.update_header_stats()

    print(f"MRC file '{filename}' generated successfully!")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate cylindrical MRC volume")

    parser.add_argument("--filename", type=str, required=True)

    parser.add_argument("--shape", type=int, nargs=3, required=True, help="z y x")

    parser.add_argument("--voxel_size", type=float, required=True)

    parser.add_argument("--radius", type=float, required=True)

    parser.add_argument("--z_range", type=int, nargs=2, required=True, help="z_min z_max")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    generate_cylinder_mrc(
        filename=args.filename,
        shape=tuple(args.shape),
        voxel_size=args.voxel_size,
        radius=args.radius,
        z_range=tuple(args.z_range)
    )