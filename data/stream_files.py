import boto3
import tifffile
import tempfile
import os

s3 = boto3.client("s3")
bucket = "lin-2021-crc-atlas"
key = "data/WD-76845-106.ome.tif"

# 1️⃣ Download to temp file (streaming, no RAM explosion)
with tempfile.NamedTemporaryFile(delete=False, suffix=".ome.tif") as f:
    tmp_path = f.name
    #s3.download_fileobj(bucket, key, f)
tmp_path = "./CRC02-HE.ome.tif"
#try:
# 2️⃣ Open as OME-TIFF
with tifffile.TiffFile(tmp_path) as tif:
    print("Number of series:", len(tif.series))

    series = tif.series[0]
    print("Shape:", series.shape)
    print("Axes:", series.axes)

    # 3️⃣ Memory-map instead of full load
    img = series.asarray(out="memmap")

    print("Memmap dtype:", img.dtype)

    # Example: read small patch only
    patch = img[0, 0:512, 0:512]
    print("Patch shape:", patch.shape)

#finally:
#    os.remove(tmp_path)