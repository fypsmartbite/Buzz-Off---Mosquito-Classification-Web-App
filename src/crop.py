import csv
from PIL import Image
import os


def crop_images(image_folder, csv_file):
    with open(csv_file, "r") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            filename = row["img_fName"]
            bbx_xtl = int(row["bbx_xtl"])
            bbx_ytl = int(row["bbx_ytl"])
            bbx_xbr = int(row["bbx_xbr"])
            bbx_ybr = int(row["bbx_ybr"])

            try:
                # Load the image
                image = Image.open(os.path.join(image_folder, filename))

                # Crop the image to the bounding box
                cropped_image = image.crop((bbx_xtl, bbx_ytl, bbx_xbr, bbx_ybr))

                # Save the cropped image
                cropped_image.save(os.path.join(image_folder, filename))
            except:
                continue


if __name__ == "__main__":
    image_folder = "dengue"
    csv_file = "phase2_train_v0.csv"

    crop_images(image_folder, csv_file)
