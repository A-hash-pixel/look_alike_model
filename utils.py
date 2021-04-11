import os
from time import time
from app import app, photos

def save_file(image):
    """saves an image locally"""
    if image.filename:
        image_name_split = image.filename.split(".")
        image.filename = image_name_split[0]+"_"+get_time()+"."+image_name_split[1] if image_name_split and len(image_name_split) > 0 else image.filename+"_"+get_time()
        print("image name ===================>"+image.filename)
        # filename = image.save(os.path.join(app.config['UPLOAD_FOLDER'], image.filename))
        filename = os.path.join(app.config['UPLOAD_FOLDER'], photos.save(image))
        # s3_bucket = make_bucket(os.getenv('S3_IMAGES_BUCKET'), 'public-read')
        if filename:
            return filename
        return None
    else:
        return None

def get_time():
    return str(int(time()*1000))