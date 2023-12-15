import os
import os.path

import torch.utils.data as data
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

    classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
    image = [img for img in classes if img.find('frames_cleanpass') > -1]
    disp  = [dsp for dsp in classes if dsp.find('disparity') > -1]

    # monkaa_path = filepath + [x for x in image if 'monkaa' in x][0]
    # monkaa_disp = filepath + [x for x in disp if 'monkaa' in x][0]
    monkaa_path = "/data/tychang/SceneFlow_archive/Monkaa/frames_cleanpass"
    monkaa_disp = "/data/tychang/SceneFlow_archive/Monkaa/disparity"

    monkaa_dir  = os.listdir(monkaa_path)

    all_left_img=[]
    all_right_img=[]
    all_left_disp = []
    test_left_img=[]
    test_right_img=[]
    test_left_disp = []

    for dd in monkaa_dir:
        left_path = monkaa_path + '/' + dd + '/left/'
        right_path = monkaa_path + '/' + dd + '/right/'
        disp_path = monkaa_disp + '/' + dd + '/left/'
        rdisp_path = monkaa_disp + '/' + dd + '/right/'

        right_imgs = os.listdir(right_path)
        for img in right_imgs:
            img_path = os.path.join(right_path, img)
            if is_image_file(img_path):
                if img_path != '/data/tychang/SceneFlow_archive/Monkaa/frames_cleanpass/family_x2/right/0025.png':
                    all_right_img.append(img_path)
                    all_left_img.append(os.path.join(left_path, img))
                    all_left_disp.append(os.path.join(left_path, img).replace('frames_cleanpass', 'disparity').replace('.png', '.pfm'))

    # for dd in monkaa_dir:
    #   for im in os.listdir(monkaa_path+'/'+dd+'/left/'):
    #     if is_image_file(monkaa_path+'/'+dd+'/left/'+im):
    #       all_left_img.append(monkaa_path+'/'+dd+'/left/'+im)
    #     all_left_disp.append(monkaa_disp+'/'+dd+'/left/'+im.split(".")[0]+'.pfm')

    #   for im in os.listdir(monkaa_path+'/'+dd+'/right/'):
    #     if is_image_file(monkaa_path+'/'+dd+'/right/'+im):
    #       all_right_img.append(monkaa_path+'/'+dd+'/right/'+im)

    # flying_path = filepath + [x for x in image if x == 'frames_cleanpass'][0]
    # flying_disp = filepath + [x for x in disp if x == 'frames_disparity'][0]
    flying_path = "/data/tychang/SceneFlow_archive/FlyingThings3D/frames_cleanpass"
    flying_disp = "/data/tychang/SceneFlow_archive/FlyingThings3D/disparity"
    # flying_dir = flying_path+'/TRAIN/'
    # subdir = ['A','B','C']

    # for ss in subdir:
    #   flying = os.listdir(flying_dir+ss)

    #   for ff in flying:
    #     imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
    #     for im in imm_l:
    #       if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
    #         all_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)
          
    #       all_left_disp.append(flying_disp+'/TRAIN/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')

    #       if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
    #         all_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)

    # flying_dir = flying_path+'/TEST/'

    # subdir = ['A','B','C']

    # for ss in subdir:
    #   flying = os.listdir(flying_dir+ss)

    #   for ff in flying:
    #     imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
    #     for im in imm_l:
    #       if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
    #         test_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)
          
    #       test_left_disp.append(flying_disp+'/TEST/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')

    #       if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
    #         test_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)
    fimg_train = flying_path + '/TRAIN/'
    fimg_test = flying_path + '/TEST/'
    fdisp_train = flying_disp + '/TRAIN/'
    fdisp_test = flying_disp + '/TEST/'
    fsubdir = ['A', 'B', 'C']

    for dd in fsubdir:
        imgs_path = fimg_train + dd + '/'
        disps_path = fdisp_train + dd + '/'
        imgs = os.listdir(imgs_path)
        for cc in imgs:
            left_path = imgs_path + cc + '/left/'
            right_path = imgs_path + cc + '/right/'
            disp_path = disps_path + cc + '/left/'
            rdisp_path = disps_path + cc + '/right/'

            left_imgs = os.listdir(left_path)
            for img in left_imgs:
                img_path = os.path.join(left_path, img)
                if is_image_file(img_path):
                    all_left_img.append(img_path)
                    all_right_img.append(os.path.join(right_path, img))
                    all_left_disp.append(disp_path + img.split(".")[0] + '.pfm')

    for dd in fsubdir:
        imgs_path = fimg_test + dd + '/'
        disps_path = fdisp_test + dd + '/'
        imgs = os.listdir(imgs_path)
        for cc in imgs:
            left_path = imgs_path + cc + '/left/'
            right_path = imgs_path + cc + '/right/'
            disp_path = disps_path + cc + '/left/'
            rdisp_path = disps_path + cc + '/right/'

            left_imgs = os.listdir(left_path)
            for img in left_imgs:
                img_path = os.path.join(left_path, img)
                if is_image_file(img_path):
                    test_left_img.append(img_path)
                    test_right_img.append(os.path.join(right_path, img))
                    test_left_disp.append(disp_path + img.split(".")[0] + '.pfm')


    # driving_dir = filepath + [x for x in image if 'driving' in x][0] + '/'
    # driving_disp = filepath + [x for x in disp if 'driving' in x][0]
    driving_dir = "/data/tychang/SceneFlow_archive/Driving/frames_cleanpass/"
    driving_disp = "/data/tychang/SceneFlow_archive/Driving/disparity/"

    subdir1 = ['35mm_focallength','15mm_focallength']
    subdir2 = ['scene_backwards','scene_forwards']
    subdir3 = ['fast','slow']

    for i in subdir1:
      for j in subdir2:
        for k in subdir3:
            imm_l = os.listdir(driving_dir+i+'/'+j+'/'+k+'/left/')    
            for im in imm_l:
              if is_image_file(driving_dir+i+'/'+j+'/'+k+'/left/'+im):
                all_left_img.append(driving_dir+i+'/'+j+'/'+k+'/left/'+im)

              all_left_disp.append(driving_disp+'/'+i+'/'+j+'/'+k+'/left/'+im.split(".")[0]+'.pfm')

              if is_image_file(driving_dir+i+'/'+j+'/'+k+'/right/'+im):
                all_right_img.append(driving_dir+i+'/'+j+'/'+k+'/right/'+im)


    return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp


