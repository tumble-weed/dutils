import threading
# import model.utils 
ROOT_DIR = '/root/evaluate-saliency-4/GPNN/debug-results'
import os
import os
import glob
import skimage.io
import dutils
DEBUG_DIR = 'debugging'
SAVE_DIR = DEBUG_DIR
SYNC = False
SYNC_DIR = SAVE_DIR
REMOTE_SYNC_PARENT = "fong-invert"
IMAGENET_ROOT = "/root/bigfiles/dataset/imagenet"
IMAGENET_IMAGE_ROOT = os.path.join(IMAGENET_ROOT,'images','val')
PASCAL_ROOT = "/root/bigfiles/dataset/VOCdevkit/"
PASCAL_IMAGE_ROOT = os.path.join(PASCAL_ROOT,'VOC2007','JPEGImages')
#/root/bigfiles/dataset/VOCdevkit/VOC2007/JPEGImages
if not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
def sync_to_gdrive(foldername):
    folderbasename = os.path.basename(foldername.rstrip(os.path.sep))
    # oipdb('sync-gdrive')
    # import ipdb; ipdb.set_trace()
    print(f'Syncing {foldername} to {REMOTE_SYNC_PARENT}/{folderbasename}')
    os.system(f'rclone sync -Pv {foldername} aniketsinghresearch-gdrive:{REMOTE_SYNC_PARENT}/{folderbasename}')


def denormalize_tensor(t,vgg_mean=[0.485, 0.456, 0.406],
                     vgg_std=[0.229, 0.224, 0.225]):
    device = t.device
    # out = (t - torch.tensor(vgg_mean).to(device)[None,:,None,None])/torch.tensor(vgg_std).to(device)[None,:,None,None]
    out = (t * torch.tensor(vgg_std).to(device)[None,:,None,None]) + torch.tensor(vgg_mean).to(device)[None,:,None,None]
    return out
# img_save = lambda im,filename,ROOT_DIR=ROOT_DIR:model.utils.img_save(im,os.path.join(ROOT_DIR,filename))
'''
def img_save(im,filename,ROOT_DIR=ROOT_DIR):
    full_filename = os.path.join(ROOT_DIR,filename)
    print(f'saving in {full_filename}')
    model.utils.img_save(im,full_filename)
'''
def write_above_image_and_save(img,savename,text="some text",c='red'):
    # %matplotlib inline
    from skimage.draw import rectangle
    import numpy as np
    import matplotlib.pyplot as plt 
    # rr, cc = rectangle(start=(1, 1), extent=(1, 4))
    # img = np.zeros((5, 5, 3), dtype=np.int32)
    # img[rr, cc, 0] = 0
    # img[rr, cc, 1] = 255
    # img[rr, cc, 2] = 255
    plt.figure()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.axis('off')
    plt.tight_layout()
    plt.margins(0, 0)
    plt.imshow(img)
    plt.text(25, 25, text, dict(color=c, va='center', ha='right'),fontsize=10,fontweight='bold')

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())    
    plt.draw()
    # plt.savefig(savename)
    plt.gcf().canvas.draw()
    buf = plt.gcf().canvas.buffer_rgba()
    X = np.asarray(buf)
    # assert False
    plt.close()
    return X
    

from matplotlib import cm
def img_save(img, savename,ROOT_DIR=ROOT_DIR,cmap=None,save=True,return_img=False,use_matplotlib=True,syncable=False):
    saveroot = os.path.splitext(savename)[0]
    if not os.path.isabs(savename):
        savename = os.path.join(ROOT_DIR,savename)
    if save:
        print(colorful.tan(f"saving {os.path.abspath(savename)}"))
    '''
    An adaptive image saving method, that works for numpy arrays, torch tensors. Also can accomodate multiple shapes of the input as well as ranges of values    
    '''    
    if isinstance(img,torch.Tensor):
        img = tensor_to_numpy(img)
    
    # shape:
    print('img has shape: ',img.shape)
    if img.ndim == 4:
        print('got 4d input, assuming first channel is batch, saving the fist image')
        img = img[0]
    if img.ndim == 3:
        if img.shape[0] == 1:
            print('got input with 1 channel, assuming grayscale')
            img = img[0]
        elif img.shape[0] == 3:
            print('got input with 3 channels')
            img = np.transpose(img,(1,2,0))
    if img.min() >= 0 and img.max() <= 1:
        print('got img with values in [0,1] range')
    else:
        if img.min() <= -100 and img.max() >= 100:
            img = (img + 128)/255.
        print('TODO: figure out what to do with min < 0 and max> 1')
    if cmap is not None:
        if not (img.ndim == 2):
            print(f'ignoring {cmap} as image has 3 channels')
        else:
            img = cm.__dict__[cmap](img)
    if len(os.path.splitext(savename)[-1]) == 0:
        savename = savename + '.png'
    # skimage.io.imsave(savename,img)
    if use_matplotlib:
        img_with_text = write_above_image_and_save(img,savename,text=saveroot,c='red')
    else:
        img_with_text = img
    if save:
        skimage.io.imsave(savename,img_with_text)
    if SYNC:
        if syncable:
            sync_to_gdrive(SYNC_DIR)        
    if return_img:
        return img_with_text
def img_dict_save(im_dict,savename,ROOT_DIR=ROOT_DIR,cmap=None):
    savename = os.path.join(ROOT_DIR,savename)
    new_im_dict = {}
    for k,v in im_dict.items():
        v = img_save(v, k,ROOT_DIR=ROOT_DIR,cmap=cmap,save=False,return_img=True)
        new_im_dict[k] = v
        # img_save(v,os.path.join(ROOT_DIR,f'{savename}_{k}.png'),cmap=cmap)
    concatenated_image = np.concatenate(list(new_im_dict.values()),axis=1)
    skimage.io.imsave(savename,concatenated_image)
    print(colorful.tan(f"saving {os.path.abspath(savename)}"))    
    
# save_plot = lambda y,title,filename,ROOT_DIR=ROOT_DIR:model.utils.save_plot(y,title,os.path.join(ROOT_DIR,filename))

def array_info(*args):
    for ar in args:
        try:
            print(ar.__class__)
            print(ar.shape)
            print(ar.min())
            print(ar.max())
            print('*'*10)
        except Exception as e:
            print('exception')
    pass
'''
import builtins
import importlib
if 'original_import' not in builtins.__dict__:
    builtins.original_import = builtins.__import__
def autoreload(*args,**kwargs):
    module = builtins.original_import(*args,**kwargs)
    importlib.reload(module)
    return module 
builtins.__import__ = autoreload  
''' 
import ipdb
break_counts = [None]
def break_only_once(why=''):
    if not break_counts[0]:
        print(why)
        import ipdb;ipdb.set_trace()
        break_counts[0] = True
'''
def break_if(flag,reset=True):
    if os.environ.get(flag,False) == '1':
        import ipdb;ipdb.set_trace()
        os.environ[flag] = '0'
'''    
def break_if(flag,reset=True):
    # if os.environ.get(flag,False) == '1':
    if os.path.exists(flag):
        import ipdb;ipdb.set_trace()
        if os.stat(flag).st_size == 0:
            os.system(f'rm {flag}')
        else:
            print(f'WARNING:file {flag} not empty!!')
            import ipdb;ipdb.set_trace()
#==============================================================

ls = lambda start,like='*':glob.glob(os.path.join(start,like))
def pathinfo(p):
    print(p)
    i = 0
    while True:
        print('='*20)
        e = os.path.exists(p)
        print(f'{p},{e}')
        if os.path.isdir(p):
           os.system(f'ls {p} | head -n 10')
           import ipdb;ipdb.set_trace()

        p = os.path.split(p)[0]
        #i += 1
        e = False
        if  len(p)==0 or (p == os.path.sep) or e or i == 4:
            break
#  pathinfo('/root/bigfiles/other1')
#==============================================================
def dictinfo(d,tabs=0):
    desc = ''
    for k,v in d.items():
        if isinstance(v,dict):
            vinfo = dictinfo(v,tabs=tabs+1)
        else:
            #vinfo = str(v)
            vinfo = str(v.__class__)
        TAB = '\t'
        vinfo = f'{TAB*tabs}{vinfo}'
        desc += f'{k}:{vinfo}'
        desc += '\n'
    print(desc)

#==============================================================

import builtins            
import numpy as np
builtins.np = np
import os
builtins.os = os
import sys
builtins.sys = sys

import torch
builtins.torch = torch

from matplotlib import pyplot as plt
builtins.plt = plt

# from matplotlib import pyplot as plt
import colorful
builtins.colorful = colorful

# import collections
# builtins.torch = torch
#==============================================================
def cipdb(flag,val='1'):
    if os.environ.get(flag,False) == val:
        import ipdb; ipdb.set_trace()
    pass
#==============================================================
def save_plot(y,title,savename,x=None):
    print(colorful.tan(f"saving {os.path.realpath(savename)}"))
    plt.figure()
    if x is None:
        plt.plot(y)
    else:
        plt.plot(x,y)
    plt.title(title)
    plt.show()
    plt.savefig(savename)
    plt.close()
    
def save_plot2(y,title,basename,x=None,syncable=False):
    # prefix = dutils.get('save_prefix','')
    prefix = ''
    d = SAVE_DIR
    # import ipdb; ipdb.set_trace()
    if prefix:
        d = os.path.join(d,prefix)
    if not os.path.exists(d):
        os.makedirs(d)        
    savename = os.path.join(d,basename)
    save_plot(y,title,savename,x=x)
    if SYNC:
        if syncable:
            sync_to_gdrive(SYNC_DIR)
    return savename
def savefig(fig,basename):
   plt.figure(fig.number)
   plt.savefig(os.path.join(SAVE_DIR,basename+'.png'))
   plt.close()
   pass
def run_in_another_thread(f,args=[],debug=False):
    if debug:
        f(*args)
    else:
        save_thread = threading.Thread(target=f, args=args)
        save_thread.start()


class DebuggerDisableException(Exception):
    def __init__(self,message):
        super().__init__(message)

class Debug:
    def __init__(self,name):
        self.name = name
        self.enabled = False
        if os.environ.get(self.name,False) == '1':
            self.enabled = True
    def __enter__(self):
        if not self.enabled:
            # Disable the context manager by raising an exception
            raise DebuggerDisableException("Context manager is disabled")
            # return self.__exit__(None,None,None)

    def __exit__(self, exc_type, exc_value, traceback):
        # Handle the exception raised in __enter__
        if exc_type is DebuggerDisableException:
            print(f"Exception type: {exc_type}")
            print(f"Exception value: {exc_value}")
            print("Suppressing the exception")
            return True  # Suppress the exception
        else:
            print("Exiting the context manager normally")
