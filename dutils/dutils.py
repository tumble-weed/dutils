import threading
# import model.utils 
ROOT_DIR = '/root/evaluate-saliency-4/GPNN/debug-results'
import os
import os
import glob
import skimage.io
#import dutils
import inspect
import ipdb
import torchvision.utils as vutils
import torchvision
DEBUG_DIR = 'debugging'
SAVE_DIR = DEBUG_DIR
SYNC = False
SYNC_DIR = SAVE_DIR
REMOTE_SYNC_PARENT = "fong-invert"
IMAGENET_ROOT = "/root/bigfiles/dataset/imagenet"
IMAGENET_IMAGE_ROOT = os.path.join(IMAGENET_ROOT,'images','val')
PASCAL_ROOT = "/root/bigfiles/dataset/VOCdevkit/"
PASCAL_IMAGE_ROOT = os.path.join(PASCAL_ROOT,'VOC2007','JPEGImages')
#=====================================
ALWAYS = True
NEVER = False
TODO = None
UNTESTED = False
class TODO_class():
    pass
TODO_obj = TODO_class()
#=====================================
def remove_missing_args(kwargs):
    kwargs2 = {}
    for k,v in kwargs.items():
        if v != TODO_obj:
            kwargs2[k] = v
    return kwargs2
#=====================================
#/root/bigfiles/dataset/VOCdevkit/VOC2007/JPEGImages
if not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)
def tensor_to_numpy(t):
    if isinstance(t,torch.Tensor):
        t = t.detach().cpu().numpy()
    return t
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
def get_numpy_image(img,vmin=None,vmax=None,cmap=None):
    if isinstance(img,torch.Tensor):
        img = tensor_to_numpy(img)
    
    # shape:
    print('img has shape: ',img.shape)
    # 224,224
    # 224,224,3
    # 1,3,224,224
    # 1,1,224,224
    # numpy array
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
    #pause()
    if vmin is not None or vmax is not None:
        if vmin is  None:
            vmin = img.min()
        if vmax is  None:
            vmax = img.max()
        img = (img  - vmin)/(vmax - vmin)
        if any([
            np.isinf(img).any(),
            np.isnan(img).any()
            ]):
            dutils.pause()

    else:
        if img.min() >= 0 and img.max() <= 1:
            print('got img with values in [0,1] range')
        else:
            if img.min() <= -75 and img.max() >= 100:
                img = (img + 128)/255.
            print('TODO: figure out what to do with min < 0 and max> 1')
    if cmap is not None:
        if not (img.ndim == 2):
            print(f'ignoring {cmap} as image has 3 channels')
        else:
            img = cm.__dict__[cmap](img)

    return img
def imshow(imlike,cmap=None,close=True):
    im = get_numpy_image(imlike,cmap=cmap)
    plt.figure()
    plt.imshow(imlike)
    plt.show()
    if close:
        plt.close()
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
def get_img_save(savedir):
    def img_save2(img, savename,ROOT_DIR=savedir,vmin=None,vmax=None,cmap=None,save=True,return_img=False,use_matplotlib=True,syncable=False):
        return img_save(img, savename,ROOT_DIR=ROOT_DIR,vmin=None,vmax=None,cmap=None,save=True,return_img=False,use_matplotlib=True,syncable=False)
    return img_save2

def img_grid_save(img_t,savename,ROOT_DIR=ROOT_DIR,use_matplotlib=True,save=True,cmap=None):
    if not isinstance(img_t,torch.Tensor):
        img_t = torch.tensor(img_t)
    img_t_grid = vutils.make_grid(img_t)
    img_save(img_t_grid, savename,ROOT_DIR=ROOT_DIR,cmap=cmap,save=save,use_matplotlib=use_matplotlib)
    """
    saveroot = os.path.splitext(savename)[0]
    if not os.path.isabs(savename):
        savename = os.path.join(ROOT_DIR,savename)
    #===========================================
    print(colorful.tan(f"saving {savename}"))
    http_prefix = 'http://localhost:10000'
    #http_prefix = 'http://0.tcp.us-cal-1.ngrok.io:13553'

    as_url = http_prefix+ savename.split('/root')[-1]
    print(colorful.tan(f"saving {as_url}"))
    #===========================================

    #===========================================
    """
def img_save(img, savename,ROOT_DIR=ROOT_DIR,vmin=None,vmax=None,cmap=None,save=True,return_img=False,use_matplotlib=True,syncable=False):
    saveroot = os.path.splitext(savename)[0]
    if not os.path.isabs(savename):
        savename = os.path.join(ROOT_DIR,savename)
    """
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
    """
    save_all = hardcode(save_all=False)
    '''
    if save_all:
        if img.ndim == 2:
            if isinstance(img, torch.Tensor):
                # 224,224 --> 1,224,224
                img = img[None,...]
            elif isinstance(img,np.ndarray):
                img = img[None,...]
        elif img.ndim == 1:
            assert False,'img.ndim == 1'
    '''
    if img.ndim == 2:
        if isinstance(img, torch.Tensor):
            # 224,224 --> 1,224,224
            img = img[None,...]
        elif isinstance(img,np.ndarray):
            img = img[None,...]
    elif img.ndim == 1:
        assert False,'img.ndim == 1'
    # img (5,3,224,224)
    # img (5,224,224,3)
    # img (5,224,224)
    # img (224,224) --> (1,224,224)
    # img (3,224,224)?
    if img.ndim == 3 and img.shape[0] == 3:
        print(colorful.red(f'ambiguous, img_shape is {img.shape}. not sure if first channel is batch or rgb'))
    nimages = img.shape[0]
    savename0 = savename
    for i in range(nimages):
        imgi = get_numpy_image(img[i:i+1],cmap=cmap,vmin=vmin,vmax=vmax)
        savename = savename0
        if len(os.path.splitext(savename)[-1]) == 0:
            savename = savename + '.png'
        if save_all:
           # myimg.png --> myimg0.png,myimg1.png etc.
           # if nimages ==1 do we want myimg0 or do we just want mmyimg.png?
           if nimages == 1:
               pass
           else:
               parts = os.path.splitext(savename)
               # parts = ['myimg','png']
               parts[0] = parts[0] + str(i)
               # parts = ['myimg0','png']
               savename = parts[0] +  parts[1]

        # skimage.io.imsave(savename,img)
        savename = os.path.abspath(savename)
        if save:
            print(colorful.tan(f"saving {savename}"))
            #http_prefix = 'http://localhost:10000'
            #http_prefix = 'http://0.tcp.us-cal-1.ngrok.io:13553'

            #as_url = http_prefix+ savename.split('/root')[-1]
            #====================================================
            assert savename.startswith(os.environ['httproot'])
            as_url = os.environ['httpurl']+'/'+savename[len(os.environ['httproot']):]
            #====================================================
            print(colorful.tan(f"saving {as_url}"))

        imgi = imgi.astype(np.float32)
        if use_matplotlib:
            img_with_text = write_above_image_and_save(imgi,savename,text=saveroot,c='red')
        else:
            img_with_text = imgi
        if i == 0:
            img_with_text0 = img_with_text
        if save:
            os.makedirs(os.path.dirname(savename),exist_ok=True)
            skimage.io.imsave(savename,img_with_text)
            #print(colorful.salmon(savename))
            #pause()
        if not save_all:
            break
    if SYNC:
        if syncable:
            sync_to_gdrive(SYNC_DIR)        

    if return_img:
        return img_with_text0
    return savename
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
    #===========================================================
    print(colorful.tan(f"saving {os.path.realpath(savename)}"))
    # /root/debug/myplot.png
    # myplot.png
    # http://localhost:10000/debug/myplot.png
    savename = os.path.abspath(savename)
    # p46()
    assert savename.startswith(os.environ['httproot'])
    saveurl = os.environ['httpurl']+'/'+savename[len(os.environ['httproot']):]
    print(colorful.tan(f'saving {saveurl}'))
    #=====================================================
    plt.figure()
    if x is None:
        plt.plot(y)
    else:
        plt.plot(x,y)
    plt.title(title)
    plt.show()
    plt.savefig(savename)
    plt.close()
    return savename
def save_plot2(y,title,basename,x=None,syncable=False):
    # prefix = dutils.get('save_prefix','')
    prefix = ''
    if os.path.isabs(basename):
      savename = basename
    else:
        d = SAVE_DIR
        # import ipdb; ipdb.set_trace()
        if prefix:
            d = os.path.join(d,prefix)
        if not os.path.exists(d):
            os.makedirs(d)        
        savename = os.path.join(d,basename)
    if isinstance(y,torch.Tensor):
       y = tensor_to_numpy(y)
    if x is not None and isinstance(x,torch.Tensor):
       x = tensor_to_numpy(x)
    save_plot(y,title,savename,x=x)
    if SYNC:
        if syncable:
            sync_to_gdrive(SYNC_DIR)
    return savename
def savefig(fig,basename):
    plt.figure(fig.number)
    if os.path.isabs(basename):
        fname = basename
    else:
        fname = os.path.join(ROOT_DIR,basename+'.png')
    print(colorful.tan(fname))
    as_url = 'http://localhost:10000/'+ fname[len('/root'):]
    print(colorful.tan(as_url))
    plt.savefig(fname)
    plt.close()
    pass
def run_in_another_thread(f,args=[],debug=False):
    if debug or os.environ.get('DUTILS_DISABLE_RUN_THREAD',False) == '1':
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

def hardcode(**kwargs):
    k = kwargs.keys()
    v = list(kwargs.values())
    print(colorful.red('hardcoding'))
    print(colorful.red(kwargs))
    if len(v) == 1:
        return v[0]
    return v
def hack(name,default=False,env=None):
   state = default
   if env is not None:
      state = os.environ.get(env,'0') == '1'
   if state:
      frame = inspect.currentframe()
      info = f'{frame.f_back.f_code.co_filename}:{frame.f_back.f_code.co_name}:{frame.f_back.f_lineno}'
      print(colorful.red(f'HACK:')+ info + colorful.red(f'{name}'))
   return state
hack2 = hack
hardcode2 = hardcode
def note(message):
   frame = inspect.currentframe()
   info = f'{frame.f_back.f_code.co_filename}:{frame.f_back.f_code.co_name}:{frame.f_back.f_lineno}'
   print(colorful.yellow(f'HACK:') + info + colorful.yellow(f'{message}'))
def printgreen(message):
    print(colorful.green(message))
class Timer():
    def __init__(self,name,verbose=True):
        self.name= name
        self.verbose = verbose
        pass
    def __enter__(self):
        self.tic = time.time()
        return self
    def __exit__(self,*args,**kwargs):
        self.toc = time.time()
        self.elapsed = self.toc-self.tic
        if self.verbose:
           print(colorful.yellow(f'{self.name} took {self.elapsed}'))

pause = ipdb.set_trace
pauseonce = ipdb.set_trace
pausetodo = ipdb.set_trace
p45 = pause
p46 = pauseonce
p47 = pausetodo
def init():
    init_objects()
    init_modules()
def init_objects():
    builtins = globals()['__builtins__']
    ''' 
    essntial_dict = {
      'np':'numpy',
      'torch':'torch',
      'matplotlib':'matplotlib'
      'plt':'mapl
    }
    #'''
    #try:
    if True:
        #if 'np' not in globals():
        if True:
            if 'tensor_to_numpy' not in builtins:
                builtins['tensor_to_numpy'] = tensor_to_numpy
                print('tensor_to_numpy')
        obj_dict = {'pause2':'pause2',
'p45':'p45',
'p47':'p47',
'p46':'p46',
        }
        for obj in obj_dict:
            if obj not in builtins:
                builtins[obj] = globals()[obj_dict[obj]]
    #except ImportError:
    #    print("ImportError")
    #    import ipdb;ipdb.set_trace()
    #    pass

import importlib
def init_modules():
    builtins = globals()['__builtins__']
    ''' 
    essntial_dict = {
      'np':'numpy',
      'torch':'torch',
      'matplotlib':'matplotlib'
      'plt':'mapl
    }
    #'''
    try:
        #if 'np' not in globals():
        module_dict = {'IPython':'IPython'}
        if True:
            for usename,importname in module_dict.items():
                #! import module by name
                builtins[usename] = importlib.import_module(importname)
                
            if 'np' not in builtins:
                import numpy as np
                builtins['np'] = np
                print('np')
            if 'torch' not in builtins:
                import torch
                builtins['torch'] = torch
                print('torch')
            if 'plt' not in builtins:
                import matplotlib.plt as plt
                builtins['plt'] = plt
                print('plt')
            if 'skimage' not in builtins:
                import skimage
                builtins['skimage'] = skimage
                print('skimage')
            if 'Image' not in builtins:
                import PIL.Image as Image
                builtins['Image'] = Image
                print('Image')
            if 'tqdm' not in builtins:
                import tqdm
                builtins['tqdm'] = tqdm
                print('tqdm')
            if 'os' not in builtins:
                import os
                builtins['os'] = os
                print('os')
            if 'sys' not in builtins:
                import sys
                builtins['sys'] = sys
                print('sys')
            if 'ipdb' not in builtins:
                import ipdb
                builtins['ipdb'] = ipdb
                print('ipdb')
            if 'colorful' not in builtins:
                import colorful
                builtins['colorful'] = colorful
                print('colorful')
            if 'sns' not in builtins:
                import seaborn 
                builtins['sns'] = seaborn
                print('sns')
            if 'inspect' not in builtins:
                import inspect
                builtins['inspect'] = inspect
                print('inspect')
            if 'dutils' not in builtins:
                import dutils
                builtins['dutils'] = dutils
                print('dutils')
            if 'argparse' not in builtins:
                import argparse
                builtins['argparse'] = argparse
            if 'time' not in builtins:
                import time
                builtins['time'] = time
            if 'pickle' not in builtins:
                import pickle
                builtins['pickle'] = pickle
            if 'lzma' not in builtins:
                import lzma
                builtins['lzma'] = lzma
            if 'glob' not in builtins:
                import glob
                builtins['glob'] = glob
            if 'json' not in builtins:
                import json
                builtins['json'] = json
            if 'importlib' not in builtins:
                builtins['importlib'] = importlib
            #if 'p45' not in builtins:
            #    builtins['p45'] = p45
            #if 'p46' not in builtins:
            #    builtins['p46'] = p46
            #if 'p47' not in builtins:
            #    builtins['p47'] = p47

        print("numpy imported as np")
    except ImportError:
        print("ImportError")
        import ipdb;ipdb.set_trace()
        pass

# class Check():
#     def __init__(self, do=[],dont=[]):
#         pass
#     def do(self):
#         pass
#     def 
# dutils.checker  = Check()

# Example usage
#import_numpy_to_builtins()

# Now you can use np as if it were a built-in module
#print(np.array([1, 2, 3]))

class trunciter():
    def __init__(self,iterable,enabled=False,max_iter=None):
        self.iter0 = iter(iterable)
        self.enabled = enabled
        self.max_iter = max_iter
        self.i = 0
        pass
    def __next__(self):
        item = next(self.iter0)
        if self.enabled:
            if self.i >= self.max_iter:
                print(colorful.red(f'stopping iteration at {self.i}'))
                raise StopIteration
        self.i += 1
        return item
    def __iter__(self):
        return self
    def __len__(self):
        return len(self.iter0)
        pass
def get_caller_namespace(n_levels_above=2):
    frame = inspect.currentframe()
    f_required = frame.f_back
    for i in range(n_levels_above-1):
        f_required = f_required.f_back
    ns = f_required.f_locals
    #..............................................
    if 'self' in ns:
        
        ns['self_'] = ns['self']
        del ns['self']
    for k in ns:
        if k.startswith('__') and k.endswith('__'):
            del ns[k]
    #..............................................
    ns = argparse.Namespace(**ns)
    return ns
'''
class open(fname,mode):
    def __init__(self,fname,mode):
        self.fname = fname
        self.mode = mode
    def __enter__(self,*args,**kwargs):
        if 'w' in mode:

        else:

        pass
    def __exit__(self,*args,**kwargs):
        pass
'''     
import os
from IPython.core.debugger import Pdb

class ConditionalIPdb(Pdb):
    def set_trace(self, flag_env_var=None):
        # Check if the environment flag is set
        flag = os.environ.get(flag_env_var,False)
        if flag == '1':

            print(f'stopping as {flag_env_var} as set')
            '''
            #super().set_trace()
            self.reset()
            #self._set_stopinfo(None,None)
            pause()
            if self.stack and 0 <= self.curindex < len(self.stack):
                self.curframe = self.stack[self.curindex][0]
                self._set_stopinfo(self.curframe, None)
            else:
                self.curframe = None            
            self.interaction(None, None)            
            '''
            # https://github.com/ipython/ipython/blob/fd2cf18f8109637662faf70862a84594625b132a/IPython/core/debugger.py#L1120C5-L1120C53
            # dutils.pause()
            Pdb().set_trace(sys._getframe().f_back)
ipdb2 = ConditionalIPdb()
pause2 = ipdb2.set_trace
def in_limits(t,min=None,max=None):
    flag = True
    if min is not None:
        flag = flag and (t.min() >= min)
    if max is not None:
        flag = flag and  (t.max() <= max)
    return flag

class If(Pdb):
    def set_trace(self, cond,flag_env_var=None):
        # Check if the environment flag is set
        assert flag_env_var is not None,f'flag_env_var should not be {flag_env_var}'
        flag = os.environ.get(flag_env_var,False)
        if flag == '1':
            if not cond:
                print(f'stopping as {flag_env_var} as set')
                '''
                #super().set_trace()
                self.reset()
                #self._set_stopinfo(None,None)
                pause()
                if self.stack and 0 <= self.curindex < len(self.stack):
                    self.curframe = self.stack[self.curindex][0]
                    self._set_stopinfo(self.curframe, None)
                else:
                    self.curframe = None            
                self.interaction(None, None)            
                '''
                # https://github.com/ipython/ipython/blob/fd2cf18f8109637662faf70862a84594625b132a/IPython/core/debugger.py#L1120C5-L1120C53
                # dutils.pause()
                #! how to set up a bp in the previous function
                Pdb().set_trace(sys._getframe().f_back)
        return cond
if_ = If().set_trace
def isbad(a):
    if isinstance(a,torch.Tensor):
        return a.isnan().any() or a.isinf().any()
    else:
        return np.isnan(a).any() or np.isinf(a).any()
class reach():
    def __init__(self,env):
        self.env = env
        self.reached = False
        pass
    def __enter__(self,*args,**kwargs):
        return self
    def __call__(self):
        self.reached = True
    def __exit__(self,*args,**kwargs):
        if os.environ.get(self.env,False) == '1' and not self.reached:
            p45()
        pass
import lzma
import pickle
def read_torchray_result(pklname,savename='saliency.png'):
    #print(pklname)
    #return 0
    with lzma.open(pklname,'rb') as f:
        loaded = pickle.load(f)
    saliency = loaded['saliency']
    img_save(saliency,savename)
#def import_by_filepath(file_path,module_name):
#    import importlib.util
#
## Get the absolute path of the file
#    file_path = os.path.abspath(file_path)
#
## Create a module specification from the file location
#    spec = importlib.util.spec_from_file_location(module_name, file_path)
#
## Create a new module based on the specification
#    module = importlib.util.module_from_spec(spec)
#    spec.loader.exec_module(module)    
## Execute the module to load it
#    #exec(compile(spec.loader.get_source(file_path), file_path, 'exec'), module.__dict__)
#    sys.modules[module_name] = module
#    return module


def import_by_filepath(file_path,module_name):
    from pydoc import importfile
    module = importfile(file_path)
    return module
def simple_normalize(t):
    m = t.min()
    M = t.max()
    fuzz = 0
    if m == M:
        fuzz = 1e-8
    tn = (t - t.min())/(t.max() - t.min() + fuzz)
    return tn

def verboseiter(**kwargs):
    name = list(kwargs.keys())[0]
    iterable = list(kwargs.values())[0]
    # for name,iterable in kwargs.items():        
    #     print(f'{name}:{value}')
    
    class MyIter():
        def __init__(self,name,iterable):
            self.iterable = iterable
            self.name = name
        def __iter__(self):
            self.iter_ = iter(self.iterable)
            return self
        def __next__(self):
            val = next(self.iter_)
            print(f'{self.name}:{val}')
            return val
    return MyIter(name,iterable)

def get_imagenet_model_and_transform():
    from torchvision.models import resnet50
    model = resnet50(pretrained=True)
    model.eval()
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean,std=std),
    ])
    return model,transform
def listset(*args):
    return list(set(*args))
