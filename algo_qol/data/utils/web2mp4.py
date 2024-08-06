"""convert web2mp4
usage:
    python web2mp4.py SVD.webp svd.mp4
"""
import os
import time

import moviepy.video.io.ImageSequenceClip
import PIL


def analyseImage(path):
    '''
    Pre-process pass over the image to determine the mode (full or additive).
    Necessary as assessing single frames isn't reliable. Need to know the mode
    before processing all frames.
    '''
    im = PIL.Image.open(path)
    results = {
        'size': im.size,
        'mode': 'full',
    }
    try:
        while True:
            if im.tile:
                tile = im.tile[0]
                update_region = tile[1]
                update_region_dimensions = update_region[2:]
                if update_region_dimensions != im.size:
                    results['mode'] = 'partial'
                    break
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    return results


def processImage(path):
    '''
    Iterate the animated image extracting each frame.
    '''
    images = []
    mode = analyseImage(path)['mode']

    im = PIL.Image.open(path)

    i = 0
    p = im.getpalette()
    last_frame = im.convert('RGBA')

    try:
        while True:
            print("saving %s (%s) frame %d, %s %s" % (path, mode, i, im.size, im.tile))

            '''
            If the GIF uses local colour tables, each frame will have its own palette.
            If not, we need to apply the global palette to the new frame.
            '''
            if '.gif' in path:
                if not im.getpalette():
                    im.putpalette(p)

            new_frame = PIL.Image.new('RGBA', im.size)

            '''
            Is this file a "partial"-mode GIF where frames update a region of a different size to the entire image?
            If so, we need to construct the new frame by pasting it on top of the preceding frames.
            '''
            if mode == 'partial':
                new_frame.paste(last_frame)

            new_frame.paste(im, (0, 0), im.convert('RGBA'))
            nameoffile = path.split('/')[-1]
            output_folder = path.replace(nameoffile, '')

            name = '%s%s-%d.png' % (output_folder, ''.join(os.path.basename(path).split('.')[:-1]), i)
            print(name)
            new_frame.save(name, 'PNG')
            images.append(name)
            i += 1
            last_frame = new_frame
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    return images


def webp_mp4(input=r'C:/Users/N28828/Downloads/ComfyUI_00003_.webp', output='svd_car.mp4', fps=6):
    images = processImage("%s" % input)
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps=fps)
    clip.write_videofile(output)
    for item in images:
        try:
            os.remove(item)
        except:
            pass
    return


if __name__ == '__main__':
    import fire

    fire.Fire(webp_mp4)
