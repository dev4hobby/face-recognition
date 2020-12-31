from os.path import isfile
import math
from time import sleep

def bar_progress(current: int, total: int, width=80):
    '''
    wget의 progress를 커스텀 하기 위해 구현함.
    진행상황을 알려야 할 다른 곳에 사용해도 됨.
    '''
    width=28
    avail_dots = width - 2
    print(current, total)
    shaded_dots = int(math.floor(float(current) / total * avail_dots))
    percent_bar = '[{}{}]'.format('■' * shaded_dots, ' ' * (avail_dots - shaded_dots))
    progress = '{}% {} [{} / {}]\n'.format(
        round(current / total* 100, 1),
        percent_bar,
        current,
        total
    )
    return progress

def text_spary(text: str, interval=0.03):
    '''
    텍스트를 한 캐릭터 단위로 뿌려줌.
    출력 애니메이션을 위한 기능
    '''
    for character in text:
        print(character, end='', flush=True)
        sleep(interval)
    print()

def is_image_file(path: str) -> True:
    '''
    이미지 형식의 파일인지 확인하는 기능
    '''
    ext_list = ['bmp', 'jpg', 'jpeg', 'png']
    if path.split('.')[-1] and isfile(path):
        return True
    return False