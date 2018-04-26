# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


cho = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"  # len = 19
jung = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"  # len = 21
# len = 27
jong = "ㄱ/ㄲ/ㄱㅅ/ㄴ/ㄴㅈ/ㄴㅎ/ㄷ/ㄹ/ㄹㄱ/ㄹㅁ/ㄹㅂ/ㄹㅅ/ㄹㅌ/ㄹㅍ/ㄹㅎ/ㅁ/ㅂ/ㅂㅅ/ㅅ/ㅆ/ㅇ/ㅈ/ㅊ/ㅋ/ㅌ/ㅍ/ㅎ".split(
    '/')
test = cho + jung + ''.join(jong)

hangul_length = len(cho) + len(jung) + len(jong)  # 67

mv_ac_dict = {}
num_han = 251

def is_valid_decomposition_atom(x):
    return x in test


def decompose(x):
    in_char = x
    if x < ord('가') or x > ord('힣'):
        return chr(x)
    x = x - ord('가')
    y = x // 28
    z = x % 28
    x = y // 21
    y = y % 21
    # if there is jong, then is z > 0. So z starts from 1 index.
    zz = jong[z - 1] if z > 0 else ''
    if x >= len(cho):
        print('Unknown Exception: ', in_char, chr(in_char), x, y, z, zz)
    return cho[x] + jung[y] + zz



def decompose_as_one_hot(in_char, warning=True):
    one_hot = []
    # print(ord('ㅣ'), chr(0xac00))
    # [0,66]: hangul / [67,194]: ASCII / [195,245]: hangul danja,danmo / [246,249]: special characters
    # Total 250 dimensions.
    if ord('가') <= in_char <= ord('힣'):  # 가:44032 , 힣: 55203
        x = in_char - 44032  # in_char - ord('가')
        y = x // 28
        z = x % 28
        x = y // 21
        y = y % 21
        # if there is jong, then is z > 0. So z starts from 1 index.
        zz = jong[z - 1] if z > 0 else ''
        if x >= len(cho):
            if warning:
                print('Unknown Exception: ', in_char,
                      chr(in_char), x, y, z, zz)

        one_hot.append(x)
        one_hot.append(len(cho) + y)
        if z > 0:
            one_hot.append(len(cho) + len(jung) + (z - 1))
        return one_hot
    else:
        if in_char < 128:
            result = hangul_length + in_char  # 67~
        elif ord('ㄱ') <= in_char <= ord('ㅣ'):
            # 194~ # [ㄱ:12593]~[ㅣ:12643] (len = 51)
            result = hangul_length + 128 + (in_char - 12593)
        elif in_char == ord('♡'):
            result = hangul_length + 128 + 51  # 245~ # ♡
        elif in_char == ord('♥'):
            result = hangul_length + 128 + 51 + 1  # ♥
        elif in_char == ord('★'):
            result = hangul_length + 128 + 51 + 2  # ★
        elif in_char == ord('☆'):
            result = hangul_length + 128 + 51 + 3  # ☆
        else:
            if warning:
                print('Unhandled character:', chr(in_char), in_char)
            # unknown character
            result = hangul_length + 128 + 51 + 4  # for unknown character

        return [result]


def decompose_str(string):
    return ''.join([decompose(ord(x)) for x in string])


def decompose_str_as_one_hot(string, warning=True):
    tmp_list = []
    for i in range(0, len(string)):
        x = string[i]
        if((x == 'm' and string[i + 1] == 'v') or (x == 'a' and string[i + 1] == 'c')):
            name = string[i:i+10]
            if name not in mv_ac_dict.keys():
                mv_ac_dict[name] = num_han + len(mv_ac_dict) + 1
            tmp_list.append(mv_ac_dict[name])
            i += 10
            continue
        da = decompose_as_one_hot(ord(x), warning=warning)
        tmp_list.extend(da)
    return tmp_list

def get_voca_num():
    return num_han + len(mv_ac_dict)

'''
※ 마스킹 정보

영화명과 배우명은 마스킹 처리되어 있습니다.
mv로 시작하는 경우 마스킹된 영화명, ac로 시작하는 경우 마스킹된 배우명입니다.

영화명 마스킹 예: mv00041958에 대한 좋은 이상향을 심어주는 영화
배우명 마스킹 예: ac00440758의 연기가 매우 인상적이었던 고전영화의 명작
'''