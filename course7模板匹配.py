import cv2 as cv
import numpy as np

def template_demo():
    tpl = cv.imread('img\\yao.png')
    target = cv.imread('img\\nba.jpg')
    cv.imshow("target",target)
    #使用三种匹配方式，标准方差、标准相关、标准相关系数匹配
    medthods = [cv.TM_SQDIFF_NORMED,cv.TM_CCORR_NORMED,cv.TM_CCOEFF_NORMED]
    #模板的宽高
    th,tw = tpl.shape[:2]
    for md in medthods:
        result = cv.matchTemplate(target,tpl,md)
        min_val,max_val,min_loc,max_loc = cv.minMaxLoc(result)
        print(cv.minMaxLoc(result))
        # 如果方法是TM_SQDIFF或TM_SQDIFF_NORMED，则取最小值
        if md==cv.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0]+tw,tl[1]+th)
        cv.rectangle(target,tl,br,(0,255,0),2)
        cv.imshow("match-"+np.str(md),target)
        cv.imshow("match-"+np.str(md),result)


template_demo()
cv.waitKey(0)
cv.destroyAllWindows()