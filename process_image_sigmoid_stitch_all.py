import numpy as np
import os,shutil
import cv2  ####pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple

def calWeight(d,k):
    x = np.arange(-d/2,d/2)
    y = 1/(1+np.exp(-k*x))  ###sigmoid
    return y
 
def imgFusion(img1,img2,overlap,w,left_right=True):
    if left_right:  # 左右融合
        row, col = img1.shape[:2]
        col_right = img2.shape[1]
        img_new = np.zeros((row,col+col_right-overlap, 3), dtype=np.uint8)
        img_new[:, :col, :] = img1.copy()
        w_expand = np.expand_dims(np.tile(w,(row,1)), axis=2) # 权重扩增
        
        img_new[:, col-overlap:col, :] = (1-w_expand)*img1[:, col-overlap:col, :] + w_expand*img2[:, :overlap, :]
        img_new[:, col:, :] = img2[:,overlap:, :]

    else:   # 上下融合
        row, col = img1.shape[:2]
        row_bottom = img2.shape[0]
        img_new = np.zeros((row+row_bottom-overlap, col, 3), dtype=np.uint8)
        img_new[:row, :, :] = img1
        w = np.reshape(w,(overlap,1))
        w_expand = np.expand_dims(np.tile(w,(1,col)), axis=2) # 权重扩增
        img_new[row-overlap:row, :, :] = (1-w_expand)*img1[row-overlap:row, :, :]+w_expand*img2[:overlap, :, :]
        img_new[row:, :, :] = img2[overlap:, :, :]
    return img_new

def connect_left_right_images(imglist, overlap_x, w_x):
    '''
    左右拼接, 近景摄像头在水平方向的,采集的多张图片，拼接成一张图
    Parameters:
        imglist -      近景摄像头在水平方向拍摄的多张局部图片列表, 例如[img1, img2, img3], 列表里的元素是np.array
        overlap_x -    左右两张图片重叠区域的长度
        w_x -          重叠区域融合的权重
    Returns:
        拼接后的图, 数据格式是np.array
    '''
    if len(imglist)<2:
        print('左右拼接时, 请提供至少2张以上的图片输入')

    row_imga = imglist[0].copy()
    for col in range(1, len(imglist)):
        imgb = imglist[col]
        row_imga = imgFusion(row_imga.copy(), imgb, overlap_x, w_x, left_right=True)
    
    return row_imga

def connect_top_down_images(imglist, overlap_y, w_y):
    '''
    上下拼接, 近景摄像头在竖直方向的,采集的多张图片，拼接成一张图
    Parameters:
        imglist -     近景摄像头在竖直方向拍摄的多张局部图片列表, 例如[img1, img2, img3], 列表里的元素是np.array
        overlap_y -    上下两张图片重叠区域的长度
        w_y -          重叠区域融合的权重
    Returns:
        拼接后的图, 数据格式是np.array
    '''
    if len(imglist)<2:
        print('上下拼接时, 请提供至少2张以上的图片输入')

    col_imga = imglist[0].copy()
    for row in range(1, len(imglist)):
        imgb = imglist[row]
        col_imga = imgFusion(col_imga.copy(), imgb, overlap_y, w_y, left_right=False)
    
    return col_imga

def connect_images(partial_image_list, pix_point_list, pt_xy_index, step_x=50, step_y=50):
    '''
    上下左右拼接, 近景摄像头在水平方向和竖直方向的采集的多张图片，拼接成一张图
    Parameters:
        partial_image_list -     近景摄像头在竖直方向拍摄的多张局部图片列表, 列表里的元素是一个列表，在这个列表里的的元素是左右拼接的图像列表,
            例如[[img1, img2], [img3, img4]]，说明需要上下左右拼接；[[img1, img2]]只需要左右拼接；[[img1, ], [img2, ]]只需要上下拼接
        pix_point_list -       在partial_image_list里面的某一张图片, 图片的位置由pt_xy_index决定, 它的数据格式是numpy格式列表np.array([[x1,y1],[x2,y2]])
        pt_xy_index  -         记录pix_point_list点集是在partial_image_list里的哪一张图片,也就是图片位置,第几行第几列, 它的数据格式是tuple,例如(0,1), 表示在partial_image_list里的第0行第1列的图片
        step_x -               近景摄像头在水平方向的滑动步长,建议取值50,这时候摄像头在水平方向,每移动50步采集一张图像
        step_y -               近景摄像头在竖直方向的滑动步长,建议取值50,这时候摄像头在竖直方向,每移动50步采集一张图像
    Returns:
        拼接后的图和pix_point_list
    '''
    if not isinstance(partial_image_list, list):
        print("输入参数不合法, 第1个参数输入, 不是列表形式的")
        return None, pix_point_list
    
    num_top_down = len(partial_image_list)   ###上下拼接的图片数量
    if num_top_down>0:
        if not isinstance(partial_image_list[0], list):
            print("输入参数不合法, 第1个参数输入的列表里面的元素不是列表")
            return None, pix_point_list
        if not isinstance(partial_image_list[0][0], np.ndarray):
            print("输入参数不合法, 第1个参数输入的列表里面的列表里面的元素不是np.array形式的")
            return None, pix_point_list
        
        imgh, imgw = partial_image_list[0][0].shape[:2]
        overlap_x = imgw - int(step_x*20)
        overlap_y = imgh - int(step_y*20)
        w_x = calWeight(overlap_x, 0.05)
        w_y = calWeight(overlap_y, 0.05)

        if num_top_down>1:   ###需要上下拼接
            num_left_right = len(partial_image_list[0])  ###左右拼接的图片数量
            if num_left_right>1: ###需要左右拼接
                row_imga = connect_left_right_images(partial_image_list[0], overlap_x, w_x) ###左右拼接第1行
                for row in range(1, num_top_down):
                    row_imgb = connect_left_right_images(partial_image_list[row], overlap_x, w_x)  ###左右拼接第row行
                    row_imga = imgFusion(row_imga.copy(), row_imgb, overlap_y, w_y, left_right=False)  ###上下拼接前row行
                
                if pt_xy_index[1]>0:
                    stitch_pt_img_w = (pt_xy_index[1]+1)*imgw - pt_xy_index[1]*overlap_x
                    dist_right = imgw - pix_point_list[:, 0]
                    pix_point_list[:, 0] = stitch_pt_img_w - dist_right
                if pt_xy_index[0]>0:
                    stitch_pt_img_h = (pt_xy_index[0]+1)*imgh - pt_xy_index[0]*overlap_y
                    dist_down = imgh - pix_point_list[:, 1]
                    pix_point_list[:, 1] = stitch_pt_img_h - dist_down

                return row_imga, pix_point_list
            elif num_left_right==1:  ###只需要上下拼接，没有左右拼接的
                top_down_imglist = [x[0] for x in partial_image_list]
                dstimg = connect_top_down_images(top_down_imglist, overlap_y, w_y)

                if pt_xy_index[0]>0:
                    stitch_pt_img_h = (pt_xy_index[0]+1)*imgh - pt_xy_index[0]*overlap_y
                    dist_down = imgh - pix_point_list[:, 1]
                    pix_point_list[:, 1] = stitch_pt_img_h - dist_down

                return dstimg, pix_point_list
            else:
                print("需要上下拼接时, 水平方向, 请输入至少2张以上的图片!!!")
                return None, pix_point_list
        else:  ###没有上下拼接的
            num_left_right = len(partial_image_list[0])  ###左右拼接的图片数量
            if num_left_right>1: ###需要左右拼接
                row_imga = connect_left_right_images(partial_image_list[0], overlap_x, w_x) ###左右拼接第1行

                if pt_xy_index[1]>0:
                    stitch_pt_img_w = (pt_xy_index[1]+1)*imgw - pt_xy_index[1]*overlap_x
                    dist_right = imgw - pix_point_list[:, 0]
                    pix_point_list[:, 0] = stitch_pt_img_w - dist_right

                return row_imga, pix_point_list
            else:
                print("没有上下拼接时, 水平方向, 请输入至少2张以上的图片!!!")
                return None, pix_point_list
    else:
        print("输入的图片列表是空的")
        return None, pix_point_list


if __name__=='__main__':
    step_x, step_y = 50, 50
    imgroot = 'testdata/shaozi/data_'+str(step_x)+'x'+str(step_y)  ###存放待拼接图片的文件夹路径
    print('step_x, step_y =', step_x, step_y)

    left_right_imgpaths = ["(350,250).png", "(400,250).png", "(450,250).png", "(500,250).png"]   ###水平方向的待拼接图片名称的列表
    top_down_imgpaths = ["(400,100).png", "(400,150).png", "(400,200).png", "(400,250).png"]    ###竖直方向的待拼接图片名称的列表
    left_righttop_down_imgpaths = [["(350,150).png", "(400,150).png", "(450,150).png"],
                                   ["(350,200).png", "(400,200).png", "(450,200).png"],
                                   ["(350,250).png", "(400,250).png", "(450,250).png"]]
    
    partial_image_list = []
    
    # left_right_imglist = []    ####水平方向拼接的
    # for impath in left_right_imgpaths:
    #     srcimg = cv2.imread(os.path.join(imgroot,impath))
    #     left_right_imglist.append(srcimg)
    # partial_image_list.append(left_right_imglist)

    # for impath in top_down_imgpaths:     ###竖直方向拼接的
    #     srcimg = cv2.imread(os.path.join(imgroot,impath))
    #     partial_image_list.append([srcimg])

    for impaths in left_righttop_down_imgpaths:   ###水平+竖直方向拼接的
        left_right_imglist = []
        for impath in impaths:
            srcimg = cv2.imread(os.path.join(imgroot,impath))
            left_right_imglist.append(srcimg)
        partial_image_list.append(left_right_imglist)

    imgh, imgw = partial_image_list[0][0].shape[:2]
    pix_point_list = np.array([[int(imgw*0.5), int(imgh*0.5)], [int(imgw*0.3), int(imgh*0.3)], [int(imgw*0.7), int(imgh*0.3)],[int(imgw*0.7), int(imgh*0.7)], [int(imgw*0.3), int(imgh*0.7)]], dtype = np.int32)
    pt_xy_index = (1,2)

    part_img = partial_image_list[pt_xy_index[0]][pt_xy_index[1]].copy()
    for i in range(pix_point_list.shape[0]):
            cv2.circle(part_img, tuple(pix_point_list[i,:]), 20, (0,0,255), thickness=-1)
    cv2.polylines(part_img, [pix_point_list[1:,:]], True, (0,255,0), 5)

    dstimg, pix_point_list = connect_images(partial_image_list, pix_point_list, pt_xy_index, step_x=step_x, step_y=step_y)

    if dstimg is not None:
        for i in range(pix_point_list.shape[0]):
            cv2.circle(dstimg, tuple(pix_point_list[i,:]), 20, (0,0,255), thickness=-1)
        cv2.polylines(dstimg, [pix_point_list[1:,:]], True, (0,255,0), 5)

        cv2.imwrite('part_img.jpg', part_img)
        cv2.imwrite('result.jpg', dstimg)