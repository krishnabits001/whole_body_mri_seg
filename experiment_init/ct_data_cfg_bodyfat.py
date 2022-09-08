import sys

def train_data(no_of_tr_imgs,comb_of_tr_imgs):
    #print('train data')
    if(no_of_tr_imgs=='tr3' and comb_of_tr_imgs=='c1'):
        labeled_id_list=["001","002","003"]
    elif(no_of_tr_imgs=='tr3' and comb_of_tr_imgs=='c2'):
        labeled_id_list=["002","003","004"]
    elif(no_of_tr_imgs=='tr3' and comb_of_tr_imgs=='c3'):
        labeled_id_list=["003","004","005"]
    elif(no_of_tr_imgs=='tr3' and comb_of_tr_imgs=='c4'):
        labeled_id_list=["001"]
    else:
        print('Error! Select valid combination of training images')
        sys.exit()
    return labeled_id_list

def val_data(no_of_tr_imgs,comb_of_tr_imgs):
    #print('val data')
    if(no_of_tr_imgs=='tr3' and comb_of_tr_imgs=='c1'):
        val_list=["004"]
    elif(no_of_tr_imgs=='tr3' and comb_of_tr_imgs=='c2'):
        val_list=["005"]
    elif(no_of_tr_imgs=='tr3' and comb_of_tr_imgs=='c3'):
        val_list=["001"]
    elif(no_of_tr_imgs=='tr3' and comb_of_tr_imgs=='c4'):
        val_list=["004"]
    else:
        print('Error! Select valid combination of val images')
        sys.exit()
    return val_list

def test_data(no_of_tr_imgs,comb_of_tr_imgs):
    #print('test data')
    if(no_of_tr_imgs=='tr3' and comb_of_tr_imgs=='c1'):
        test_list=["005"]
    elif(no_of_tr_imgs=='tr3' and comb_of_tr_imgs=='c2'):
        test_list=["001"]
    elif(no_of_tr_imgs=='tr3' and comb_of_tr_imgs=='c3'):
        test_list=["002"]
    elif(no_of_tr_imgs=='tr3' and (comb_of_tr_imgs=='c4' or comb_of_tr_imgs=='c5' or comb_of_tr_imgs=='c6' or comb_of_tr_imgs=='c7' or comb_of_tr_imgs=='c8' or comb_of_tr_imgs=='c9' or comb_of_tr_imgs=='c10')):
        test_list=["005"]
    else:
        print('Error! Select valid combination of test images')
        sys.exit()
    return test_list
