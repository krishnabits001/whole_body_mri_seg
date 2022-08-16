import sys

def train_data(no_of_tr_imgs,comb_of_tr_imgs):
    #print('train data')
    if(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c1'):
        labeled_id_list=["001","004","007","010","011","005","006","018"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c2'):
        labeled_id_list=["045","038","025","028","027","023","020"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c3'):
        labeled_id_list=["001"]#,"004","007","018","010"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c4'):
        labeled_id_list=["001","011","015","019","023","025","038"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c5'):
        labeled_id_list=["001","019"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c6'):
        labeled_id_list=["023","038"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c7'):
        labeled_id_list=["001"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c8'):
        labeled_id_list=["019"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c9'):
        labeled_id_list=["018"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c10'):
        labeled_id_list=["025"]
    else:
        print('Error! Select valid combination of training images')
        sys.exit()
    return labeled_id_list

def val_data(no_of_tr_imgs,comb_of_tr_imgs):
    #print('val data')
    if(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c1'):
        val_list=["019","021","013","015","016"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c2'):
        val_list=["005","006","001","004","010"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c3'):
        val_list=["005"]#,"006"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c4'):
        val_list=["005","006","018","021"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c5'):
        val_list=["005","021"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c6'):
        val_list=["006","018"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c7'):
        val_list=["005"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c8'):
        val_list=["021"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c9'):
        val_list=["023"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c10'):
        val_list=["038"]
    else:
        print('Error! Select valid combination of val images')
        sys.exit()
    return val_list

def test_data(no_of_tr_imgs,comb_of_tr_imgs):
    #print('test data')
    if(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c1'):
        test_list=["020","023","025","027","028","038","045"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c2'):
        test_list=["007","011","013","015","016","018","019","021"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c3'):
        test_list=["004"]#,"005"]
    elif(no_of_tr_imgs=='tr8' and (comb_of_tr_imgs=='c4' or comb_of_tr_imgs=='c5' or comb_of_tr_imgs=='c6' or comb_of_tr_imgs=='c7' or comb_of_tr_imgs=='c8' or comb_of_tr_imgs=='c9' or comb_of_tr_imgs=='c10')):
        test_list=["004","007","010","016","020","027","028","045"]
    else:
        print('Error! Select valid combination of test images')
        sys.exit()
    return test_list
