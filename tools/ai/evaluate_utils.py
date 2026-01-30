import numpy as np

class Calculator_For_mIoU:
    def __init__(self):

        self.class_names = ['background'] + ['change']
        self.classes = len(self.class_names)

        self.clear()

    def get_data(self, pred_mask, gt_mask):
        obj_mask = gt_mask>0
        correct_mask = (pred_mask==gt_mask) * obj_mask
        
        P_list, T_list, TP_list = [], [], []
        for i in range(self.classes):
            P_list.append(np.sum((pred_mask==i)*obj_mask))
            T_list.append(np.sum((gt_mask==i)*obj_mask))
            TP_list.append(np.sum((gt_mask==i)*correct_mask))

        return (P_list, T_list, TP_list)

    def add_using_data(self, data):
        P_list, T_list, TP_list = data
        for i in range(self.classes):
            self.P[i] += P_list[i]
            self.T[i] += T_list[i]
            self.TP[i] += TP_list[i]

    def add(self, pred_mask, gt_mask):
        obj_mask = gt_mask>=0
        correct_mask = (pred_mask==gt_mask) * obj_mask

        for i in range(self.classes):
            self.P[i] += np.sum((pred_mask==i)*obj_mask)
            self.T[i] += np.sum((gt_mask==i)*obj_mask)
            self.TP[i] += np.sum((gt_mask==i)*correct_mask)

    def get(self, detail=False, clear=True):
        IoU_dic = {}
        IoU_list = []

        FP_list = [] # over activation
        FN_list = [] # under activation

        for i in range(self.classes):
            IoU = self.TP[i]/(self.T[i]+self.P[i]-self.TP[i]+1e-10) * 100
            FP = (self.P[i]-self.TP[i])/(self.T[i] + self.P[i] - self.TP[i] + 1e-10)
            FN = (self.T[i]-self.TP[i])/(self.T[i] + self.P[i] - self.TP[i] + 1e-10)

            IoU_dic[self.class_names[i]] = IoU

            IoU_list.append(IoU)
            FP_list.append(FP)
            FN_list.append(FN)
        
        mIoU = np.mean(np.asarray(IoU_list))
        mIoU_foreground = np.mean(np.asarray(IoU_list)[1:])

        FP = np.mean(np.asarray(FP_list))
        FN = np.mean(np.asarray(FN_list))
        
        if clear:
            self.clear()
        
        if detail:
            return mIoU, mIoU_foreground, IoU_dic, FP, FN
        else:
            return mIoU, mIoU_foreground

    def clear(self):
        self.TP = []
        self.P = []
        self.T = []
        
        for _ in range(self.classes):
            self.TP.append(0)
            self.P.append(0)
            self.T.append(0)


class Calculator_For_F1:
    def __init__(self):
        self.class_names = ['background'] + ['change']
        self.classes = len(self.class_names)
        self.clear()

    def add(self, pred_mask, gt_mask):
        # 统计 TP, P(预测总数), T(标签总数)
        # 这里重点计算 index 1 (change)
        for i in range(self.classes):
            self.TP[i] += np.sum((pred_mask == i) & (gt_mask == i))
            self.P[i] += np.sum(pred_mask == i)
            self.T[i] += np.sum(gt_mask == i)

    def get(self, clear=True):
        # 计算 Change 类的 F1 (index 1)
        i = 1 
        precision = self.TP[i] / (self.P[i] + 1e-10)
        recall = self.TP[i] / (self.T[i] + 1e-10)
        
        f1 = (2 * precision * recall) / (precision + recall + 1e-10) * 100
        
        if clear:
            self.clear()
            
        return f1

    def clear(self):
        self.TP = [0] * self.classes
        self.P = [0] * self.classes
        self.T = [0] * self.classes