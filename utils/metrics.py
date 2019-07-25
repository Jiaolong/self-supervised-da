"""
This file will contain the metrics of the framework
"""
import numpy as np

class RunningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self, ignored_classes=None):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        overall_acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        #mean_acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

        classes = list(range(self.n_classes))
        if ignored_classes is not None:
            classes = list(set(classes) - set(ignored_classes))

        iu = [iu[i] for i in classes]
        mean_iu = np.nanmean(iu)
        #freq = hist.sum(axis=1) / hist.sum()
        #fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(classes, iu))

        return mean_iu, cls_iu, overall_acc, acc_cls

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class AverageMeter:
    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def val(self):
        return self.avg


class AverageMeterList:
    def __init__(self, num_cls):
        self.cls = num_cls
        self.value = [0] * self.cls
        self.avg = [0] * self.cls
        self.sum = [0] * self.cls
        self.count = [0] * self.cls
        self.reset()

    def reset(self):
        self.value = [0] * self.cls
        self.avg = [0] * self.cls
        self.sum = [0] * self.cls
        self.count = [0] * self.cls

    def update(self, val, n=1):
        for i in range(self.cls):
            self.value[i] = val[i]
            self.sum[i] += val[i] * n
            self.count[i] += n
            self.avg[i] = self.sum[i] / self.count[i]

    @property
    def val(self):
        return self.avg


class FPSMeter:
    """
    Class to measure frame per second in our networks
    """

    def __init__(self, batch_size):
        self.frame_per_second = 0.0
        self.f_in_milliseconds = 0.0

        self.frame_count = 0
        self.milliseconds = 0.0

        self.batch_size = batch_size

    def reset(self):
        self.frame_per_second = 0.0
        self.f_in_milliseconds = 0.0

        self.frame_count = 0

    def update(self, seconds):
        self.milliseconds += seconds * 1000
        self.frame_count += self.batch_size

        self.frame_per_second = self.frame_count / (self.milliseconds / 1000.0)
        self.f_in_milliseconds = self.milliseconds / self.frame_count

    @property
    def mspf(self):
        return self.f_in_milliseconds

    @property
    def fps(self):
        return self.frame_per_second

    def print_statistics(self):
        print("""
Statistics of the FPSMeter
Frame per second: {:.2f} fps
Milliseconds per frame: {:.2f} ms in one frame
These statistics are calculated based on
{:d} Frames and the whole taken time is {:.4f} Seconds
        """.format(self.frame_per_second, self.f_in_milliseconds, self.frame_count, self.milliseconds / 1000.0))
